# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Publish the rendered library to S3-compatible object storage, and check what is there.

The library is too large to commit, so it lives in a bucket and the web app is pointed at that
bucket's public base URL. The addressing contract is deliberately trivial: an object's key is the
path ``index.json`` already states, under an optional prefix, so a consumer builds a URL as
``{base}/{path-from-the-index}`` and needs no translation logic of any kind.

Three entrypoints are the source of truth for what belongs in the library: ``index.json`` for the
sweeps, ``tutorials/manifest.json`` for the worked examples and ``alchemy/manifest.json`` for the
post-processor evidence, each the file the web app fetches first for its part of the tree. Anything
on disk that none of them references is not published, and anything in the bucket that none of
them references is an orphan. Sidecars are published along with the images because the index names one
per stop: a bucket where those paths 404 would make the index a liar, and the whole set of them is
a rounding error beside the images.

Change is detected by content: a single-part S3 upload's ETag is the object's MD5, and every file
here is far below the multipart threshold, so the ETag the bucket reports is compared against the
local digest. Objects are written with ``put_object`` rather than the transfer manager expressly
to keep that true.

Critical public members:

- [`S3Settings`][param_sweeps.publish.S3Settings]: the bucket's connection details.
- [`load_s3_settings`][param_sweeps.publish.load_s3_settings]: the environment and ``.env`` lookup.
- [`enumerate_library`][param_sweeps.publish.enumerate_library]: every file the index references.
- [`plan_sync`][param_sweeps.publish.plan_sync]: what an upload would do, decided without any I/O
  against the bucket.
- [`SweepLibraryBucket`][param_sweeps.publish.SweepLibraryBucket]: the bucket operations themselves.
- [`verify_remote`][param_sweeps.publish.verify_remote] and
  [`verify_local`][param_sweeps.publish.verify_local]: the two checks behind the ``verify`` command.

``boto3`` is an optional dependency, imported where a client is actually built rather than at the
top of this module. Everything above the bucket itself, including planning a whole sync, is pure
computation over the index and the files on disk, so it runs in an environment that has no
publishing stack at all: rendering happens in a GPU environment that has no business carrying an
object-storage SDK, and importing this module must not put one there.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from enum import auto
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field
from strenum import StrEnum

from param_sweeps.alchemy import ALCHEMY_DIRECTORY_NAME, AlchemyManifest
from param_sweeps.alchemy import MANIFEST_FILENAME as ALCHEMY_MANIFEST_FILENAME
from param_sweeps.flows import MANIFEST_FILENAME as TUTORIAL_MANIFEST_FILENAME
from param_sweeps.flows import TUTORIALS_DIRECTORY_NAME, TutorialManifest
from param_sweeps.horde_client import ENV_ASSIGNMENT_PATTERN
from param_sweeps.outputs import INDEX_FILENAME, SweepIndex
from param_sweeps.promo import CARDS_MANIFEST_FILENAME, CardsManifest

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

FROZEN_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid")

HASH_CHUNK_BYTES = 1024 * 1024
"""How much of a file is digested at a time; the images are a few hundred kilobytes each."""

DEFAULT_UPLOAD_WORKERS = 8
"""How many objects are put at once. Each is a single small request, so the wait is latency, not bandwidth."""

TUTORIAL_MANIFEST_PATH = f"{TUTORIALS_DIRECTORY_NAME}/{TUTORIAL_MANIFEST_FILENAME}"
"""Where the tutorial manifest sits, relative to the library root."""

ALCHEMY_MANIFEST_PATH = f"{ALCHEMY_DIRECTORY_NAME}/{ALCHEMY_MANIFEST_FILENAME}"
"""Where the alchemy manifest sits, relative to the library root."""

PROMO_MANIFEST_PATH = CARDS_MANIFEST_FILENAME
"""Where the share-card manifest sits, relative to the promo tree's root."""

PROMO_GALLERY_MANIFEST_PATH = "gallery/gallery.json"
"""Where the promo gallery's manifest sits, relative to the promo tree's root, when one was made."""

ENTRYPOINT_PATHS: frozenset[str] = frozenset(
    {INDEX_FILENAME, TUTORIAL_MANIFEST_PATH, ALCHEMY_MANIFEST_PATH, PROMO_MANIFEST_PATH, PROMO_GALLERY_MANIFEST_PATH},
)
"""The files a consumer fetches first: held briefly by caches and uploaded after the files they name."""


class PublishEnvironment:
    """The environment variables that configure the bucket.

    They are read from the process environment first and from a ``.env`` second, so a shell that
    already exports them never needs a file. The secret is only ever handed to the S3 client: it
    is not logged, printed or written into any output.
    """

    ENDPOINT_URL = "PARAM_SWEEPS_S3_ENDPOINT_URL"
    BUCKET = "PARAM_SWEEPS_S3_BUCKET"
    ACCESS_KEY_ID = "PARAM_SWEEPS_S3_ACCESS_KEY_ID"
    SECRET_ACCESS_KEY = "PARAM_SWEEPS_S3_SECRET_ACCESS_KEY"
    PREFIX = "PARAM_SWEEPS_S3_PREFIX"
    REGION = "PARAM_SWEEPS_S3_REGION"


class CacheControl:
    """The ``Cache-Control`` headers written with each kind of object.

    Images are not truly immutable: inserting a stop into a strip renumbers the ones after it, so
    a path can come to hold a different image. A week is long enough that a consumer refetches
    almost nothing between publishes and short enough that a renumbering settles on its own; a
    publish that renumbers and needs to be seen immediately wants a cache purge at the CDN, not a
    shorter header here.

    ``index.json`` and the two manifests are the mutable entrypoints every consumer starts from,
    so they are held briefly.
    """

    ASSET = "public, max-age=604800, stale-while-revalidate=86400"
    INDEX = "public, max-age=300, must-revalidate"


class PublishTree(StrEnum):
    """Which tree a publishing command works on.

    The evidence library and the promotional cards cut from it are published apart, under
    prefixes of their own, so that the library's prefix holds only what a page cites as evidence.
    """

    SWEEPS = "sweeps"
    """The evidence library: the index's closure and the tutorial and alchemy manifests'."""

    PROMO = "promo"
    """The share cards and posters, and the manifest that names them."""


DEFAULT_TREE_PREFIXES: dict[PublishTree, str | None] = {PublishTree.SWEEPS: None, PublishTree.PROMO: "promo"}
"""The key prefix each tree publishes under when none is given; None reads the configured one."""


class ContentType:
    """The ``Content-Type`` written with each kind of object, by file extension."""

    WEBP = "image/webp"
    PNG = "image/png"
    JPEG = "image/jpeg"
    SVG = "image/svg+xml"
    JSON = "application/json"
    TEXT = "text/plain; charset=utf-8"
    FALLBACK = "application/octet-stream"

    BY_SUFFIX: ClassVar[dict[str, str]] = {
        ".webp": WEBP,
        ".png": PNG,
        ".jpg": JPEG,
        ".jpeg": JPEG,
        ".svg": SVG,
        ".json": JSON,
        ".txt": TEXT,
    }


class UploadReason(StrEnum):
    """Why a file is in an upload plan."""

    NEW = auto()
    """The bucket has no object at that key."""

    CHANGED = auto()
    """The bucket's object has a different content hash."""


class MissingS3ConfigurationError(RuntimeError):
    """Raised when the bucket's connection details are not fully configured."""


class MissingPublishDependencyError(RuntimeError):
    """Raised when reaching the bucket is asked of an environment without the publish extra."""


class IncompleteLibraryError(RuntimeError):
    """Raised when the index or a manifest references a file that is not on disk."""


PUBLISH_EXTRA_HINT = (
    "the publishing commands need the tool's publish extra, which is not installed here."
    ' Install it with `uv sync --extra publish`, or `pip install "haidra-param-sweeps[publish]"`.'
    " It is deliberately optional: rendering environments do not carry an object-storage SDK."
)
"""What to tell an operator who asked for a bucket operation in an environment that cannot do one."""


def publish_dependencies_installed() -> bool:
    """Return whether this environment can reach a bucket at all."""
    return importlib.util.find_spec("boto3") is not None


def bucket_transport_errors() -> tuple[type[BaseException], ...]:
    """Return the exception types a bucket call can fail with, or none where there is no SDK.

    An empty tuple matches nothing, which is correct rather than merely convenient: without the
    SDK no bucket call can have been made, so no such failure can be in flight.
    """
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return ()

    return (BotoCoreError, ClientError)


class S3Settings(BaseModel):
    """Represents the connection details of the bucket the library is published to."""

    model_config = FROZEN_MODEL_CONFIG

    endpoint_url: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    prefix: str = ""
    """An optional key prefix, so one bucket can hold more than one thing. No leading slash."""

    region: str = "auto"
    """Cloudflare R2 and several other S3-compatible services want ``auto``; AWS wants a region."""


class LibraryFile(BaseModel):
    """Represents one file the index references, on disk."""

    model_config = FROZEN_MODEL_CONFIG

    relative_path: str
    """The path as the index states it, relative to the library root, with forward slashes."""

    local_path: Path
    size_bytes: int


class RemoteObject(BaseModel):
    """Represents one object in the bucket, as a listing reports it."""

    model_config = FROZEN_MODEL_CONFIG

    key: str
    size_bytes: int
    content_hash: str
    """The ETag with its quotes removed: the MD5 for the single-part uploads this tool makes."""


class PlannedUpload(BaseModel):
    """Represents one file an upload would write, and why."""

    model_config = FROZEN_MODEL_CONFIG

    key: str
    local_path: Path
    relative_path: str
    size_bytes: int
    reason: UploadReason


class SyncPlan(BaseModel):
    """Represents everything an upload would do, before any of it is done.

    ``uploads`` is ordered so that ``index.json`` is written last. A bucket is then never in a
    state where the index it serves advertises files that have not arrived yet: the worst an
    interrupted upload leaves behind is objects nothing points at, which the next run reconciles.
    """

    model_config = FROZEN_MODEL_CONFIG

    uploads: list[PlannedUpload] = Field(default_factory=list)
    unchanged_keys: list[str] = Field(default_factory=list)
    orphan_keys: list[str] = Field(default_factory=list)
    """Objects under the prefix that the index does not reference, whether or not they are deleted."""

    deletions: list[str] = Field(default_factory=list)
    """The orphans this plan would actually remove, which is none unless deletion was asked for."""

    @property
    def upload_bytes(self) -> int:
        """Return how many bytes the uploads add up to."""
        return sum(upload.size_bytes for upload in self.uploads)

    @property
    def is_empty(self) -> bool:
        """Return whether the plan would touch the bucket at all."""
        return not self.uploads and not self.deletions


class VerificationReport(BaseModel):
    """Represents the outcome of checking a library against its index."""

    model_config = FROZEN_MODEL_CONFIG

    checked_count: int
    missing_paths: list[str] = Field(default_factory=list)
    mismatched_paths: list[str] = Field(default_factory=list)
    orphan_keys: list[str] = Field(default_factory=list)
    """Always empty for a local check: the disk is allowed to hold things the index does not name."""

    @property
    def is_intact(self) -> bool:
        """Return whether every file the index references was present and matched."""
        return not self.missing_paths and not self.mismatched_paths


def load_s3_settings(*, env_file: Path | None, prefix_override: str | None = None) -> S3Settings:
    """Return the bucket's connection details from the environment, falling back to a ``.env``.

    Args:
        env_file: A ``.env`` to read variables the environment does not set, or None to use only
            the environment.
        prefix_override: A prefix given on the command line, which wins over the environment.

    Returns:
        The settings.

    Raises:
        MissingS3ConfigurationError: If the endpoint, bucket or credentials are not all present.
    """
    from_file = _read_env_file(env_file) if env_file is not None and env_file.is_file() else {}

    def value_of(variable_name: str) -> str:
        from_environment = os.environ.get(variable_name, "").strip()
        return from_environment if from_environment else from_file.get(variable_name, "")

    required = {
        PublishEnvironment.ENDPOINT_URL: value_of(PublishEnvironment.ENDPOINT_URL),
        PublishEnvironment.BUCKET: value_of(PublishEnvironment.BUCKET),
        PublishEnvironment.ACCESS_KEY_ID: value_of(PublishEnvironment.ACCESS_KEY_ID),
        PublishEnvironment.SECRET_ACCESS_KEY: value_of(PublishEnvironment.SECRET_ACCESS_KEY),
    }
    unset = sorted(variable_name for variable_name, variable_value in required.items() if not variable_value)
    if unset:
        looked_in = f", and {env_file}" if env_file is not None else ""
        raise MissingS3ConfigurationError(f"not set in the environment{looked_in}: {', '.join(unset)}")

    prefix = prefix_override if prefix_override is not None else value_of(PublishEnvironment.PREFIX)
    region = value_of(PublishEnvironment.REGION)

    return S3Settings(
        endpoint_url=required[PublishEnvironment.ENDPOINT_URL],
        bucket=required[PublishEnvironment.BUCKET],
        access_key_id=required[PublishEnvironment.ACCESS_KEY_ID],
        secret_access_key=required[PublishEnvironment.SECRET_ACCESS_KEY],
        prefix=normalise_prefix(prefix),
        region=region if region else "auto",
    )


def normalise_prefix(prefix: str) -> str:
    """Return a prefix with no leading or trailing slash, so keys join it unambiguously."""
    return prefix.strip().strip("/")


def object_key(*, prefix: str, relative_path: str) -> str:
    """Return the bucket key for a path the index states, under the prefix.

    The path is used exactly as the index states it. That is the whole addressing contract: a
    consumer joins its configured base URL to the same string and reaches the same object.
    """
    return f"{prefix}/{relative_path}" if prefix else relative_path


def indexed_relative_paths(index: SweepIndex) -> list[str]:
    """Return every path the index references, deduplicated, with ``index.json`` first.

    The index itself is included: it is as much a part of the published library as the images,
    and a consumer fetches it before anything else.
    """
    paths: list[str] = [INDEX_FILENAME]
    seen: set[str] = {INDEX_FILENAME}

    def take(path: str | None) -> None:
        if path is None or path in seen:
            return
        seen.add(path)
        paths.append(path)

    for strip in index.strips:
        take(strip.source_image)
        # A strip held to one detector publishes the map beside its source; a gallery stop
        # publishes the map beside its render. Both are read off the index by consumers, so
        # both are part of the closure the library has to carry.
        take(strip.control_map)

        for stop in strip.stops:
            take(stop.file)
            take(stop.sidecar)
            take(stop.control_map)
            # The crop and the bounded copy are what a page actually draws for a frame it cannot
            # show whole, so they belong to the closure exactly as the frame itself does.
            take(stop.crop.file if stop.crop is not None else None)
            take(stop.fit)

    return paths


def tutorial_manifest_relative_paths(manifest: TutorialManifest) -> list[str]:
    """Return every path the tutorial manifest references, with the manifest itself first."""
    paths: list[str] = [TUTORIAL_MANIFEST_PATH]
    seen: set[str] = set(paths)
    for flow in manifest.flows:
        for file_name in flow.files:
            path = f"{TUTORIALS_DIRECTORY_NAME}/{flow.directory}/{file_name}"
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def alchemy_manifest_relative_paths(manifest: AlchemyManifest) -> list[str]:
    """Return every path the alchemy manifest references, with the manifest itself first.

    Entries share their source image, so a source is listed once however many forms ran on it.
    """
    paths: list[str] = [ALCHEMY_MANIFEST_PATH]
    seen: set[str] = set(paths)
    for entry in manifest.entries:
        crop_names = [crop.file for crop in (entry.source_crop, entry.result_crop) if crop is not None]
        for file_name in (
            entry.files.source,
            entry.files.result,
            entry.files.result_text,
            entry.files.result_json,
            *crop_names,
            entry.result_fit,
        ):
            if file_name is None:
                continue
            path = f"{ALCHEMY_DIRECTORY_NAME}/{file_name}"
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def cards_manifest_relative_paths(manifest: CardsManifest) -> list[str]:
    """Return every path the share-card manifest references, with the manifest itself first."""
    paths: list[str] = [PROMO_MANIFEST_PATH]
    seen: set[str] = set(paths)
    for entry in manifest.cards.values():
        if entry.file not in seen:
            seen.add(entry.file)
            paths.append(entry.file)
    return paths


def gallery_manifest_relative_paths(manifest: dict[str, Any]) -> list[str]:
    """Return every path the gallery manifest references, the manifest itself first.

    The gallery is written by the frontpage's promo runner rather than by this tool, so its
    manifest is read as plain JSON: each image entry has a ``file`` and an ``alchemy`` map of
    entries with a ``file``, and each text entry has a ``file``.
    """
    paths: list[str] = [PROMO_GALLERY_MANIFEST_PATH]
    seen: set[str] = set(paths)

    def take(path: object) -> None:
        if isinstance(path, str) and path and path not in seen:
            seen.add(path)
            paths.append(path)

    for image in manifest.get("images", []):
        take(image.get("file"))
        for record in image.get("alchemy", {}).values():
            take(record.get("file") if isinstance(record, dict) else None)
    for text in manifest.get("texts", []):
        take(text.get("file"))
    return paths


def promo_relative_paths(*, output_root: Path) -> list[str]:
    """Return every path the promo tree publishes.

    The card manifest and every card it names come first, then the gallery manifest and
    everything it names where a gallery has been made.

    Raises:
        FileNotFoundError: If there is no card manifest to read.
    """
    manifest_path = output_root / PROMO_MANIFEST_PATH
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no {PROMO_MANIFEST_PATH} at {output_root}: draw the cards first")
    paths = cards_manifest_relative_paths(CardsManifest.model_validate_json(manifest_path.read_text(encoding="utf-8")))

    gallery_path = output_root / PROMO_GALLERY_MANIFEST_PATH
    if gallery_path.is_file():
        gallery = json.loads(gallery_path.read_text(encoding="utf-8"))
        paths.extend(path for path in gallery_manifest_relative_paths(gallery) if path not in paths)
    return paths


def published_relative_paths(
    *,
    output_root: Path,
    tree: PublishTree = PublishTree.SWEEPS,
    index: SweepIndex | None = None,
) -> list[str]:
    """Return every path a tree publishes.

    For the evidence library that is the index's closure, then each manifest's. The index is
    required; the two manifests are optional, because a library can be rendered without the
    tutorial or alchemy campaigns having run. A manifest that is present is published with
    everything it names, so the part of the tree the web app fetches through it never 404s. For
    the promo tree it is the card manifest and everything it names.

    Args:
        output_root: The root of the tree on disk.
        tree: Which tree ``output_root`` is.
        index: The already-parsed index, or None to read it from ``output_root``; unused for the promo tree.

    Returns:
        The paths, each once, entrypoint first within its group.

    Raises:
        FileNotFoundError: If there is no index, or no card manifest, to read.
    """
    if tree is PublishTree.PROMO:
        return promo_relative_paths(output_root=output_root)

    resolved_index = index if index is not None else read_index(output_root=output_root)
    paths = indexed_relative_paths(resolved_index)

    tutorial_manifest_path = output_root / TUTORIAL_MANIFEST_PATH
    if tutorial_manifest_path.is_file():
        tutorial_manifest = TutorialManifest.model_validate_json(tutorial_manifest_path.read_text(encoding="utf-8"))
        paths.extend(tutorial_manifest_relative_paths(tutorial_manifest))

    alchemy_manifest_path = output_root / ALCHEMY_MANIFEST_PATH
    if alchemy_manifest_path.is_file():
        alchemy_manifest = AlchemyManifest.model_validate_json(alchemy_manifest_path.read_text(encoding="utf-8"))
        paths.extend(alchemy_manifest_relative_paths(alchemy_manifest))

    return paths


def read_index(*, output_root: Path) -> SweepIndex:
    """Return the library's index, parsed from the root of the output tree.

    Raises:
        FileNotFoundError: If there is no index there to read.
    """
    index_path = output_root / INDEX_FILENAME
    if not index_path.is_file():
        raise FileNotFoundError(f"no {INDEX_FILENAME} at {output_root}: render the library first")

    return SweepIndex.model_validate_json(index_path.read_text(encoding="utf-8"))


def enumerate_library(
    *,
    output_root: Path,
    tree: PublishTree = PublishTree.SWEEPS,
    index: SweepIndex | None = None,
) -> list[LibraryFile]:
    """Return every file a tree publishes, with its size, in the tree's own order.

    Args:
        output_root: The root of the tree on disk.
        tree: Which tree ``output_root`` is.
        index: The already-parsed index, or None to read it from ``output_root``; unused for the promo tree.

    Returns:
        One entry per referenced path.

    Raises:
        FileNotFoundError: If there is no index to read.
        IncompleteLibraryError: If the index or a manifest references a file that is not on disk.
    """
    files: list[LibraryFile] = []
    missing: list[str] = []

    for relative_path in published_relative_paths(output_root=output_root, tree=tree, index=index):
        local_path = output_root / relative_path
        if not local_path.is_file():
            missing.append(relative_path)
            continue

        files.append(
            LibraryFile(
                relative_path=relative_path,
                local_path=local_path,
                size_bytes=local_path.stat().st_size,
            ),
        )

    if missing:
        raise IncompleteLibraryError(
            f"{len(missing)} referenced file(s) are not on disk, first: {missing[0]}",
        )

    return files


def file_content_hash(path: Path) -> str:
    """Return the MD5 of a file's contents, as the hexadecimal digest an ETag carries.

    MD5 is used because it is what the object store reports back, not to guard against tampering.
    """
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)

    return digest.hexdigest()


def content_type_for(relative_path: str) -> str:
    """Return the ``Content-Type`` to write an object with, from its extension."""
    return ContentType.BY_SUFFIX.get(PurePosixPath(relative_path).suffix.lower(), ContentType.FALLBACK)


def cache_control_for(relative_path: str) -> str:
    """Return the ``Cache-Control`` to write an object with: brief for an entrypoint, long otherwise."""
    return CacheControl.INDEX if relative_path in ENTRYPOINT_PATHS else CacheControl.ASSET


def plan_sync(
    *,
    files: Sequence[LibraryFile],
    remote_objects: Mapping[str, RemoteObject],
    prefix: str = "",
    delete_orphans: bool = False,
) -> SyncPlan:
    """Return what an upload would do, deciding entirely from a listing and the local files.

    A file is uploaded when the bucket has no object at its key, or when the object's ETag is not
    the file's MD5. An ETag that carries a part count (it contains a hyphen) is not an MD5 and is
    treated as a mismatch, so an object uploaded by something other than this tool is replaced
    rather than silently trusted.

    Args:
        files: The library's files, from [`enumerate_library`][param_sweeps.publish.enumerate_library].
        remote_objects: The bucket's objects under the prefix, keyed by their full key.
        prefix: The key prefix the library is published under.
        delete_orphans: Whether the plan should remove objects the index does not reference.

    Returns:
        The plan, with the index and the manifests ordered last among the uploads.

    Performance:
        Hashes only the files that already exist in the bucket; a file that has to be uploaded
        anyway is never read.
    """
    uploads: list[PlannedUpload] = []
    unchanged_keys: list[str] = []
    expected_keys: set[str] = set()

    for library_file in files:
        key = object_key(prefix=prefix, relative_path=library_file.relative_path)
        expected_keys.add(key)

        remote_object = remote_objects.get(key)
        if remote_object is None:
            uploads.append(_planned_upload(library_file, key=key, reason=UploadReason.NEW))
            continue

        if remote_object.content_hash == file_content_hash(library_file.local_path):
            unchanged_keys.append(key)
            continue

        uploads.append(_planned_upload(library_file, key=key, reason=UploadReason.CHANGED))

    orphan_keys = sorted(key for key in remote_objects if key not in expected_keys)

    return SyncPlan(
        uploads=sorted(uploads, key=lambda upload: upload.relative_path in ENTRYPOINT_PATHS),
        unchanged_keys=unchanged_keys,
        orphan_keys=orphan_keys,
        deletions=list(orphan_keys) if delete_orphans else [],
    )


def verify_remote(
    *,
    files: Sequence[LibraryFile],
    remote_objects: Mapping[str, RemoteObject],
    prefix: str = "",
) -> VerificationReport:
    """Return whether the bucket holds exactly the library the index describes.

    Args:
        files: The library's files, from [`enumerate_library`][param_sweeps.publish.enumerate_library].
        remote_objects: The bucket's objects under the prefix, keyed by their full key.
        prefix: The key prefix the library is published under.

    Returns:
        The report, whose ``is_intact`` is what an exit code should be taken from.
    """
    missing: list[str] = []
    mismatched: list[str] = []
    expected_keys: set[str] = set()

    for library_file in files:
        key = object_key(prefix=prefix, relative_path=library_file.relative_path)
        expected_keys.add(key)

        remote_object = remote_objects.get(key)
        if remote_object is None:
            missing.append(key)
            continue

        if remote_object.content_hash != file_content_hash(library_file.local_path):
            mismatched.append(key)

    return VerificationReport(
        checked_count=len(files),
        missing_paths=missing,
        mismatched_paths=mismatched,
        orphan_keys=sorted(key for key in remote_objects if key not in expected_keys),
    )


def verify_local(*, output_root: Path, tree: PublishTree = PublishTree.SWEEPS) -> VerificationReport:
    """Return whether every file a tree publishes is on disk and not empty.

    There is no hash to check against locally, so this is an existence and non-emptiness check:
    it catches the deletion or truncation that would make an upload publish a broken library.

    Args:
        output_root: The root of the tree on disk.
        tree: Which tree ``output_root`` is.

    Returns:
        The report.

    Raises:
        FileNotFoundError: If there is no index, or no card manifest, to read.
    """
    relative_paths = published_relative_paths(output_root=output_root, tree=tree)

    missing = [path for path in relative_paths if not (output_root / path).is_file()]
    empty = [path for path in relative_paths if path not in missing and (output_root / path).stat().st_size == 0]

    return VerificationReport(checked_count=len(relative_paths), missing_paths=missing, mismatched_paths=empty)


class SweepLibraryBucket:
    """The bucket the library is published to, and the four operations this tool needs of it.

    The client is constructed from [`S3Settings`][param_sweeps.publish.S3Settings] and nothing
    here is specific to any one provider: Cloudflare R2 is an S3 endpoint URL and a pair of
    credentials like any other.

    Objects are written with ``put_object`` rather than the transfer manager, which keeps every
    upload single-part and therefore keeps the ETag equal to the object's MD5. That equality is
    what makes an upload incremental, so it is worth giving up parallel multipart transfers for.
    """

    def __init__(self, *, settings: S3Settings, client: S3Client | None = None) -> None:
        """Create a bucket handle, optionally around an already-configured client.

        Args:
            settings: The connection details.
            client: A client to use instead of building one, which is how the tests inject a
                stubbed client.
        """
        self.settings = settings
        self._client = client if client is not None else create_s3_client(settings=settings)

    @property
    def client(self) -> S3Client:
        """Return the underlying S3 client."""
        return self._client

    def list_objects(self) -> dict[str, RemoteObject]:
        """Return every object under the configured prefix, keyed by its full key.

        Network Behavior:
            Pages through the whole listing, which is one request per thousand objects.
        """
        listed: dict[str, RemoteObject] = {}
        paginator = self._client.get_paginator("list_objects_v2")
        list_prefix = f"{self.settings.prefix}/" if self.settings.prefix else ""

        for page in paginator.paginate(Bucket=self.settings.bucket, Prefix=list_prefix):
            for entry in page.get("Contents", []):
                key = entry["Key"]
                listed[key] = RemoteObject(
                    key=key,
                    size_bytes=entry.get("Size", 0),
                    content_hash=entry.get("ETag", "").strip('"'),
                )

        logger.debug("Listed %d object(s) under %r", len(listed), list_prefix)
        return listed

    def put_file(self, *, key: str, local_path: Path, relative_path: str) -> None:
        """Write one file to the bucket with its content type and cache header.

        Side Effects:
            Overwrites whatever is at that key.
        """
        with local_path.open("rb") as file_handle:
            self._client.put_object(
                Bucket=self.settings.bucket,
                Key=key,
                Body=file_handle.read(),
                ContentType=content_type_for(relative_path),
                CacheControl=cache_control_for(relative_path),
            )

    def delete_key(self, *, key: str) -> None:
        """Remove one object from the bucket.

        Side Effects:
            Deletes remote data, and does not ask twice.
        """
        self._client.delete_object(Bucket=self.settings.bucket, Key=key)


def create_s3_client(*, settings: S3Settings) -> S3Client:
    """Create an S3 client for the configured endpoint and credentials.

    Raises:
        MissingPublishDependencyError: If this environment has no object-storage SDK installed.
    """
    try:
        import boto3
    except ImportError as missing_dependency:
        raise MissingPublishDependencyError(PUBLISH_EXTRA_HINT) from missing_dependency

    return boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.access_key_id,
        aws_secret_access_key=settings.secret_access_key,
        region_name=settings.region,
    )


def apply_sync_plan(*, bucket: SweepLibraryBucket, plan: SyncPlan, workers: int = DEFAULT_UPLOAD_WORKERS) -> None:
    """Carry out a plan against the bucket, uploading before deleting and the entrypoints last.

    The ordinary objects are put concurrently: each is one small request, so a serial run is
    bound by round trips rather than bytes. The entrypoints are put only after every one of those
    has landed, one at a time, so no consumer fetches an index or manifest naming an object that
    is not there yet. Deletions come after everything, for the same reason.

    Args:
        bucket: The bucket to write to.
        plan: The plan from [`plan_sync`][param_sweeps.publish.plan_sync].
        workers: How many ordinary objects to put at once; 1 puts them in plan order.

    Side Effects:
        Writes and, when the plan says so, deletes remote data.
    """

    def put(upload: PlannedUpload) -> None:
        logger.info("Uploading %s (%s, %d bytes)", upload.key, upload.reason, upload.size_bytes)
        bucket.put_file(key=upload.key, local_path=upload.local_path, relative_path=upload.relative_path)

    ordinary = [upload for upload in plan.uploads if upload.relative_path not in ENTRYPOINT_PATHS]
    entrypoints = [upload for upload in plan.uploads if upload.relative_path in ENTRYPOINT_PATHS]

    if workers <= 1:
        for upload in ordinary:
            put(upload)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(put, ordinary):
                pass

    for upload in entrypoints:
        put(upload)

    for key in plan.deletions:
        logger.info("Deleting orphan %s", key)
        bucket.delete_key(key=key)


def _planned_upload(library_file: LibraryFile, *, key: str, reason: UploadReason) -> PlannedUpload:
    """Create the plan entry for one file."""
    return PlannedUpload(
        key=key,
        local_path=library_file.local_path,
        relative_path=library_file.relative_path,
        size_bytes=library_file.size_bytes,
        reason=reason,
    )


def _read_env_file(env_file: Path) -> dict[str, str]:
    """Return every assignment in a ``.env``, as a mapping. Comments and blank lines are ignored."""
    assignments: dict[str, str] = {}

    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue

        assignment = ENV_ASSIGNMENT_PATTERN.match(line)
        if assignment is None:
            continue

        assignments[assignment.group("name")] = assignment.group("value").strip().strip("\"'")

    return assignments


def format_byte_size(byte_count: int) -> str:
    """Return a byte count as a short human-readable string."""
    if byte_count < 1024:
        return f"{byte_count} B"

    size = float(byte_count)
    for unit in ("KiB", "MiB"):
        size /= 1024
        if size < 1024:
            return f"{size:.1f} {unit}"

    return f"{size / 1024:.1f} GiB"
