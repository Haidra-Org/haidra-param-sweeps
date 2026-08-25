# SPDX-FileCopyrightText: 2026 Tazlin
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check the publishing side: what the index says belongs, and what a sync would do about it.

Publishing is an optional extra, so this whole module is skipped where the object-storage SDK is
not installed: there is no client to stub and nothing here would be exercising the tool. The one
test that guards the extra's optionality lives here too, because the environment it can catch a
regression in is exactly the one where the SDK is present and could be imported by accident.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, override

import pytest

from param_sweeps import publish
from param_sweeps.cli import main
from param_sweeps.outputs import INDEX_FILENAME
from param_sweeps.publish import (
    ALCHEMY_MANIFEST_PATH,
    PUBLISH_EXTRA_HINT,
    TUTORIAL_MANIFEST_PATH,
    CacheControl,
    ContentType,
    IncompleteLibraryError,
    MissingPublishDependencyError,
    MissingS3ConfigurationError,
    PublishEnvironment,
    RemoteObject,
    S3Settings,
    SweepLibraryBucket,
    UploadReason,
    apply_sync_plan,
    cache_control_for,
    content_type_for,
    create_s3_client,
    enumerate_library,
    file_content_hash,
    load_s3_settings,
    object_key,
    plan_sync,
    verify_local,
    verify_remote,
)

pytest.importorskip(
    "botocore.stub",
    reason="the tool's publish extra is not installed, so there is no bucket client to stub",
)

from botocore.stub import Stubber

TOOL_ROOT = Path(__file__).resolve().parents[1]

STRIP_DIRECTORY = "steps/deliberate/person"
FIRST_STOP_IMAGE = f"{STRIP_DIRECTORY}/001_4.webp"
FIRST_STOP_SIDECAR = f"{STRIP_DIRECTORY}/001_4.json"
SECOND_STOP_IMAGE = f"{STRIP_DIRECTORY}/002_8.webp"
SECOND_STOP_SIDECAR = f"{STRIP_DIRECTORY}/002_8.json"
SOURCE_IMAGE = "denoising-strength/deliberate/place/source.webp"
FIRST_STOP_CROP = f"{STRIP_DIRECTORY}/001_4_crop.webp"
FIRST_STOP_FIT = f"{STRIP_DIRECTORY}/001_4_fit.webp"

TUTORIAL_FILES = ["tutorials/txt2img/meta.json", "tutorials/txt2img/request.json", "tutorials/txt2img/result.webp"]
ALCHEMY_FILES = [
    "alchemy/sources/anchor.webp",
    "alchemy/describe/caption/result.txt",
    "alchemy/enlarge/vectorize/result.svg",
    "alchemy/enlarge/upscale/result.webp",
]
ALCHEMY_DERIVATIVE_FILES = [
    "alchemy/enlarge/vectorize/source_crop.webp",
    "alchemy/enlarge/upscale/result_fit.webp",
]

TEST_SETTINGS = S3Settings(
    endpoint_url="https://accountid.r2.cloudflarestorage.com",
    bucket="sweeps",
    access_key_id="key",
    secret_access_key="secret",
)


def _index_document(*, with_source_image: bool = True, with_derivatives: bool = False) -> dict[str, Any]:
    """Return a minimal but structurally real index: one strip, two stops, one source image."""
    return {
        "schema_version": 1,
        "seed": 1234567890,
        "strips": [
            {
                "strip_id": "steps/deliberate/person",
                "parameter": "steps",
                "presentation": "strip",
                "caption": "More steps, more refinement.",
                "rationale": "Steps are the first dial anyone touches.",
                "complete": True,
                "image_model": {
                    "slug": "deliberate",
                    "name": "Deliberate",
                    "baseline": "stable_diffusion_1",
                    "note": "",
                },
                "scene": {"scene_id": "person", "label": "A person", "prompt": "a lighthouse keeper"},
                "held": {
                    "width": 512,
                    "height": 512,
                    "steps": 4,
                    "cfg_scale": 7.0,
                    "sampler_name": "k_euler",
                    "clip_skip": 1,
                    "denoising_strength": 1.0,
                    "karras": True,
                    "hires_fix": False,
                    "tiling": False,
                    "seed": 1234567890,
                    "negative_prompt": "",
                    "prompt": "a lighthouse keeper",
                },
                "varying": ["steps"],
                "stops": [
                    {
                        "value": 4,
                        "label": "4",
                        "caption": "Four steps.",
                        "file": FIRST_STOP_IMAGE,
                        "sidecar": FIRST_STOP_SIDECAR,
                        "width": 2048,
                        "height": 2048,
                        "crop": (
                            {"file": FIRST_STOP_CROP, "width": 512, "height": 512, "x": 1024, "y": 256}
                            if with_derivatives
                            else None
                        ),
                        "fit": FIRST_STOP_FIT if with_derivatives else None,
                    },
                    {
                        "value": 8,
                        "label": "8",
                        "caption": "Eight steps.",
                        "file": SECOND_STOP_IMAGE,
                        "sidecar": SECOND_STOP_SIDECAR,
                        "width": 512,
                        "height": 512,
                    },
                ],
                "source_image": SOURCE_IMAGE if with_source_image else None,
            },
        ],
    }


def _write_library(root: Path, *, with_source_image: bool = True, with_derivatives: bool = False) -> list[str]:
    """Write an index and every file it references, and return those paths in the index's order."""
    document = _index_document(with_source_image=with_source_image, with_derivatives=with_derivatives)
    (root / INDEX_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    (root / INDEX_FILENAME).write_text(json.dumps(document, indent=2), encoding="utf-8")

    referenced = [FIRST_STOP_IMAGE, FIRST_STOP_SIDECAR]
    if with_derivatives:
        referenced += [FIRST_STOP_CROP, FIRST_STOP_FIT]
    referenced += [SECOND_STOP_IMAGE, SECOND_STOP_SIDECAR]
    if with_source_image:
        referenced.insert(0, SOURCE_IMAGE)

    for relative_path in referenced:
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(relative_path.encode("utf-8"))

    return [INDEX_FILENAME, *referenced]


def _write_manifests(root: Path, *, with_derivatives: bool = False) -> list[str]:
    """Write a tutorial and an alchemy manifest beside the index, with every file they name.

    The two alchemy entries share one source, so the closure lists it once.
    """
    tutorial_manifest = {
        "generated_against": "test",
        "flows": [
            {
                "id": "txt2img",
                "directory": "txt2img",
                "title": "Text to image",
                "summary": "One request.",
                "status": "available",
                "files": ["meta.json", "request.json", "result.webp"],
            },
        ],
    }
    alchemy_manifest = {
        "generated_against": "test",
        "entries": [
            {
                "use_case": "describe",
                "form": "caption",
                "title": "Caption",
                "files": {"source": "sources/anchor.webp", "result_text": "describe/caption/result.txt"},
            },
            {
                "use_case": "enlarge",
                "form": "vectorize",
                "title": "Vectorize",
                "files": {"source": "sources/anchor.webp", "result": "enlarge/vectorize/result.svg"},
                "source_crop": (
                    {"file": "enlarge/vectorize/source_crop.webp", "width": 128, "height": 128, "x": 256, "y": 48}
                    if with_derivatives
                    else None
                ),
            },
            {
                "use_case": "enlarge",
                "form": "upscale",
                "title": "Upscale",
                "files": {"source": "sources/anchor.webp", "result": "enlarge/upscale/result.webp"},
                "result_fit": "enlarge/upscale/result_fit.webp" if with_derivatives else None,
            },
        ],
    }
    for manifest_path, document in (
        (TUTORIAL_MANIFEST_PATH, tutorial_manifest),
        (ALCHEMY_MANIFEST_PATH, alchemy_manifest),
    ):
        path = root / manifest_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    alchemy_files = [*ALCHEMY_FILES, *(ALCHEMY_DERIVATIVE_FILES if with_derivatives else [])]
    for relative_path in [*TUTORIAL_FILES, *alchemy_files]:
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(relative_path.encode("utf-8"))

    return [TUTORIAL_MANIFEST_PATH, *TUTORIAL_FILES, ALCHEMY_MANIFEST_PATH, *alchemy_files]


def _remote_mirror(root: Path, relative_paths: list[str], *, prefix: str = "") -> dict[str, RemoteObject]:
    """Return a listing that matches the files on disk exactly."""
    listing: dict[str, RemoteObject] = {}
    for relative_path in relative_paths:
        key = object_key(prefix=prefix, relative_path=relative_path)
        local_path = root / relative_path
        listing[key] = RemoteObject(
            key=key,
            size_bytes=local_path.stat().st_size,
            content_hash=file_content_hash(local_path),
        )

    return listing


def test_the_closure_carries_the_crops_and_the_bounded_copies(tmp_path: Path) -> None:
    """Assert what a page draws is published, and not only the frames the index names.

    The closure once missed the control maps, which published an index whose own references
    404ed. Every derived file the index or a manifest names belongs to it for the same reason.
    """
    _write_library(tmp_path, with_derivatives=True)
    _write_manifests(tmp_path, with_derivatives=True)

    published = [library_file.relative_path for library_file in enumerate_library(output_root=tmp_path)]

    assert FIRST_STOP_CROP in published
    assert FIRST_STOP_FIT in published
    assert "alchemy/enlarge/vectorize/source_crop.webp" in published
    assert "alchemy/enlarge/upscale/result_fit.webp" in published


def test_enumerate_library_covers_index_stops_sidecars_and_source(tmp_path: Path) -> None:
    expected_paths = _write_library(tmp_path)

    files = enumerate_library(output_root=tmp_path)

    assert [library_file.relative_path for library_file in files] == [
        INDEX_FILENAME,
        SOURCE_IMAGE,
        FIRST_STOP_IMAGE,
        FIRST_STOP_SIDECAR,
        SECOND_STOP_IMAGE,
        SECOND_STOP_SIDECAR,
    ]
    assert sorted(library_file.relative_path for library_file in files) == sorted(expected_paths)


def test_enumerate_library_adds_each_manifest_and_its_closure_after_the_index(tmp_path: Path) -> None:
    index_paths = _write_library(tmp_path)
    manifest_paths = _write_manifests(tmp_path)
    (tmp_path / "alchemy/describe/caption/meta.json").write_bytes(b"provenance the manifest does not name")

    relative_paths = [library_file.relative_path for library_file in enumerate_library(output_root=tmp_path)]

    assert relative_paths == [*index_paths, *manifest_paths]


def test_enumerate_library_refuses_a_manifest_that_names_a_missing_file(tmp_path: Path) -> None:
    _write_library(tmp_path)
    _write_manifests(tmp_path)
    (tmp_path / ALCHEMY_FILES[-1]).unlink()

    with pytest.raises(IncompleteLibraryError, match=ALCHEMY_FILES[-1]):
        enumerate_library(output_root=tmp_path)


def test_a_plan_uploads_the_index_and_both_manifests_after_everything_else(tmp_path: Path) -> None:
    _write_library(tmp_path)
    _write_manifests(tmp_path)

    plan = plan_sync(files=enumerate_library(output_root=tmp_path), remote_objects={})

    trailing = [upload.relative_path for upload in plan.uploads[-3:]]
    assert sorted(trailing) == sorted([INDEX_FILENAME, TUTORIAL_MANIFEST_PATH, ALCHEMY_MANIFEST_PATH])
    assert not any(upload.relative_path in trailing for upload in plan.uploads[:-3])


def test_enumerate_library_ignores_files_the_index_does_not_reference(tmp_path: Path) -> None:
    _write_library(tmp_path)
    (tmp_path / "leftover.webp").write_bytes(b"not referenced")

    relative_paths = [library_file.relative_path for library_file in enumerate_library(output_root=tmp_path)]

    assert "leftover.webp" not in relative_paths


def test_the_closure_carries_the_control_maps_a_strip_and_its_stops_publish(tmp_path: Path) -> None:
    """A map beside the strip source or beside a render is read off the index, so it ships with it."""
    document = _index_document()
    strip_map = f"{STRIP_DIRECTORY}/person-source_map.webp"
    stop_map = f"{STRIP_DIRECTORY}/001_4_map.webp"
    document["strips"][0]["control_map"] = strip_map
    document["strips"][0]["stops"][0]["control_map"] = stop_map
    (tmp_path / INDEX_FILENAME).write_text(json.dumps(document, indent=2), encoding="utf-8")
    for relative_path in (
        SOURCE_IMAGE,
        FIRST_STOP_IMAGE,
        FIRST_STOP_SIDECAR,
        SECOND_STOP_IMAGE,
        SECOND_STOP_SIDECAR,
        strip_map,
    ):
        file_path = tmp_path / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(relative_path.encode("utf-8"))

    with pytest.raises(IncompleteLibraryError, match=r"001_4_map.webp"):
        enumerate_library(output_root=tmp_path)

    (tmp_path / stop_map).write_bytes(b"map")
    relative_paths = [library_file.relative_path for library_file in enumerate_library(output_root=tmp_path)]

    assert strip_map in relative_paths
    assert stop_map in relative_paths


def test_enumerate_library_refuses_an_index_that_names_a_missing_file(tmp_path: Path) -> None:
    _write_library(tmp_path)
    (tmp_path / SECOND_STOP_SIDECAR).unlink()

    with pytest.raises(IncompleteLibraryError, match=SECOND_STOP_SIDECAR):
        enumerate_library(output_root=tmp_path)


def test_enumerate_library_without_an_index_says_so(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=INDEX_FILENAME):
        enumerate_library(output_root=tmp_path)


def test_fresh_bucket_uploads_everything_with_the_index_last(tmp_path: Path) -> None:
    _write_library(tmp_path)
    files = enumerate_library(output_root=tmp_path)

    plan = plan_sync(files=files, remote_objects={})

    assert len(plan.uploads) == len(files)
    assert {upload.reason for upload in plan.uploads} == {UploadReason.NEW}
    assert plan.uploads[-1].key == INDEX_FILENAME
    assert plan.unchanged_keys == []
    assert plan.orphan_keys == []
    assert plan.upload_bytes == sum(library_file.size_bytes for library_file in files)


def test_rerunning_against_a_matching_bucket_does_nothing(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    files = enumerate_library(output_root=tmp_path)

    plan = plan_sync(files=files, remote_objects=_remote_mirror(tmp_path, relative_paths))

    assert plan.uploads == []
    assert plan.is_empty
    assert len(plan.unchanged_keys) == len(files)


def test_a_changed_file_is_the_only_upload(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    remote_objects = _remote_mirror(tmp_path, relative_paths)
    (tmp_path / FIRST_STOP_IMAGE).write_bytes(b"a different image entirely")
    files = enumerate_library(output_root=tmp_path)

    plan = plan_sync(files=files, remote_objects=remote_objects)

    assert [upload.key for upload in plan.uploads] == [FIRST_STOP_IMAGE]
    assert plan.uploads[0].reason == UploadReason.CHANGED


def test_a_multipart_etag_is_never_trusted_as_a_hash(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    remote_objects = _remote_mirror(tmp_path, relative_paths)
    remote_objects[FIRST_STOP_IMAGE] = remote_objects[FIRST_STOP_IMAGE].model_copy(
        update={"content_hash": f"{hashlib.md5(b'x', usedforsecurity=False).hexdigest()}-2"},
    )
    files = enumerate_library(output_root=tmp_path)

    plan = plan_sync(files=files, remote_objects=remote_objects)

    assert [upload.key for upload in plan.uploads] == [FIRST_STOP_IMAGE]


def test_orphans_are_reported_but_not_deleted_by_default(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    remote_objects = _remote_mirror(tmp_path, relative_paths)
    remote_objects["steps/deliberate/person/003_16.webp"] = RemoteObject(
        key="steps/deliberate/person/003_16.webp",
        size_bytes=10,
        content_hash="0" * 32,
    )
    files = enumerate_library(output_root=tmp_path)

    without_delete = plan_sync(files=files, remote_objects=remote_objects)
    with_delete = plan_sync(files=files, remote_objects=remote_objects, delete_orphans=True)

    assert without_delete.orphan_keys == ["steps/deliberate/person/003_16.webp"]
    assert without_delete.deletions == []
    assert with_delete.deletions == ["steps/deliberate/person/003_16.webp"]


def test_a_prefix_is_joined_to_every_key_and_nothing_else_changes(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    files = enumerate_library(output_root=tmp_path)

    fresh = plan_sync(files=files, remote_objects={}, prefix="param-sweeps")
    matching = plan_sync(
        files=files,
        remote_objects=_remote_mirror(tmp_path, relative_paths, prefix="param-sweeps"),
        prefix="param-sweeps",
    )

    assert all(upload.key.startswith("param-sweeps/") for upload in fresh.uploads)
    assert fresh.uploads[-1].key == f"param-sweeps/{INDEX_FILENAME}"
    assert matching.uploads == []


def test_an_unprefixed_listing_is_orphaned_under_a_prefix(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    files = enumerate_library(output_root=tmp_path)

    plan = plan_sync(
        files=files,
        remote_objects=_remote_mirror(tmp_path, relative_paths),
        prefix="param-sweeps",
        delete_orphans=True,
    )

    assert len(plan.uploads) == len(files)
    assert plan.deletions == sorted(relative_paths)


def test_headers_are_long_for_assets_and_brief_for_the_entrypoints() -> None:
    assert content_type_for(FIRST_STOP_IMAGE) == ContentType.WEBP
    assert content_type_for(FIRST_STOP_SIDECAR) == ContentType.JSON
    assert content_type_for(INDEX_FILENAME) == ContentType.JSON
    assert content_type_for(ALCHEMY_FILES[1]) == ContentType.TEXT
    assert content_type_for(ALCHEMY_FILES[2]) == ContentType.SVG
    assert content_type_for("alchemy/sources/mask.png") == ContentType.PNG
    assert cache_control_for(FIRST_STOP_IMAGE) == CacheControl.ASSET
    assert cache_control_for(TUTORIAL_FILES[0]) == CacheControl.ASSET
    assert cache_control_for(INDEX_FILENAME) == CacheControl.INDEX
    assert cache_control_for(TUTORIAL_MANIFEST_PATH) == CacheControl.INDEX
    assert cache_control_for(ALCHEMY_MANIFEST_PATH) == CacheControl.INDEX


def test_verify_passes_on_a_faithful_mirror_and_reports_what_is_wrong(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path)
    files = enumerate_library(output_root=tmp_path)
    remote_objects = _remote_mirror(tmp_path, relative_paths)

    assert verify_remote(files=files, remote_objects=remote_objects).is_intact

    del remote_objects[SECOND_STOP_IMAGE]
    remote_objects[FIRST_STOP_SIDECAR] = remote_objects[FIRST_STOP_SIDECAR].model_copy(
        update={"content_hash": "1" * 32},
    )
    remote_objects["stale.webp"] = RemoteObject(key="stale.webp", size_bytes=1, content_hash="2" * 32)

    report = verify_remote(files=files, remote_objects=remote_objects)

    assert not report.is_intact
    assert report.missing_paths == [SECOND_STOP_IMAGE]
    assert report.mismatched_paths == [FIRST_STOP_SIDECAR]
    assert report.orphan_keys == ["stale.webp"]


def test_verify_local_catches_a_deleted_and_an_emptied_file(tmp_path: Path) -> None:
    _write_library(tmp_path)

    assert verify_local(output_root=tmp_path).is_intact

    (tmp_path / SECOND_STOP_IMAGE).unlink()
    (tmp_path / FIRST_STOP_IMAGE).write_bytes(b"")

    report = verify_local(output_root=tmp_path)

    assert report.missing_paths == [SECOND_STOP_IMAGE]
    assert report.mismatched_paths == [FIRST_STOP_IMAGE]
    assert report.orphan_keys == []


def test_settings_come_from_the_environment_first_then_the_env_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                f"{PublishEnvironment.ENDPOINT_URL}=https://from-file.example",
                f'{PublishEnvironment.BUCKET}="sweeps"',
                f"export {PublishEnvironment.ACCESS_KEY_ID}=file-key",
                f"{PublishEnvironment.SECRET_ACCESS_KEY}=file-secret",
                f"{PublishEnvironment.PREFIX}=/param-sweeps/",
            ],
        ),
        encoding="utf-8",
    )
    # The variables have to be cleared by their own names rather than by the attribute names that
    # hold them: importing horde_sdk loads whatever .env is above the working directory into the
    # process, so a machine that publishes for real has the live bucket set before the test starts.
    for variable_name in (
        PublishEnvironment.ENDPOINT_URL,
        PublishEnvironment.BUCKET,
        PublishEnvironment.ACCESS_KEY_ID,
        PublishEnvironment.SECRET_ACCESS_KEY,
        PublishEnvironment.PREFIX,
        PublishEnvironment.REGION,
    ):
        monkeypatch.delenv(variable_name, raising=False)
    monkeypatch.setenv(PublishEnvironment.ENDPOINT_URL, "https://from-environment.example")

    settings = load_s3_settings(env_file=env_file)

    assert settings.endpoint_url == "https://from-environment.example"
    assert settings.bucket == "sweeps"
    assert settings.access_key_id == "file-key"
    assert settings.prefix == "param-sweeps"
    assert settings.region == "auto"


def test_missing_settings_name_every_variable_that_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable_name in (
        PublishEnvironment.ENDPOINT_URL,
        PublishEnvironment.BUCKET,
        PublishEnvironment.ACCESS_KEY_ID,
        PublishEnvironment.SECRET_ACCESS_KEY,
    ):
        monkeypatch.delenv(variable_name, raising=False)

    with pytest.raises(MissingS3ConfigurationError, match=PublishEnvironment.BUCKET):
        load_s3_settings(env_file=None)


def _stubbed_bucket(*, settings: S3Settings = TEST_SETTINGS) -> tuple[SweepLibraryBucket, Stubber]:
    """Create a bucket whose client answers only the calls a test declares in advance."""
    client = create_s3_client(settings=settings)
    stubber = Stubber(client)
    return SweepLibraryBucket(settings=settings, client=client), stubber


def test_listing_pages_through_the_bucket_and_unquotes_etags() -> None:
    bucket, stubber = _stubbed_bucket()
    first_etag = "d" * 32
    second_etag = "e" * 32
    stubber.add_response(
        "list_objects_v2",
        {
            "Contents": [{"Key": FIRST_STOP_IMAGE, "Size": 11, "ETag": f'"{first_etag}"'}],
            "IsTruncated": True,
            "NextContinuationToken": "page-2",
        },
        {"Bucket": "sweeps", "Prefix": ""},
    )
    stubber.add_response(
        "list_objects_v2",
        {"Contents": [{"Key": SECOND_STOP_IMAGE, "Size": 12, "ETag": f'"{second_etag}"'}], "IsTruncated": False},
        {"Bucket": "sweeps", "Prefix": "", "ContinuationToken": "page-2"},
    )

    with stubber:
        listing = bucket.list_objects()

    assert listing[FIRST_STOP_IMAGE].content_hash == first_etag
    assert listing[SECOND_STOP_IMAGE].content_hash == second_etag
    stubber.assert_no_pending_responses()


def test_applying_a_plan_writes_every_object_with_its_headers_and_the_index_last(tmp_path: Path) -> None:
    _write_library(tmp_path, with_source_image=False)
    files = enumerate_library(output_root=tmp_path)
    plan = plan_sync(files=files, remote_objects={})
    bucket, stubber = _stubbed_bucket()

    for upload in plan.uploads:
        stubber.add_response(
            "put_object",
            {},
            {
                "Bucket": "sweeps",
                "Key": upload.key,
                "Body": (tmp_path / upload.relative_path).read_bytes(),
                "ContentType": content_type_for(upload.relative_path),
                "CacheControl": cache_control_for(upload.relative_path),
            },
        )

    with stubber:
        apply_sync_plan(bucket=bucket, plan=plan, workers=1)

    stubber.assert_no_pending_responses()
    assert plan.uploads[-1].relative_path == INDEX_FILENAME


class _RecordingBucket(SweepLibraryBucket):
    """A bucket that records the order keys land in, from whichever thread puts them."""

    def __init__(self) -> None:
        super().__init__(settings=TEST_SETTINGS)
        self.landed: list[str] = []
        self._lock = threading.Lock()

    @override
    def put_file(self, *, key: str, local_path: Path, relative_path: str) -> None:
        time.sleep(0.01)
        with self._lock:
            self.landed.append(key)

    @override
    def delete_key(self, *, key: str) -> None:
        with self._lock:
            self.landed.append(f"delete:{key}")


def test_concurrent_uploads_still_land_the_entrypoints_last_and_deletions_after(tmp_path: Path) -> None:
    _write_library(tmp_path)
    _write_manifests(tmp_path)
    files = enumerate_library(output_root=tmp_path)
    plan = plan_sync(
        files=files,
        remote_objects={"stale.webp": RemoteObject(key="stale.webp", size_bytes=1, content_hash="x")},
        delete_orphans=True,
    )
    bucket = _RecordingBucket()

    apply_sync_plan(bucket=bucket, plan=plan, workers=4)

    entrypoints = [INDEX_FILENAME, TUTORIAL_MANIFEST_PATH, ALCHEMY_MANIFEST_PATH]
    assert bucket.landed[-1] == "delete:stale.webp"
    assert bucket.landed[-4:-1] == [upload.key for upload in plan.uploads if upload.relative_path in entrypoints]
    assert sorted(bucket.landed[:-4]) == sorted(
        upload.key for upload in plan.uploads if upload.relative_path not in entrypoints
    )


def test_applying_a_plan_deletes_orphans_after_the_uploads(tmp_path: Path) -> None:
    relative_paths = _write_library(tmp_path, with_source_image=False)
    files = enumerate_library(output_root=tmp_path)
    remote_objects = _remote_mirror(tmp_path, relative_paths)
    remote_objects["gone.webp"] = RemoteObject(key="gone.webp", size_bytes=1, content_hash="3" * 32)
    plan = plan_sync(files=files, remote_objects=remote_objects, delete_orphans=True)
    bucket, stubber = _stubbed_bucket()
    stubber.add_response("delete_object", {}, {"Bucket": "sweeps", "Key": "gone.webp"})

    with stubber:
        apply_sync_plan(bucket=bucket, plan=plan)

    assert plan.uploads == []
    stubber.assert_no_pending_responses()


def test_a_dry_run_lists_the_bucket_and_writes_nothing_to_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_library(tmp_path)
    client = create_s3_client(settings=TEST_SETTINGS)
    stubber = Stubber(client)
    stubber.add_response("list_objects_v2", {"Contents": [], "IsTruncated": False}, {"Bucket": "sweeps", "Prefix": ""})
    monkeypatch.setattr(publish, "create_s3_client", lambda *, settings: client)
    monkeypatch.setenv(PublishEnvironment.ENDPOINT_URL, TEST_SETTINGS.endpoint_url)
    monkeypatch.setenv(PublishEnvironment.BUCKET, TEST_SETTINGS.bucket)
    monkeypatch.setenv(PublishEnvironment.ACCESS_KEY_ID, TEST_SETTINGS.access_key_id)
    monkeypatch.setenv(PublishEnvironment.SECRET_ACCESS_KEY, TEST_SETTINGS.secret_access_key)
    monkeypatch.delenv(PublishEnvironment.PREFIX, raising=False)

    with stubber:
        exit_code = main(
            ["upload", "--dry-run", "--output-root", str(tmp_path), "--env-file", str(tmp_path / "absent.env")],
        )

    assert exit_code == 0
    stubber.assert_no_pending_responses()
    printed = capsys.readouterr().out
    assert "Would upload" in printed
    assert printed.rstrip().endswith("orphan(s) left alone.")


def test_verify_exits_non_zero_when_the_bucket_is_short_a_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relative_paths = _write_library(tmp_path)
    client = create_s3_client(settings=TEST_SETTINGS)
    stubber = Stubber(client)
    stubber.add_response(
        "list_objects_v2",
        {
            "Contents": [
                {
                    "Key": relative_path,
                    "Size": (tmp_path / relative_path).stat().st_size,
                    "ETag": f'"{file_content_hash(tmp_path / relative_path)}"',
                }
                for relative_path in relative_paths
                if relative_path != SECOND_STOP_IMAGE
            ],
            "IsTruncated": False,
        },
        {"Bucket": "sweeps", "Prefix": ""},
    )
    monkeypatch.setattr(publish, "create_s3_client", lambda *, settings: client)
    monkeypatch.setenv(PublishEnvironment.ENDPOINT_URL, TEST_SETTINGS.endpoint_url)
    monkeypatch.setenv(PublishEnvironment.BUCKET, TEST_SETTINGS.bucket)
    monkeypatch.setenv(PublishEnvironment.ACCESS_KEY_ID, TEST_SETTINGS.access_key_id)
    monkeypatch.setenv(PublishEnvironment.SECRET_ACCESS_KEY, TEST_SETTINGS.secret_access_key)
    monkeypatch.delenv(PublishEnvironment.PREFIX, raising=False)

    with stubber:
        exit_code = main(["verify", "--output-root", str(tmp_path), "--env-file", str(tmp_path / "absent.env")])

    assert exit_code == 1


def test_nothing_a_render_imports_drags_in_the_object_storage_sdk() -> None:
    """Assert importing the tool, including its command line, pulls in no publishing SDK.

    The rendering environment shares an interpreter with the GPU stack and must not acquire an
    object-storage SDK by importing this tool. A module-level import anywhere on that path would
    put one there, which is invisible until an environment that lacks it fails to start at all.
    """
    # The probe's answer is fenced with a sentinel because dependencies are allowed to emit their own
    # import-time log lines on stdout; only the fenced line is this probe's output.
    probe = (
        "import sys\n"
        "import param_sweeps\n"
        "import param_sweeps.cli\n"
        "loaded = sorted({name.split('.')[0] for name in sys.modules} & {'boto3', 'botocore'})\n"
        "print('PROBE_RESULT:' + repr(loaded))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=TOOL_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith("PROBE_RESULT:")]
    assert len(result_lines) == 1, f"probe output missing from {completed.stdout!r}"
    assert result_lines[0] == "PROBE_RESULT:[]", f"importing the tool loaded {result_lines[0]}"


def test_a_bucket_command_without_the_extra_says_what_to_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Assert an environment without the extra is told what to install, not handed a traceback."""
    _write_library(tmp_path)
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setenv(PublishEnvironment.ENDPOINT_URL, TEST_SETTINGS.endpoint_url)
    monkeypatch.setenv(PublishEnvironment.BUCKET, TEST_SETTINGS.bucket)
    monkeypatch.setenv(PublishEnvironment.ACCESS_KEY_ID, TEST_SETTINGS.access_key_id)
    monkeypatch.setenv(PublishEnvironment.SECRET_ACCESS_KEY, TEST_SETTINGS.secret_access_key)

    with pytest.raises(MissingPublishDependencyError, match="publish extra"):
        create_s3_client(settings=TEST_SETTINGS)

    exit_code = main(["upload", "--output-root", str(tmp_path), "--env-file", str(tmp_path / "absent.env")])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "--extra publish" in captured.err
    assert PUBLISH_EXTRA_HINT in captured.err


def test_the_commands_that_read_only_the_library_need_no_extra(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert the two commands that touch no bucket still run where the extra is absent."""
    _write_library(tmp_path)
    monkeypatch.setitem(sys.modules, "boto3", None)

    assert main(["manifest", "--output-root", str(tmp_path)]) == 0
    assert main(["verify", "--local", "--output-root", str(tmp_path)]) == 0


def test_verify_local_needs_no_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_library(tmp_path)
    for variable_name in (
        PublishEnvironment.ENDPOINT_URL,
        PublishEnvironment.BUCKET,
        PublishEnvironment.ACCESS_KEY_ID,
        PublishEnvironment.SECRET_ACCESS_KEY,
    ):
        monkeypatch.delenv(variable_name, raising=False)

    assert main(["verify", "--local", "--output-root", str(tmp_path)]) == 0


def test_manifest_lists_every_referenced_file_with_its_size(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    relative_paths = _write_library(tmp_path)

    exit_code = main(["manifest", "--output-root", str(tmp_path)])

    assert exit_code == 0
    captured = capsys.readouterr()
    listed = [line.split("\t")[1] for line in captured.out.splitlines()]
    assert sorted(listed) == sorted(relative_paths)
    assert f"{len(relative_paths)} files" in captured.err
