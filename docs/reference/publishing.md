---
title: "Publishing contract"
summary: "How published object keys map to index paths, which files each tree publishes, the upload order, and the Content-Type and Cache-Control headers."
order: 50
---

# Publishing contract

`manifest`, `upload` and `verify` publish a rendered tree to an S3-compatible bucket. An object's key is the prefix
joined to the path that an entrypoint file states, and only files that an entrypoint references are published. The
reasons for this design are in [Publishing and caching](../explanation/publishing-and-caching.md). The procedure is in
[Publish the library](../how-to/publish.md).

## Code map

| Concept | File | Symbol |
| --- | --- | --- |
| Bucket key for a path | `param_sweeps/publish.py` | `object_key`, `normalise_prefix` |
| Files a tree publishes | `param_sweeps/publish.py` | `published_relative_paths`, `enumerate_library` |
| Index closure | `param_sweeps/publish.py` | `indexed_relative_paths` |
| Manifest closures | `param_sweeps/publish.py` | `tutorial_manifest_relative_paths`, `alchemy_manifest_relative_paths`, `cards_manifest_relative_paths`, `gallery_manifest_relative_paths` |
| Entrypoint files | `param_sweeps/publish.py` | `ENTRYPOINT_PATHS` |
| Trees and default prefixes | `param_sweeps/publish.py` | `PublishTree`, `DEFAULT_TREE_PREFIXES` |
| Headers | `param_sweeps/publish.py` | `CacheControl`, `ContentType`, `cache_control_for`, `content_type_for` |
| Sync plan and upload | `param_sweeps/publish.py` | `plan_sync`, `apply_sync_plan`, `DEFAULT_UPLOAD_WORKERS` |
| Verification | `param_sweeps/publish.py` | `verify_remote`, `verify_local` |
| Bucket settings | `param_sweeps/publish.py` | `load_s3_settings`, `S3Settings` |

The sync planning, ordering and headers are covered by `tests/test_publish.py`, against a `botocore` stub.

## Addressing

**An object's key is the path that the index or a manifest already states**, under the optional prefix. A consumer
joins its configured base URL (prefix included) to that string:

```text
{evidenceBaseUrl}/index.json
{evidenceBaseUrl}/{strips[].source_image}
{evidenceBaseUrl}/{strips[].control_map}
{evidenceBaseUrl}/{strips[].stops[].file}
{evidenceBaseUrl}/{strips[].stops[].sidecar}
{evidenceBaseUrl}/{strips[].stops[].control_map}
{evidenceBaseUrl}/{strips[].stops[].crop.file}
{evidenceBaseUrl}/{strips[].stops[].fit}
{evidenceBaseUrl}/tutorials/manifest.json
{evidenceBaseUrl}/tutorials/{flows[].directory}/{flows[].files[]}
{evidenceBaseUrl}/alchemy/manifest.json
{evidenceBaseUrl}/alchemy/{entries[].files.source|result|result_text|result_json}
{evidenceBaseUrl}/alchemy/{entries[].source_crop.file|result_crop.file|result_fit}
```

The prefix has no leading or trailing slash. An empty prefix publishes at the bucket root.

## What each tree publishes

| Tree | Entrypoints | Published |
| --- | --- | --- |
| `sweeps` (default) | `index.json` (required), `tutorials/manifest.json`, `alchemy/manifest.json` | Each entrypoint and every file it references. A manifest that is not on disk is skipped. |
| `promo` | `cards.json` (required), `gallery/gallery.json` | The card manifest and every card it names, then the gallery manifest and every image, alchemy result and text file it names, when a gallery exists. |

A file that is not referenced by any entrypoint is not published, for example a provenance `meta.json` beside an alchemy result,
a review batch, or `review.html`. If an entrypoint references a file that is not on disk, the command fails before
anything is uploaded.

The promo tree publishes under the key prefix `promo` unless `--prefix` is given. Its default root on disk is the promo
output root.

## Upload behaviour

- `upload` lists the bucket under the prefix and compares each file's MD5 with the object's ETag. Only new or changed
  files are uploaded.
- Every object is written with a single `put_object` call, never multipart, so the ETag stays equal to the MD5.
- Ordinary objects are uploaded eight at a time.
- Entrypoint files (`index.json`, `tutorials/manifest.json`, `alchemy/manifest.json`, `cards.json`,
  `gallery/gallery.json`) are uploaded last, one at a time, after every other object.
- Objects under the prefix that are not referenced by any entrypoint are orphans. `upload` and `verify` report them. Only
  `upload --delete` removes them.

## Verification

- `verify` checks that every referenced file exists in the bucket with the same MD5. It exits non-zero when anything is
  missing or different. Orphans are reported and do not fail it.
- `verify --local` checks that every referenced file is on disk. It does not need credentials and does not report
  orphans.

## Cache headers

Every object is written with a `Content-Type` and a `Cache-Control` header.

| Objects | `Cache-Control` |
| --- | --- |
| Entrypoint files: `index.json`, `tutorials/manifest.json`, `alchemy/manifest.json`, `cards.json`, `gallery/gallery.json` | `public, max-age=300, must-revalidate` |
| Everything else: images, sidecars and every other referenced file | `public, max-age=604800, stale-while-revalidate=86400` |

| Extension | `Content-Type` |
| --- | --- |
| `.webp` | `image/webp` |
| `.png` | `image/png` |
| `.jpg`, `.jpeg` | `image/jpeg` |
| `.svg` | `image/svg+xml` |
| `.json` | `application/json` |
| `.txt` | `text/plain; charset=utf-8` |
| anything else | `application/octet-stream` |

## Sharp edges

- **Renumbered stops leave orphans.** Inserting a stop renumbers the stops after it, so the old file names become
  orphans in the bucket. Run `upload --delete` after such a change. Cached copies of a renumbered path can show the old
  image for up to a week unless the CDN cache is purged.
- **CORS is the bucket's configuration.** A web app fetches the index and the manifests with `XMLHttpRequest`, which the
  browser refuses without an `Access-Control-Allow-Origin` header. Images load through `<img>` without one. The bucket
  must allow cross-origin `GET` from the consuming site's origin. The tool does not set this.
- **The publish extra is optional.** Without `boto3`, `upload` and `verify` (without `--local`) print what to install
  and exit 1. `manifest` and `verify --local` work without it.
