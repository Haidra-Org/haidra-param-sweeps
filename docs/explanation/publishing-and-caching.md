---
title: "Publishing and caching"
summary: "Why the library lives in object storage, why an object's key is the path the index states, and how upload order and cache headers keep consumers consistent."
order: 50
---

# Publishing and caching

The rendered library is around 70 MiB across 1400 files, so it is not committed to git. `output/` is ignored, and the
published copy lives in S3-compatible object storage. The design keeps three properties: a consumer never translates
paths, a bucket never serves an index that names files which have not arrived yet, and caches hold images for a week
while the entrypoint files stay fresh.

The exact contract is in the [publishing reference](../reference/publishing.md), and the procedure is in
[Publish the library](../how-to/publish.md).

## Any S3-compatible host

The intended host is Cloudflare R2, which is a plain S3 API behind an account endpoint URL. The tool does not
depend on anything specific to R2. An AWS bucket or MinIO works the same way with a different endpoint.

## Keys are the paths the index states

An object's key is the path that the index or a manifest already states. A consumer joins its configured base URL to
that string and has the file. Neither the tool nor the consumer translates paths, so moving the library to another host, or between a
prefix and the bucket root, is a change to one configured string. The optional prefix is applied when the objects are
written, so it belongs to the base URL and never to a path from the index.

## Entrypoints define what is published

Three entrypoints define what belongs to the library: `index.json` for the sweeps, `tutorials/manifest.json` for the
worked examples and `alchemy/manifest.json` for the alchemy evidence. A file on disk that is not referenced by any of them
is not published. Examples are a provenance `meta.json` beside an alchemy result, a review batch and the review sheet. An
object in the bucket that is not referenced by any of them is an orphan.

**Sidecars are published.** The index names one sidecar per stop. If those keys returned 404, the index would point at
files that do not exist, and `verify` would need a special case to ignore them. Sidecars are also cheap: at about a
kilobyte each they are about 2% of the library. The web side does not have to fetch them. They are published so that
every path the index names resolves, without exceptions.

## Upload order

Ordinary objects are uploaded eight at a time. Each is one small request, so a serial upload is limited by round trips
instead of bandwidth. **The index and the manifests are uploaded last**, one at a time, after every other object is in
the bucket. A bucket therefore never serves an entrypoint that names files which have not arrived yet. The worst that an
interrupted upload leaves behind is unreferenced objects, and the next run reconciles them.

`upload` is incremental. It lists the bucket, compares each file's MD5 with the object's ETag, and uploads only what is
new or changed. A re-run after a render that did not change anything does not upload anything. Objects are written
with `put_object`, so every upload is single-part and the ETag stays equal to the MD5. A multipart upload would produce a different ETag
and break the comparison.

## Cache headers

Images, sidecars and everything else that a manifest names are cached for a week. The entrypoint files are cached for
five minutes. The exact headers are in the [publishing reference](../reference/publishing.md#cache-headers).

Images are **not** immutable, so the week-long header is a deliberate trade-off and does not claim `immutable`.
Inserting a stop into a strip renumbers the stops after it, so a path can come to hold a different image. A week is long
enough that consumers do not refetch between publishes, and short enough that a renumbering settles without
intervention. A publish that renumbers stops and must be visible at once needs a cache purge at the CDN. The header stays at
one week. The index and the manifests are the mutable files that every consumer starts from, so they are cached for five
minutes and no longer.

## The promo tree is published separately

Share cards and the promo gallery are promotional material, and pages do not cite them as evidence. They are published
as their own tree under the `promo` key prefix by default, so the library's prefix holds only what a page cites as
evidence.

## The object-storage SDK is optional

A rendering environment often shares its interpreter with a GPU stack unrelated to object storage, so installing the
tool there must not pull in an SDK. `boto3` is in the optional `publish` extra, and
`param_sweeps.publish` imports it where it builds a client instead of at the top of the module. Every command that does
not reach a bucket, including `manifest` and `verify --local`, runs without the extra. A command that does reach one
prints what to install and exits 1.

`tests/test_publish.py` includes a test that keeps the extra optional. It imports the tool in a subprocess and fails if
`boto3` or `botocore` is in `sys.modules`.

## Secrets

The API key and the bucket secret are never printed, logged or written into any output. The API key is only put in a
request header, and the bucket secret is only passed to the S3 client.
