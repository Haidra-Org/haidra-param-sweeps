---
title: "Publish the library"
summary: "Configure the bucket, check the rendered library and its QR codes, upload it, verify it, remove orphans, publish the promo tree and point a web app at it."
order: 80
---

# Publish the library

This guide publishes the rendered library to an S3-compatible bucket, such as Cloudflare R2. Before you start, render
the library so that `index.json` exists under the output root. When you finish, `verify` confirms that every file the
index and the manifests reference is in the bucket with the same content.

The contract (keys, entrypoints, upload order, headers) is in the [publishing reference](../reference/publishing.md).
The reasons for it are in [Publishing and caching](../explanation/publishing-and-caching.md).

## 1. Install the publish extra

`uv sync` in this repository already installs it. In another environment:

```sh
uv sync --extra publish
# or
pip install "haidra-param-sweeps[publish]"
```

Without the extra, `upload` and `verify` print what to install and exit 1. `manifest` and `verify --local` work
without it.

## 2. Configure the bucket

Set the bucket variables in the environment or in the `.env` file. The `.env` is the one given with `--env-file`, else
the one named by `PARAM_SWEEPS_ENV_FILE`, else the `.env` at the repository root, which also holds the API key and is
ignored by git.

```sh
PARAM_SWEEPS_S3_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
PARAM_SWEEPS_S3_BUCKET=<bucket>
PARAM_SWEEPS_S3_ACCESS_KEY_ID=<access-key-id>
PARAM_SWEEPS_S3_SECRET_ACCESS_KEY=<secret-access-key>
# Optional
PARAM_SWEEPS_S3_PREFIX=<prefix>
PARAM_SWEEPS_S3_REGION=auto
```

The access key needs object read and write permission. For an AWS bucket, set the bucket's real region. The full list
is in the [environment reference](../reference/environment.md#bucket-variables).

## 3. Check the library on disk

```sh
# What is in the library, and how large it is
uv run param-sweeps manifest

# Confirm that the files on disk match what the index and manifests reference. No credentials needed.
uv run param-sweeps verify --local

# Confirm that every published QR code still decodes to its payload
uv run param-sweeps qr-check
```

**Check:** `verify --local` and `qr-check` both exit 0.

## 4. Upload

```sh
# What an upload would do, without touching the bucket
uv run param-sweeps upload --dry-run

# Upload
uv run param-sweeps upload
```

Only new or changed files are uploaded. The index and manifests are uploaded last. Running the upload again after a
render that did not change anything does not upload anything.

## 5. Verify the bucket

```sh
uv run param-sweeps verify
```

This compares every referenced file with the bucket by content hash. It exits non-zero if anything is missing or
different, so you can use it as a gate. Orphans are reported but do not fail it.

## Remove orphans

`upload` and `verify` report objects under the prefix that are not referenced by the index or a manifest. They do not
remove them unless you pass `--delete`. Orphans appear after stops are inserted into a strip, because inserting a stop renumbers the
stops after it and the old file names become orphans.

```sh
# Read this output first
uv run param-sweeps upload --delete --dry-run

uv run param-sweeps upload --delete
```

Deleted objects cannot be restored by the tool. Re-rendering and uploading recreates any that were still needed.

## Publish the promo tree

`manifest`, `upload` and `verify` take `--tree promo` to work on the share cards and gallery instead of the library.
The promo tree is read from the promo root on disk and published under the key prefix `promo`, unless `--prefix` is
given. It needs `cards.json`, so draw the cards first.

```sh
uv run param-sweeps verify --tree promo --local
uv run param-sweeps upload --tree promo --dry-run
uv run param-sweeps upload --tree promo
uv run param-sweeps verify --tree promo
```

## Point a web app at the bucket

Configure the web app with the bucket's public base URL, prefix included. The app joins that URL to the paths in the
index and the manifests. The setting belongs to the consuming app. This repository does not set it.

The bucket must allow cross-origin `GET` from the site's origin. The app fetches the index and the manifests with
`XMLHttpRequest`, which the browser refuses without an `Access-Control-Allow-Origin` header. Images load through `<img>`
without one. Set this as a CORS policy on the bucket.

**Check:** fetch `<base-url>/index.json` from the site's origin. The response has an `Access-Control-Allow-Origin`
header and a `Cache-Control` of `public, max-age=300, must-revalidate`.

## Publish a renumbering at once

Images are cached for a week. If a publish renumbers stops and the change must be visible immediately, purge the CDN
cache for the affected paths after `verify` passes.
