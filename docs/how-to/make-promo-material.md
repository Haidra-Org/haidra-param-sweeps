---
title: "Make share cards and the promo gallery"
summary: "Draw share cards and posters from the rendered library, and make the promo gallery of pictures, texts and conversations through the live horde."
order: 60
---

# Make share cards and the promo gallery

Promotional material is written under the promo root, `output/promo/` unless `PARAM_SWEEPS_PROMO_OUTPUT_ROOT` is
set, beside the library and never inside it. It is published as a separate tree. This guide covers the two
commands that make it: `promo`, which draws cards from renders already on disk, and `gallery`, which makes new pictures,
texts and conversations through the live horde.

The specification formats are in the [campaign specifications reference](../reference/campaign-specifications.md),
and the files are in the [output tree reference](../reference/output-tree.md#promo-tree).

## Draw share cards

A link to a page unfurls as whatever its `og:image` names. A page whose evidence is a strip of renders has a better
picture to offer than a logo. `promo` draws link previews, square and tall cards for feeds, and large grids for a forum
post. Every picture on a card is a published frame.

Before you start, render the strips and tutorial flows the cards use, and write a promo specification with your brand,
fonts and the cards you want. This repository does not include one.

```sh
# Draw everything the specification requests
uv run param-sweeps promo --promo-spec <promo_spec.json>

# One card, by key
uv run param-sweeps promo --promo-spec <promo_spec.json> --job sweep:steps

# Redraw cards already on disk
uv run param-sweeps promo --promo-spec <promo_spec.json> --force
```

Cards are read from the library at `--output-root` and written to `--promo-root`, with `cards.json` at its root naming
every card by key. Cards already on disk are skipped unless `--force` is given.

**Check:** `cards.json` lists each key, and its file exists under the promo root.

## Make the promo gallery

The cards are cut from the sweep library, which shows the effect of settings. It does not show what the horde makes on
a good day. `gallery` makes that kind of material: promo pictures, post-processing pairs, short texts and short
conversations, all through the live horde, each recorded with its cost. The default specification is
`gallery_spec.json` at the repository root.

Before you start, set `AIHORDE_DEV_KEY` as for the horde runner. Runs cost kudos.

1. See what would be submitted, with each picture's model and settings:

   ```sh
   uv run param-sweeps gallery --dry-run
   ```

   Even a dry run queries the horde, because the model chosen for a picture depends on what the horde is serving at
   that moment.

2. Run it:

   ```sh
   # Everything in the specification that is not on disk yet
   uv run param-sweeps gallery

   # Only some entries
   uv run param-sweeps gallery --only fox-snow,airship

   # Pictures only
   uv run param-sweeps gallery --skip-text --skip-conversations
   ```

   Entries already on disk are not made again. Up to 25 requests are held open at once. `--skip-alchemy` leaves out the
   alchemy forms requested by picture entries.

3. **Check:** `gallery/gallery.json` under the promo root lists every entry on disk. It is rebuilt from disk on every
   run, so a run narrowed by `--only` or `--skip-text` still produces a complete manifest.

### How texts and conversations are sent

Texts and conversations are sent to a completion endpoint, so the tool builds the prompt format itself. An
instruction is wrapped in the Alpaca
prompt format and cut where the format would start another turn, so each single-turn text records
`format: "instruct:alpaca"`. A conversation is run turn by turn, and each turn includes the turns before it. An
`instruct` conversation sends them as instruction and response pairs inside the format. A `raw` one sends the bare text
so far, so the model continues it instead of replying to it. Every turn's request body, reply, model, seconds and kudos
are kept in the entry's `meta.json`, beside a `transcript.txt`.

### Entries refused by the safety checker

A refused entry leaves a `censored.json` with the request and the time, and is listed under `censored` in the manifest.
Later runs skip it, because the refusal is about the prompt and paying for it again does not change the result.
After reworking the prompt, submit it again:

```sh
uv run param-sweeps gallery --retry-censored
```

## Publish the promo tree

See [Publish the promo tree](publish.md#publish-the-promo-tree).
