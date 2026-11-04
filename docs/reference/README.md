---
title: "Reference"
summary: "The specification formats, the command line, the environment variables, the output tree and the publishing contract."
---

# Reference

These documents state the contracts of `param-sweeps`. Each has a code map from concept to file and symbol. The code
is the authority on behaviour.

- [Sweep specification](sweep-specification.md): the format of `sweep_spec.json`, from anchor parameters to stops,
  source images, masks, crops and the rules checked on load.
- [Campaign specifications](campaign-specifications.md): the formats of `flows_spec.json`, `alchemy_spec.json`,
  `gallery_spec.json` and the promo card specification.
- [Review sheet](review-sheet.md): batches, review questions, the contact sheet, `choices.json` and exported answers.
- [Command line](cli.md): every subcommand, its flags and defaults.
- [Environment variables](environment.md): every variable read or set, including the output roots and the bucket
  settings.
- [Output tree](output-tree.md): `index.json`, sidecars, crops, the alchemy and tutorial trees, measurements and the
  promo tree.
- [Publishing contract](publishing.md): object keys, what each tree publishes, upload order and cache headers.
