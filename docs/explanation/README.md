---
title: "Explanation"
summary: "The design of the sweep library, QR rendering, the alchemy evidence, the sampler cost measurement and publishing."
---

# Explanation

These documents describe why `param-sweeps` works the way it does. Exact field lists and commands are in the
[reference](../reference/README.md).

- [One parameter per strip](one-parameter-per-strip.md): the shared worlds and scenes, the presentations, the
  exceptions to one-parameter strips, and how the masked and hires modes behave on each baseline.
- [QR codes that scan](qr-codes.md): what determines whether a QR render decodes, and how every published code is checked.
- [Alchemy evidence](alchemy-evidence.md): why every form in a use case shares one source, why the run uses three
  backends, and why a form that could not run is still listed.
- [Sampler cost measurement](sampler-cost-measurement.md): how a per-step cost is separated from the fixed costs of a
  render.
- [Publishing and caching](publishing-and-caching.md): the addressing contract, why sidecars are published, the
  upload order and the cache headers.
