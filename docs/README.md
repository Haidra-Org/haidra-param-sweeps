---
title: "haidra-param-sweeps"
summary: "Routes to the explanation, reference and how-to documents for the parameter-sweep renderer."
---

# haidra-param-sweeps

`param-sweeps` renders the AI Horde parameter-sweep illustration library from a declarative specification, through
the public horde or on a local GPU, and publishes it to S3-compatible object storage.

- [Explanation](explanation/README.md): the design of the library and the reasons for it. Why each strip varies one
  parameter, how QR codes are made to scan, how the alchemy evidence and the sampler cost measurement are built, and
  how publishing and caching work.
- [Reference](reference/README.md): exact contracts. The specification formats, every subcommand and flag, the
  environment variables, the output tree and the publishing contract.
- [How-to](how-to/README.md): procedures. Setting up each runner, rendering and regenerating strips, reviewing a
  batch, the alchemy, tutorial, share card and gallery campaigns, measuring sampler cost, and publishing.

The repository [README](https://github.com/Haidra-Org/haidra-param-sweeps/blob/main/README.md) covers installation
and a first run.
