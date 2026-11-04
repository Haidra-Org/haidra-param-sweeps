---
title: "Measure sampler cost"
summary: "Run the sampler-cost measurement on the local GPU and read the per-step cost ratio of each sampler from its report."
order: 70
---

# Measure sampler cost

`sampler-cost` measures what one step of each sampler costs relative to `k_euler`, on the local GPU. How the
measurement works is in [Sampler cost measurement](../explanation/sampler-cost-measurement.md).

Before you start, set up the local runner as in [Set up the runners](set-up-runners.md). The installed horde-engine must
support every sampler you measure. When you finish, a JSON report is in `measurements/` under the output root.

## 1. Check the plan

```sh
uv run param-sweeps sampler-cost --plan-only
```

This prints each sampler, the schedule it will be measured on and why, and the total number of renders. It does not
measure anything. By default the measurement covers every sampler in `horde_sdk`'s vocabulary, on `deliberate` (SD1.5) and
`albedobase-xl` (SDXL), at 10, 20, 30 and 40 steps, with 3 repeats per point and 1 warmup render per model.

## 2. Run a short measurement first

A short run checks that the setup works before you commit to the full measurement:

```sh
uv run param-sweeps sampler-cost --model deliberate --sampler k_euler --sampler k_dpmpp_2m \
  --steps 10 --steps 20 --repeats 1
```

**Check:** the run prints a summary table and writes a report to `measurements/`.

## 3. Run the full measurement

```sh
uv run param-sweeps sampler-cost
```

`--model`, `--sampler` and `--steps` are repeatable and narrow the run. `--repeats` and `--warmups` change the number of
timed and discarded renders.

**Check:** the summary table lists every sampler with its ratio to `k_euler`. Samplers marked `DISCREPANT` are more
than 20% away from the ratio that their evaluations per step predict. Adaptive samplers are labelled as adaptive and
are never flagged.

## Read the report

Each run writes a new file, named after the run's start time, so a new measurement never overwrites an older one. Each
sampler's record includes the schedule used, every raw timing, the fit and its R-squared, the ratio against `k_euler`,
the predicted ratio, and whether it is discrepant. The report also records the seed and cfg scale, which come from the
sweep specification.

A discrepancy is a finding. The tool does not adjust anything because of it.
