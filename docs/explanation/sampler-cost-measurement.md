---
title: "Sampler cost measurement"
summary: "How sampler-cost separates the cost of one sampling step from model load and per-job overhead, and how it compares the result with the SDK's expectation."
order: 40
---

# Sampler cost measurement

`sampler-cost` does not render any part of the library. It measures what one step of each sampler costs relative to
`k_euler`, through the same pipeline the library renders on. It does this by fitting a line through renders at several
step counts, so the per-step cost is separated from everything that does not scale with steps.

The procedure is in [Measure sampler cost](../how-to/measure-sampler-cost.md), and the flags are in the
[command line reference](../reference/cli.md#sampler-cost).

## Separating the per-step cost

A cost taken from a single render is three numbers added together, and only one of them depends on the sampler:

- the model load;
- the fixed overhead that every job pays, whatever the sampler;
- the sampling itself.

The measurement renders the same job at several step counts (10, 20, 30 and 40 by default) and fits an ordinary least
squares line through the timings. The slope is the cost of one step. The intercept is everything that does not scale
with steps. The ratio of a sampler's slope to the slope of `k_euler` is the figure that a cost model needs.

## Holding everything else constant

Within each model, only the sampler differs between measurements:

- One seed, one prompt, one negative prompt and one cfg scale throughout. The seed and cfg scale come from the sweep
  specification and are recorded in the report.
- Each model renders at its native size.
- Discarded warmup renders run after every model load, before anything is timed, so the first render's graph building
  does not end up in the intercept.
- Each point is the median of its repeats, so a single interrupted render cannot move it.
- Every sampler is measured on the `karras` schedule, because a measurement whose schedule changes with the sampler
  measures two things at once. If the SDK's constraints table does not allow a sampler on `karras`, the sampler is
  measured on the first schedule the table recommends instead, and the report records the substitution.

The prompt does not need to depict anything in particular. It needs to stay identical, because prompt length changes
the conditioning cost that appears in the intercept.

## The sampler vocabulary comes from the SDK

The list of samplers and the schedule pairings are read from the image sampler constraints table in `horde_sdk`
(`horde_sdk.generation_parameters.image.constraints`), so a sampler added there is measured here without changing this
tool. The minimum supported `horde_sdk`, 0.29.3, includes that table. The installed horde-engine must also support the samplers
being measured. If the table is missing, the command prints an error and exits.

## Expected ratios and discrepancies

For each sampler with a fixed rate of model evaluations per step, the SDK's work profile states that rate, which
predicts the sampler's ratio to `k_euler`. A measured ratio more than 20% away from that prediction is flagged as
discrepant. The tool does not adjust anything. The flag is the finding.

An adaptive sampler, such as `k_dpm_adaptive`, sets its own number of iterations. It is still measured, but it does not
have an expected ratio and is never flagged. The plan and the summary table label it as adaptive.
