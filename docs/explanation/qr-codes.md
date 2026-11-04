---
title: "QR codes that scan"
summary: "What determines whether a QR ControlNet render decodes, and how every published QR image is checked with zbar."
order: 20
---

# QR codes that scan

Two strips and one tutorial flow render through the `qr_code` workflow, which lays a QR ControlNet over an ordinary
render. Three things determine whether the result is worth publishing: the prompt, the control strength, and whether
the worker accepts the strength at all. Every published code is then decoded by zbar.

To run the check, see [Publish the library](../how-to/publish.md). The `qr-check` subcommand is described in the
[command line reference](../reference/cli.md#qr-check).

## The prompt must not describe the code

The ControlNet already supplies the geometry. A prompt that describes blocks, grids and hard black gaps competes with
it, and the result is a barcode with weather behind it instead of a picture with a code inside it. The prompts here
describe vapour, falling leaves and climbing ivy: soft, flowing subjects that the modules can be made of. The
`cloud-sea` scene exists for this reason, and it does not have a straight line in it.

The tutorial flows show the same effect. A skyline, a garden or a shrine named in the prompt takes the pixels that the
code needs, and the result does not decode. Weather and light alone, with the place named only as somewhere far below,
scanned on the first seed tried.

## Control strength

`control_strength` is the main control. At low strength the picture dominates until no scanner can detect the code. At
high strength the code sits on the picture like printed ink. The usable range is narrower than it looks, and it changes
with the scene. On `cloud-sea` at 1024 pixels with AlbedoBase XL, values at or below 1.2 did not decode on any seed
tried, 1.25 is the lowest value that decodes, and by 1.6 the code dominates the picture. The `control_strength` strip
publishes that range stop by stop.

## Where the horde accepts the strength

The horde accepts `control_strength` together with a `control_type` or the `qr_code` workflow, and refuses a request
that sets the strength without either. A job that includes it is only sent to a recently updated worker. Older workers
do not receive it. The strength strips here were rendered locally, which produces the same picture that a current
worker returns.

## Every published code is decoded

A contrast heuristic is not enough to tell whether a code scans. Every code is decoded by `zbar`, through `pyzbar`,
against the image exactly as published and at the published size.

- A single-image render that does not decode fails in the campaign driver and is not written.
- A review batch is not failed that way, because its published frame is whichever candidate a person picks. The
  decode result of every candidate is logged instead.
- `param-sweeps qr-check` scans every QR image referenced by the index and the tutorial manifest, and exits non-zero
  if any of them decodes to anything other than its requested payload. This is the check over what is actually
  published.

`pyzbar` is a base dependency, unlike `boto3`. It is a few hundred kilobytes, including the bundled `zbar` library on
Windows, and it is needed wherever QR images are rendered as well as wherever they are checked. Without a decoder, a
rendering environment could not check whether its own QR output scans, and could publish codes that do not scan.
Keeping `pyzbar` in the base install prevents that.

## Seeds and models in the tutorial flows

Even with a good prompt, the recipe does not guarantee a decode. The QR flows were rendered over a run of seeds, and
the first seed that decoded was published. The seed recorded in each flow is that seed. The SDXL anime checkpoint
draws flat areas, and it erased the modules on every seed tried. The SD1.5 anime checkpoint at 512 pixels, with the
SD1.5 QR ControlNet, decoded on three seeds out of four, and the fox-shrine QR flow uses it.
