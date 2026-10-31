# Model weights license

This file covers **model weights** derived from this repository — any
checkpoint under `models/` or `runs/` (teacher-distilled, quantized,
pruned, or GGUF-exported), wherever it's actually distributed (Hugging
Face Hub, GitHub Releases, etc., since weights are gitignored and never
committed to this repo directly). It is separate from `LICENSE`, which
covers only the source code.

## Two independent reasons this is noncommercial-only

1. **By choice.** The weights are licensed under the terms below —
   [PolyForm Noncommercial 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0),
   the same terms as the code, applied here to the weights as a distinct
   licensable artifact.
2. **By necessity, independent of that choice.** Training data provenance
   constrains this regardless: see [DATA_LICENSES.md](DATA_LICENSES.md).
   CORD is confirmed CC BY 4.0 (attribution required, commercial use
   otherwise permitted upstream). **SROIE's dataset-level license could
   not be independently verified** and is treated as research-use-only
   pending confirmation from the ICDAR/RRC organizers — any checkpoint
   trained on SROIE (which includes every distilled/compressed variant in
   this project) inherits that same research-use-only constraint, and
   cannot be relicensed for commercial use even if the license below were
   changed. If SROIE's terms are later confirmed to permit commercial
   redistribution, this file should be revisited — but the default until
   then is noncommercial.

## Attribution required in any published model card

Per `DATA_LICENSES.md`, any model card (Hugging Face or otherwise)
publishing weights trained on CORD and/or SROIE must name both datasets,
attribute NAVER CLOVA for CORD, and link `DATA_LICENSES.md`.

## Required Notice

> Required Notice: Copyright iivek (LedgerLens) — model
> weights trained on CORD (CC BY 4.0, NAVER CLOVA) and SROIE (ICDAR 2019
> RRC Task 3, license unconfirmed, research-use-only)

## License terms

<https://polyformproject.org/licenses/noncommercial/1.0.0>

### Acceptance

In order to get any license under these terms, you must agree
to them as both strict obligations and conditions to all
your licenses.

### Copyright License

The licensor grants you a copyright license for the
model weights to do everything you might do with them
that would otherwise infringe the licensor's copyright
in them for any permitted purpose. However, you may
only distribute the weights according to [Distribution
License](#distribution-license) and make changes or new works
based on them according to [Changes and New Works
License](#changes-and-new-works-license).

### Distribution License

The licensor grants you an additional copyright license
to distribute copies of the weights. Your license
to distribute covers distributing the weights with
changes and new works permitted by [Changes and New Works
License](#changes-and-new-works-license).

### Notices

You must ensure that anyone who gets a copy of any part of
the weights from you also gets a copy of these terms or the
URL for them above, as well as copies of any plain-text lines
beginning with `Required Notice:` that the licensor provided
with the weights (see above).

### Changes and New Works License

The licensor grants you an additional copyright license to
make changes and new works based on the weights (including
further fine-tuning, quantization, or pruning) for any
permitted purpose.

### Noncommercial Purposes

Any noncommercial purpose is a permitted purpose.

### Personal Uses

Personal use for research, experiment, and testing for
the benefit of public knowledge, personal study, private
entertainment, hobby projects, amateur pursuits, or religious
observance, without any anticipated commercial application,
is use for a permitted purpose.

### Noncommercial Organizations

Use by any charitable organization, educational institution,
public research organization, public safety or health
organization, environmental protection organization,
or government institution is use for a permitted purpose
regardless of the source of funding or obligations resulting
from the funding.

### Fair Use

You may have "fair use" rights for the weights under the
law. These terms do not limit them.

### No Other Rights

These terms do not allow you to sublicense or transfer any of
your licenses to anyone else, or prevent the licensor from
granting licenses to anyone else. These terms do not imply
any other licenses.

### Violations

The first time you are notified in writing that you have
violated any of these terms, or done anything with the weights
not covered by your licenses, your licenses can nonetheless
continue if you come into full compliance with these terms,
and take practical steps to correct past violations, within
32 days of receiving notice. Otherwise, all your licenses
end immediately.

### No Liability

***As far as the law allows, the weights come as is, without
any warranty or condition, and the licensor will not be liable
to you for any damages arising out of these terms or the use
or nature of the weights, under any kind of legal claim.***

### Definitions

The **licensor** is the individual or entity offering these
terms, and the **weights** are the model weights the licensor
makes available under these terms.

**You** refers to the individual or entity agreeing to these
terms.

**Your company** is any legal entity, sole proprietorship,
or other kind of organization that you work for, plus all
organizations that have control over, are under the control of,
or are under common control with that organization.

**Your licenses** are all the licenses granted to you for the
weights under these terms.

**Use** means anything you do with the weights requiring one
of your licenses.
