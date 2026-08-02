# Long-form validation corpus

This directory contains the redistributable text fixtures used by
`python -m scyllasband.validation`. The suite is deliberately small enough for
a local smoke run while still forcing multiple long-form chunks.

The selected bundle manifest controls the actual matrix. Jobs are created only
for voice/language cells declared by that bundle. Affect conditions are also
capability-driven: a v1 affect bundle runs `questioning_2` and skips
`whisper_2`; a v2 bundle does the reverse. The runner uses
`controls.affect.axis_order_version` and `axes`, never the release-number
string, to make that decision.

All included passages were written specifically for this public test suite and
are licensed under the repository's Apache-2.0 license. Generated audio, ASR
artifacts, and review labels belong in `validation_runs/`, not this directory.

