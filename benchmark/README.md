# Scylla's Band Benchmarks

This directory contains versioned, reproducible results from the frozen public
long-form suite in `data/testing/long_form`. Each release report keeps measured
WER, ASR word edits, generated audio, and optional human listening labels
together. WER is an intelligibility signal; it does not by itself measure
pronunciation, affect, pacing, speaker identity, or audio artifacts.

[Open the interactive benchmark](https://lowkeytea.github.io/scyllasband/benchmark/).

## Shipping v1 matrix

The v1 core benchmark uses the shipping ONNX bundle and its declared
capabilities:

- all 10 managed voices;
- each voice's default English dialect, Spanish, and Italian;
- neutral, calm 4, joy 2/4, anger 2/4, sadness 2/4, and questioning 2;
- one fully held-out long-form narrative per language;
- deterministic 8-step Heun synthesis; and
- multilingual `large-v3` ASR for the published WER and transcript diff.

That is 270 planned voice/language/condition jobs. The report identifies itself
as in progress until all planned jobs have rendered; missing ASR is shown as
pending rather than scored as zero.

## Reproduce

Install the validation dependency, download the shipping ONNX bundle, and run:

```bash
ASR_PYTHON=python ./scripts/run_shipping_v1_benchmark.sh
```

The script uses two bounded ONNX render processes, resumes completed jobs, runs
four `large-v3` ASR workers, and refreshes both the repository artifact and the
Pages copy. The equivalent individual commands are:

```bash
python -m scyllasband.validation plan \
    --bundle scyllasband/models/onnx \
    --tier core \
    --voice ariadne --voice felix --voice gwen --voice ink --voice max \
    --voice orpheus --voice rex --voice scylla --voice stone --voice tuesday \
    --output validation_runs/shipping-v1-core

python -m scyllasband.validation render \
    --run validation_runs/shipping-v1-core \
    --backend onnx --bundle scyllasband/models/onnx

python -m scyllasband.validation asr \
    --run validation_runs/shipping-v1-core \
    --profile release --workers 4

python -m scyllasband.validation report \
    --run validation_runs/shipping-v1-core

python -m scyllasband.validation publish \
    --run validation_runs/shipping-v1-core \
    --output benchmark/v1 \
    --pages-output docs/benchmark/v1
```

The validation work directory retains lossless WAV files and complete render
metadata. The versioned public artifact uses 96 kbit/s MP3 review audio, a
sanitized result payload, summary JSON/CSV, and a self-contained static report.
