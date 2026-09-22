# Measured v2 release validation — 2026-09-22

Release: `v2-measured-20260922`. Graph contract: `scyllasband_measured_delivery_v1`. FP32 ONNX is the reference/default; INT8 is optional.

## Checkpoints

- Duration: completed `v3_measured_full_20260920/duration/best.pt`, epoch 16, SHA-256 `5e62af2e78d5578f27c9a259cf3c9739c59adfe0c432277e8c45fbd33ac2d9db`.
- Vector: completed `v3_measured_full_20260920/vector/best.pt`, epoch 20, SHA-256 `36f630770e4023d5efbb4f4b3c463122ccca5af8abb76c2600140a4beae73bfd`.
- Matched frozen frontend adapter/autoencoder, trained G2P plus its tokenizer/vocabulary, and cached Vocos weights. Both training stages completed 20 epochs; no training was restarted for release.

## Automated and real-model checks

- 148 targeted parent-project tests: downloader, streaming, runtime, metadata comparison, chunked synthesis, quantization and v2 staging.
- 47 inference-repository tests, including measured defaults, absent-coordinate semantics, binary whisper, invalid requests, model-family separation, contract validation, group planning, cached-model notices and safe download replacement.
- 10 native CTest tests, including new measured request cases.
- Real FP32 synthesis: all ten voices in their English dialect, plus Spanish, Italian, French, German and Vietnamese; two mixed-control requests, whisper and all-absent controls.
- Real INT8 synthesis: the ten-voice English set. Phones and rounded durations match FP32 for all ten takes. Waveforms differ; no perceptual-equivalence claim or default promotion.
- Real Linux native ONNX synthesis: measured FP32/INT8 and v1. Python/native measured phones and rounded durations match on the comparison request.
- Real three-sentence CLI synthesis: three chunks, preserving the measured request and its omitted assertiveness coordinate.
- V1 exact-phone Python synthesis compared with the original dev code (`2538a67`): output PCM is bit-for-bit identical. This isolates v1 acoustic inference from unrelated frontend edits in the workspace.

Direct original-checkpoint comparisons exercised four voices, neutral, absent, mixed and whisper conditions. Across these cases, maximum absolute duration error was `2.3842e-5` frames; vector error was `1.2458e-5`. Comparing padded and unpadded vector execution gave maximum absolute error `2.6227e-6`. This also checks the export's portable RMS normalization against the original checkpoint implementation.

Export initially exposed a comma-pause frontend mismatch. The measured release opts into comma-preserving G2P segmentation in Python and native; legacy v1 segmentation is unchanged. Real v1 native synthesis exposed a scalar condition-mask shape issue, now dispatched correctly by the legacy axis-order contract.

## Platforms and limits

Linux Python/native ONNX paths were executed. Android control/parser logic was separately compiled and exercised, but the full Android build requires an SDK absent from the validation machine. Apple bridge and UI changes require Xcode/device validation. No measured LiteRT/Core AI model is advertised.

Component and numerical checks establish execution/parity within their tested scope. They do not prove naturalness, correct pronunciation on arbitrary text, emotional extremes, or quality equivalence under INT8. Human listening remains the acceptance criterion for those claims.

## Reproduction

Trainer workspace export:

```bash
PYTHONPATH=scyllasband-trainer:scyllasband OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
venv/bin/python -m scyllasband_trainer.export.measured \
    --config scyllasband-trainer/configs/scyllasband_v3_measured_full.json \
    --base output/scyllasband/scyllasband_v2_clean_202608/bundle_onnx_fp32_v2 \
    --output output/scyllasband/release_v2_measured_20260922/onnx
```

Inference repository tests:

```bash
PYTHONPATH=. python -m unittest discover -s tests
ctest --test-dir libscyllasband/build --output-on-failure
```

Local validation audio and metadata are retained under `output/scyllasband/release_v2_measured_20260922/validation` in the trainer workspace. That directory includes original-checkpoint parity, FP32/INT8 comparisons, v1 baseline/current samples and the multi-sentence CLI output. Audio and checkpoints are not committed to the runtime repository.
