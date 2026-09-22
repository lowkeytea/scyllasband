# Scylla's Band

Scylla's Band is a local TTS runtime for ten managed voices. It predicts phone durations, generates continuous acoustic latents with rectified flow, and decodes speech at 24 kHz through an acoustic adapter and Vocos.

The **dev branch's replacement v2** uses measured delivery controls: **energy, tension, valence, assertiveness**, each from **0 to 4**, with **2 as neutral**, plus **whisper on/off**. It replaces the earlier emotion-based v2 development model. The runtime still supports v1 with its original controls.

## Quick start

From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install numpy huggingface_hub onnxruntime

python -m scyllasband download --yes
python -m scyllasband validate-bundle
python -m scyllasband list-voices
python -m scyllasband speak --voice scylla --language en_us \
    --delivery energy=2,tension=2,valence=2,assertiveness=2,whisper=off \
    --sampler heun --steps 8 -o hello.wav "Hello from Scylla's Band."
```

Downloads default to the FP32 ONNX bundle from `spybyscript/scyllasbandv2`, pinned to release `v2-measured-20260922`. It installs under `scyllasband/models/v2/onnx`. Voice/reference conditioning is embedded in the measured graphs; separate voice packs are unnecessary.

## Delivery controls

| Control | Request | Default |
| --- | --- | --- |
| Energy | 0–4 or `auto` | 2 |
| Tension | 0–4 or `auto` | 2 |
| Valence | 0–4 or `auto` | 2 |
| Assertiveness | 0–4 or `auto` | 2 |
| Whisper | `on`, `off`, or `auto` | `off` |

```bash
python -m scyllasband speak --voice ariadne \
    --delivery energy=1.2,tension=2.8,valence=1,assertiveness=3 \
    --sampler heun --steps 8 -o mixed.wav "I thought you might come back."

python -m scyllasband speak --voice ink --language en_gb \
    --delivery whisper=on -o whisper.wav "Keep your voice down."
```

Unspecified axes use neutral defaults. `--delivery auto` leaves every coordinate unspecified; `energy=auto` leaves only that coordinate unspecified. An unspecified coordinate has a separate presence mask and is distinct from requesting 2. It does **not** invoke an automatic performance planner.

The four axes interact. Requested values are learned conditioning, not guaranteed scorer outputs or named emotions. Their useful range varies with voice, language and text. Start near 2 and listen when moving toward the extremes. Expressiveness was an auxiliary training target, not a fifth inference slider.

Measured v2 rejects `--emotion`, legacy `--affect`, and emotion CFG. The Python request's separate `energy` parameter remains a waveform-gain control; use `delivery={"energy": ...}` for the new learned energy coordinate.

## Upgrading or keeping v1

An installed emotion-based v2 is detected from its manifest. Synthesis displays a replacement notice. Downloading v2 stages and validates the new bundle before replacing the selected installation; a failed download leaves the installed bundle intact.

```bash
# Replace superseded v2; leave v1 installed.
python -m scyllasband download --model-version v2 --yes

# Install or use v1 explicitly, retaining its original affect controls.
python -m scyllasband download --model-version v1 --runtime-bundles onnx --yes
python -m scyllasband speak scyllasband/models/v1/onnx \
    --voice scylla --emotion calm=0.5,joy=0.25 -o v1.wav "Hello again."
```

Automatic selection prefers an installed measured v2 over superseded v2 or v1. Explicit bundle paths always select that bundle. V1 can coexist in `models/v1`, and legacy flat v1 installations remain discoverable. V1 is removed only through the explicit `--delete-v1` option or an opted-in interactive deletion choice. See the [v1 model card](MODEL_CARD.md) for its original model contract.

## Backends

| Release | Default download | Other options |
| --- | --- | --- |
| Measured v2 | `onnx` (FP32) | `onnx-int8` for explicit comparison |
| V1 | `onnx-int8`; Core AI also on supported Apple hosts | Existing FP32, LiteRT and Core AI bundles |

Both measured variants use `--backend onnx`. INT8 quantizes the vector estimator's QKV and feed-forward layers and uses the previously exported quantized G2P. Duration, context and vocoder components remain FP32. INT8 produces different waveforms; numerical execution checks do not establish perceptual equivalence. FP32 is the release reference.

```bash
python -m scyllasband download --model-version v2 --runtime-bundles onnx-int8 --yes
python -m scyllasband speak scyllasband/models/v2/onnx-int8 \
    --voice rex --delivery energy=2.5 -o comparison.wav "Let's try this again."
```

For measured v2, `both` and `all` select FP32 and INT8 ONNX. No measured LiteRT or Core AI bundle is published in this release. V1 retains its backend choices. Each bundle's manifest is authoritative.

## Python API

```python
from scyllasband import ScyllasBandRuntime, SynthesisRequest

runtime = ScyllasBandRuntime.from_bundle("scyllasband/models/v2/onnx", backends=["onnx"])
result = runtime.synthesize(SynthesisRequest(
    text="It is a real pleasure to meet you.",
    voice_id="ariadne",
    language="en_us",
    delivery={"energy": 2.3, "tension": 1.7, "valence": 2.6,
              "assertiveness": 2.0, "whisper": "off"},
    sampler="heun", steps=8, seed=2027,
))
# result.audio, result.sample_rate, result.metadata
```

Reuse the runtime across requests. `runtime.warmup()` can initialize sessions before interactive use. `plan_text`, `plan_records`, and `synthesize_stream` support long-form work; per-record `delivery` values propagate through planning, preflight, retries and synthesis.

## Long form and dialogue

```bash
python -m scyllasband speak --voice ariadne --file story.txt \
    --delivery energy=2.2,valence=2.3 --sampler heun --steps 8 \
    --metadata story.json -o story.wav

python -m scyllasband group-speak --file dialogue.txt -o dialogue.wav
```

A measured dialogue file can set controls per tagged span:

```text
[ariadne:en_us:energy=2.3,valence=2.5] Good to see you. [es] Me alegra verte.
[rex:en_us:tension=2.6,assertiveness=2.5] We should get going.
[ink:en_gb:whisper=on] Keep this between us.
```

Language tags retain the active delivery request. A new delivery tag replaces it, with unspecified coordinates reset to neutral. Legacy emotion tags remain available with v1 and are rejected by measured v2.

The host chunks long text and assembles waveforms. **Each generated chunk currently receives one global control vector.** Training used local physical/window measurements as auxiliary objectives, but this release does not accept start/middle/end control curves or predict a changing delivery plan. Prefix carryover and old emotion-guided reference selection are disabled for measured v2 to match its trained inference path. Multi-sentence rhythm can still sound uniform or stilted.

## Voices and languages

| Voices | English dialect |
| --- | --- |
| Ariadne, Felix, Gwen, Max, Rex, Scylla, Stone | `en_us` |
| Ink, Orpheus, Tuesday | `en_gb` |

All ten also support `es`, `it`, `fr`, `de`, and `vi`: 60 trained voice/locale pairs. `en` selects the voice's own English dialect. The runtime rejects unsupported voice/dialect combinations.

The trained Scylla's Band G2P is the production text frontend. Duration training used alignment/eSpeak phone representations. Export validation distinguishes exact-phone acoustic parity from raw-text frontend behavior; a successful graph export does not rule out pronunciation mistakes.

## Native and mobile integration

The C++ ONNX runtime supports measured v2 and legacy v1. See [native build instructions](libscyllasband/README.md).

```bash
scyllasband_native_speak --bundle scyllasband/models/v2/onnx \
    --voice ariadne --language en_us --delivery energy=2.3,valence=2.5 \
    --text "Hello again." --output native.wav
```

The stable C ABI uses the existing `affect` string slot with the explicit prefix `delivery:` for measured requests, for example `delivery:energy=2.3,whisper=off`. An empty conditioning string defaults to neutral for measured v2. It retains its original meaning for v1. Python callers should use the dedicated `delivery` field.

[Android](examples/android/README.md) and [iOS](examples/ios/README.md) integrations require an ONNX bundle for measured v2. Device-specific performance and Apple builds must be validated on their target platforms.

## Validation and limitations

The release checks include component export parity, real synthesis across all ten voices and seven locale IDs, mixed controls, whisper, Python/native phone and duration comparison, and unchanged v1 inference. Tests and audio artifacts are separate evidence: passing tests does not establish naturalness.

Known limitations include pronunciation errors, occasional cracking or metallic quality, weaker extreme anger/sadness, and limited coherent multi-sentence prosody. Scorer agreement is approximate; listening decides quality. No arbitrary-speaker cloning is provided.

See [the measured v2 model card](MODEL_CARD_V2.md) for training provenance and [the next-model plan](docs/measured_delivery_next_steps.md) for temporal conditioning work. The existing [audio gallery](https://lowkeytea.github.io/scyllasband/) contains earlier demonstrations and is not a measured-v2 validation set.

## Repository structure

- `scyllasband/`: public Python API, CLI, bundle contract and backend execution.
- `libscyllasband/`: C++ runtime and platform bridges.
- `examples/`: application integrations.
- `tests/`: public-runtime regression checks.
- `data/`, `samples/`: example text and earlier listening material.

Training and export tooling live outside this inference repository. The runtime has no trainer-package dependency.

## License and acknowledgments

Apache 2.0; see [LICENSE](LICENSE). Scylla's Band uses ONNX Runtime, Vocos (`charactr/vocos-mel-24khz`), DeepPhonemizer-derived G2P tooling, and Montreal Forced Aligner. Legacy/platform deployments also use LiteRT and Apple Core AI.
