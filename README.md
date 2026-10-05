# Scylla's Band

Scylla's Band is a local text-to-speech runtime for ten managed voices. It predicts phone durations, generates continuous acoustic latents with rectified flow, and decodes speech at 24 kHz through an acoustic adapter and Vocos.

Delivery is controlled with **energy, tension, valence and assertiveness**, each from **0 to 4** with **2 as neutral**, plus **whisper on/off**.

## Quick start

From this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

python -m scyllasband download --yes
python -m scyllasband list-voices
python -m scyllasband speak --voice scylla --language en_us \
    --delivery energy=2,tension=2,valence=2,assertiveness=2,whisper=off \
    -o hello.wav "Hello from Scylla's Band."
```

The model is downloaded from [`spybyscript/scyllasband`](https://huggingface.co/spybyscript/scyllasband), pinned to release `v2-20261005`, into `scyllasband/models/litert`. Voice conditioning is embedded in the graphs; no separate voice files are needed.

## Delivery controls

| Control | Request | Default |
| --- | --- | --- |
| Energy | 0–4 or `auto` | 2 |
| Tension | 0–4 or `auto` | 2 |
| Valence | 0–4 or `auto` | 2 |
| Assertiveness | 0–4 or `auto` | 2 |
| Whisper | `on`, `off`, or `auto` | `off` |

With every control at 2, the default, voices speak in a calm, conversational delivery with a neutral tone. Moving a control away from 2 changes the delivery:

| Control | Toward 0 | Toward 4 |
| --- | --- | --- |
| Energy | Lower, darker voice with more pauses between phrases | Higher, brighter, more projected voice |
| Tension | Slightly looser and steadier | Tighter, with wider pitch swings |
| Valence | Flatter pitch, a little slower, longer pauses | Livelier pitch, a little quicker, shorter pauses |
| Assertiveness | Hesitant, with longer pauses between phrases | Faster and clipped with few pauses; the strongest control over pace |

`whisper=on` whispers: quieter and breathy.

Combining the controls gives familiar deliveries:

| Delivery | Settings |
| --- | --- |
| Calm and soft | `energy=1.2,tension=0.8,valence=2.6` |
| Assertive and rapid-fire | `energy=2.6,tension=2.2,assertiveness=3.6` |
| Joyful | `energy=3.2,valence=3.6` |
| Angry | `energy=3.4,tension=3.4,valence=0.8,assertiveness=3.4` |
| Sad | `energy=1,tension=1.6,valence=0.8,assertiveness=1.6` |
| Whispered | `whisper=on` |

The [voice gallery](https://lowkeytea.github.io/scyllasband/#delivery) plays each of these for all ten voices with the exact settings used. How far a control moves the delivery varies with voice, language and text, so start near 2 and listen as you move toward the extremes.

```bash
python -m scyllasband speak --voice ariadne \
    --delivery energy=3.2,valence=3.6 \
    -o joyful.wav "We actually did it! The whole street came out to watch!"

python -m scyllasband speak --voice ink --language en_gb \
    --delivery whisper=on -o whisper.wav "Keep your voice down."
```

Unspecified controls use the neutral default of 2 (whisper off). `--delivery auto` leaves every control unspecified, and `energy=auto` leaves only that one unspecified; an unspecified control is passed to the model as absent, which is different from requesting 2.

## Backends

| Bundle | Download | Runtime package |
| --- | --- | --- |
| LiteRT (default) | `python -m scyllasband download --yes` | `ai-edge-litert` (installed with this package) |
| ONNX Runtime | `python -m scyllasband download --flavor onnx --yes` | `pip install onnxruntime` |

Both bundles hold INT8 transformer weights and the G2P model in a matching precision; the LiteRT bundle also quantizes the vocoder, which makes it the smaller and faster download. The backend follows the bundle; `--backend` and `--bundle` select another installed bundle. `--threads` sets the CPU threads per graph. Sampling uses Heun with 8 steps by default (`--sampler`, `--steps`).

## Python API

```python
from scyllasband import ScyllasBandRuntime, SynthesisRequest

runtime = ScyllasBandRuntime.from_bundle("scyllasband/models/litert")
result = runtime.synthesize(SynthesisRequest(
    text="It is a real pleasure to meet you.",
    voice_id="ariadne",
    language="en_us",
    delivery={"energy": 2.3, "tension": 1.7, "valence": 2.6,
              "assertiveness": 2.0, "whisper": "off"},
    seed=2027,
))
# result.audio (float32 numpy array), result.sample_rate, result.metadata
```

Reuse the runtime across requests; `runtime.warmup()` creates the sessions before interactive use. `runtime.synthesize_stream(records)` yields one audio chunk per sentence as it is ready, and `runtime.plan_records(records)` returns the sentence plan.

## Long form and dialogue

Text is spoken one sentence at a time. Each sentence sees its neighbours as context and continues from the acoustics of the sentence before it, and consecutive sentences are decoded as one continuous waveform, so long text needs no chunking settings.

```bash
python -m scyllasband speak --voice ariadne --file story.txt \
    --delivery energy=2.2,valence=2.3 --metadata story.json -o story.wav

python -m scyllasband stream --voice scylla --file story.txt -o chunks/

python -m scyllasband group-speak --file dialogue.txt -o dialogue.wav
```

A dialogue file sets the voice, language and controls per tagged span:

```text
[ariadne:en_us:energy=2.3,valence=2.5] Good to see you. [es] Me alegra verte.
[rex:en_us:tension=2.6,assertiveness=2.5] We should get going.
[ink:en_gb:whisper=on] Keep this between us.
```

Language tags keep the active delivery request. A new delivery tag replaces it, with unspecified coordinates reset to neutral. `--pause-ms` adds silence where the voice, language or delivery changes.

## Voices and languages

| Voices | English dialect |
| --- | --- |
| Ariadne, Felix, Gwen, Max, Rex, Scylla, Stone | `en_us` |
| Ink, Orpheus, Tuesday | `en_gb` |

All ten also support `es`, `it`, `fr`, `de`, and `vi`: 60 trained voice/locale pairs. `en` selects the voice's own English dialect. The runtime rejects unsupported voice/dialect combinations.

Raw text is normalized to its spoken form (numbers, dates, currency, abbreviations) and converted to phones by the trained Scylla's Band G2P shipped in the bundle. `--no-normalize-text` skips normalization.

## Native and mobile integration

`libscyllasband/` is the C++ runtime for the same bundles, built against LiteRT or ONNX Runtime; see [its README](libscyllasband/README.md). [Android](examples/android/README.md) and [iOS](examples/ios/README.md) examples use it.

## Validation and limitations

Release checks include export parity against PyTorch, end-to-end comparison of full passages with the PyTorch reference path, comparison of the quantized bundles with full precision on real sentences, and listening across all ten voices. Passing tests does not establish naturalness; listening decides quality.

Known limitations include occasional pronunciation errors, occasional roughness on fast, wide pitch movements in some voices, and limited strong anger or sadness. No arbitrary-speaker cloning is provided.

See [the model card](MODEL_CARD.md) for the model contract.

## Repository structure

- `scyllasband/`: Python API, CLI, bundle contract and execution.
- `libscyllasband/`: C++ runtime and platform bridges.
- `examples/`: application integrations.
- `tests/`: runtime regression tests (`python -m pytest tests`).

Training and export tooling live outside this inference repository.

## License and acknowledgments

Apache 2.0; see [LICENSE](LICENSE). Scylla's Band uses LiteRT, ONNX Runtime, Vocos (`charactr/vocos-mel-24khz`), DeepPhonemizer-derived G2P tooling, and Montreal Forced Aligner.
