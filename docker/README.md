# scyllasband-wyoming

Single-container Wyoming TTS server wrapping [Scylla's Band](https://github.com/lowkeytea/scyllasband).
It occupies the same slot as `kokoro-wyoming` in Home Assistant: add it as a
**Wyoming Protocol** integration and point any Assist pipeline at the resulting
`tts.*` entity. No changes to Home Assistant itself are required.

## Why a shim

Scylla's Band ships a Python API, a CLI and a C API, but no server. Home
Assistant talks to Piper and Kokoro over the [Wyoming protocol](https://github.com/rhasspy/wyoming),
so this image adds a small Wyoming `Describe`/`Synthesize` adapter around the
runtime. Everything else (voice conditioning, model files) is upstream.

## Build and run

Two ways to get the image onto the Docker host:

**Build there** (needs outbound HTTPS to GitHub and Hugging Face):

```bash
docker compose build      # downloads the ~470 MB model at build time
docker compose up -d
```

**Ship a prebuilt image** (faster when the host has no outbound access, or you
already built elsewhere). The image is self-contained, so no build context or
network is needed on the target:

```bash
# on the machine that has the image
docker save scyllasband-wyoming:local | ssh root@<docker-host> docker load
# on the Docker host
docker compose up -d        # compose uses the loaded image, no rebuild
```

Useful variants:

| Command | Purpose |
| --- | --- |
| `docker compose build --no-cache` | Force a fresh build (upstream `main` can move) |
| `docker compose up -d` | Start/recreate using the existing image |
| `docker compose down` | Stop and remove the container |
| `docker logs scyllasband-wyoming -f` | Follow startup and synthesis logs |
| `docker stats scyllasband-wyoming` | Live CPU/RAM usage |

The LiteRT model is baked into the image, so the running container needs no
network access. Host port `10211` maps to container port `10210`; `10210` is
commonly taken by an existing kokoro-wyoming container. To change the host port,
edit the `ports` entry in `docker-compose.yml`. To change delivery defaults or
the listen URI, set the environment variables below — no code change needed.
First synthesis includes a short model warm-up.

## Home Assistant setup

1. Settings → Devices & services → Add integration → **Wyoming Protocol**.
2. Host: the Docker host's IP (e.g. `192.168.3.11`), Port: `10211`.
3. A new TTS entity appears (for example `tts.scyllasband_...`).
4. Settings → Voice assistants → your pipeline → Text-to-speech: pick that
   entity, language `en_GB` (or `en`), and a voice from the list below.
5. Leave an existing Piper or Kokoro pipeline in place as a fallback.

British English voices in the current model are `ink`, `orpheus` and `tuesday`;
`orpheus` is the default.
All fourteen voices also support `de`, `es`, `fr`, `it` and `vi`; the other
eleven use `en_us`. If a pipeline requests a language a voice does not support,
the server silently uses that voice's own default language instead of failing.

## Pronunciation overrides

`phoneme_dict.json` is bind-mounted into the container and holds word → phones
entries per language. It starts with the upstream runtime's built-in word fixes
(`git`, `librispeech`, `pytest`) and `Pamela` (`pæmˈələ`, "PAM-uh-luh").
Use the helper to add or remove words; it validates phones against the model's
vocabulary, writes a `phoneme_dict.json.bak` backup first, then saves the file
in place:

```bash
docker compose exec -T scyllasband-wyoming python /app/pronunciation.py add jon --say "JON"
docker compose exec -T scyllasband-wyoming python /app/pronunciation.py add jon --phones "dʒˈɒn"
docker compose exec -T scyllasband-wyoming python /app/pronunciation.py list
docker compose exec -T scyllasband-wyoming python /app/pronunciation.py remove jon
```

`--say` runs the name through the G2P and uses the phones it produces; `--phones`
takes an exact IPA phone string. Both default to `en_gb`; pass `--language en_us`
for the American locale. Each change writes `phoneme_dict.json.bak` beside the
live file on the host (one generation of backup). Restart the container after a
change (`docker compose restart`) so the runtime reloads the file. The raw
mapping is `{language: {word: phones}}`; keys are lowercased and every phone
must exist in the bundle vocabulary. Upstream's own corrections live in
`scyllasband/lexicon.py` (`WORDS`); this file extends them, it does not replace
them.

## Configuration

Environment variables in `docker-compose.yml`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCYLLASBAND_VOICE` | `orpheus` | Default voice id when a request omits one |
| `SCYLLASBAND_LANGUAGE` | `en_gb` | Default language id |
| `SCYLLASBAND_INTENSITY` | `0` | Delivery intensity, −2 to 2 |
| `SCYLLASBAND_PACE` | `0` | Delivery pace, −2 to 2 |
| `SCYLLASBAND_PITCH` | `0` | Pitch shift in semitones, −2 to 2 |
| `WYOMING_URI` | `tcp://0.0.0.0:10210` | Wyoming listen URI |

Delivery defaults of `0` use each voice's own delivery. Intensity and pace are
measured in the voice's own range; positive intensity is more expressive.

## Verification

Check the Wyoming server advertises its voices:

```bash
docker exec scyllasband-wyoming python -c "
import asyncio
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info
async def main():
    async with AsyncClient.from_uri('tcp://127.0.0.1:10210') as c:
        await c.write_event(Describe().event())
        while True:
            e = await c.read_event()
            if Info.is_type(e.type):
                i = Info.from_event(e); print(len(i.tts[0].voices), 'voices'); break
asyncio.run(main())
"
```

Or run the upstream CLI directly inside the container to compare voices:

```bash
docker exec scyllasband-wyoming python -m scyllasband list-voices /models/litert
```

## Notes

- The Python engine is used (not the optional native C++ engine). It needs no
  compiler in the image and runs many times faster than real time on CPU.
- The image is built from upstream `main` via `pip install git+...`; pin a tag
  or commit there for a reproducible production build.
