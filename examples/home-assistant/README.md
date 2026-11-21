# scyllasband-wyoming

Single-container Wyoming TTS server wrapping [Scylla's Band](https://github.com/lowkeytea/scyllasband).
It occupies the same slot as `kokoro-wyoming` in Home Assistant: add it as a
**Wyoming Protocol** integration and point any Assist pipeline at the resulting
`tts.*` entity. No changes to Home Assistant itself are required.

The server advertises Wyoming streaming synthesis. Home Assistant can start
playing audio after the first complete sentence of a long reply instead of
waiting for the whole passage. Streamed text is cut only where the runtime's
own text normalization and sentence splitting would cut it, so decimals
("21.5") and abbreviations ("Dr. Smith", "5 p.m.") read the same as in a
single request. A sentence is spoken once the next one begins, or when the
reply ends.

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
| `docker compose build --no-cache` | Force a fresh build (override the pinned commit with `--build-arg SCYLLASBAND_COMMIT=...`) |
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

## Security

Wyoming has no authentication. Docker published ports bypass host firewalls
such as ufw, so keep this port on the local network and do not forward it to
the internet. Bind it to one host address when possible:

```yaml
ports:
  - "192.168.1.10:10211:10210"
```

## Home Assistant setup

1. Settings → Devices & services → Add integration → **Wyoming Protocol**.
2. Host: the Docker host's IP (e.g. `<docker-host-ip>`), Port: `10211`.
3. A new TTS entity appears (for example `tts.scyllasband_...`).
4. Settings → Voice assistants → your pipeline → Text-to-speech: pick that
   entity, language `en_GB` (or `en`), and a voice from the list below.
5. Leave an existing Piper or Kokoro pipeline in place as a fallback.

British English voices in the current model are `ink`, `orpheus` and `tuesday`;
`orpheus` is the default.
All fourteen voices also support `de`, `es`, `fr`, `it` and `vi`; the other
eleven use `en_us`. Regional codes map to the voice's base locale (`de_DE` to
`de`). Only English variants fall back to the voice's own English; an
unsupported non-English locale returns an error to Home Assistant.

## Pronunciation overrides

The word list lives in a named volume at `/data/phoneme_dict.json`. It starts
empty and survives image rebuilds. The runtime reads it through a symlink at
the bundle path, and the helper runs as the container user, so no host file
permissions need changing.

Use the helper to add or remove words; it validates phones against the model's
vocabulary, writes a `phoneme_dict.json.bak` backup first, then saves the file
in place:

```bash
docker exec scyllasband-wyoming python /app/pronunciation.py add jon --phones "dʒˈɒn"
docker exec scyllasband-wyoming python /app/pronunciation.py list
docker exec scyllasband-wyoming python /app/pronunciation.py remove jon
```

`--phones` takes an exact IPA phone string. `--say` sends a respelling through
the G2P and prints the phones it produced; treat that as a starting point,
because the G2P stresses each word of a respelling separately and can get it
wrong. Check the printed phones and re-add with `--phones` when needed.

Both default to `en_gb`; pass `--language en_us` for the American locale, or
`--language all` to write one entry to every locale in the file. The dict
format is per-locale. Upstream's own corrections live in `scyllasband/lexicon.py`
(`WORDS`) and G2P already reads names such as Pamela correctly, so only add
words it gets wrong for your voice and text.

## Configuration

Environment variables in `docker-compose.yml`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCYLLASBAND_VOICE` | `orpheus` | Default voice id when a request omits one |
| `SCYLLASBAND_LANGUAGE` | `en_gb` | Default language id |
| `SCYLLASBAND_INTENSITY` | unset (voice default) | Delivery intensity, −2 to 2 |
| `SCYLLASBAND_PACE` | unset (voice default) | Delivery pace, −2 to 2 |
| `SCYLLASBAND_PITCH` | unset (voice default) | Pitch shift in semitones, −2 to 2 |
| `SCYLLASBAND_MAX_CHARS` | `0` (no cap) | Reject requests longer than this many characters |
| `WYOMING_URI` | `tcp://0.0.0.0:10210` | Wyoming listen URI |

The shipped compose file comments out the three delivery variables, so each
voice's own defaults apply. To override a default, uncomment that line in
`docker-compose.yml` and set the value, then recreate the container. Intensity
and pace are measured in the voice's own range; positive intensity is more
expressive.

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
- The image installs scyllasband from the pinned commit in the Dockerfile's
  `SCYLLASBAND_COMMIT` build ARG and wyoming from the `WYOMING_SPEC` range;
  override either build arg to change them.
