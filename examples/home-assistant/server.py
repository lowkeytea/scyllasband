#!/usr/bin/env python3
"""Minimal Wyoming TTS server for scyllasband.

Drop-in replacement shape for the kokoro-wyoming container: HA's Wyoming
Protocol integration talks to this over TCP, with Describe/Synthesize events.
"""
import asyncio
import json
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
from importlib.metadata import version
from scyllasband import ScyllasBandRuntime, SynthesisRequest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import Synthesize

LOG = logging.getLogger("scyllasband-wyoming")
BUNDLE = Path(os.environ.get("SCYLLASBAND_MODELS", "/models/litert"))
DEFAULT_VOICE = os.environ.get("SCYLLASBAND_VOICE", "orpheus")
DEFAULT_LANG = os.environ.get("SCYLLASBAND_LANGUAGE", "en_gb")
def _delivery(name: str) -> float | None:
    raw = os.environ.get(f"SCYLLASBAND_{name.upper()}")
    return float(raw) if raw not in (None, "") else None


DELIVERY = {
    "intensity": _delivery("intensity"),
    "pace": _delivery("pace"),
    "pitch": _delivery("pitch"),
}
MAX_CHARS = int(os.environ.get("SCYLLASBAND_MAX_CHARS", "0"))
ATTR = Attribution(name="Scylla's Band", url="https://github.com/lowkeytea/scyllasband")
_LOCK = asyncio.Lock()

runtime = ScyllasBandRuntime.from_bundle(BUNDLE)
runtime.warmup()
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
VOICES = {v["id"]: v for v in MANIFEST["voices"]}
LANGUAGE_MAP = json.loads(
    (BUNDLE / "assets" / "g2p" / "language_map.json").read_text(encoding="utf-8")
)


def _info() -> Info:
    manifest = MANIFEST
    voices = [
        TtsVoice(
            name=v["id"],
            description=v.get("name") or v["id"],
            version=None,
            attribution=ATTR,
            installed=True,
            languages=[lang.lower().replace("-", "_") for lang in v["languages"]],
        )
        for v in manifest["voices"]
    ]
    return Info(tts=[TtsProgram(
        name="scyllasband",
        description="Scylla's Band local TTS",
        attribution=ATTR,
        installed=True,
        voices=voices,
        version=version("scyllasband"),
    )])


def _resolve_language(language: str, spec: dict | None) -> str:
    """Map regional codes to the voice's supported base language.

    Only English variants fall back to the voice's own English: a stale HA
    pipeline left on en_US must still work with a British voice. A non-English
    code the voice does not support is passed through so the runtime reports a
    real error instead of speaking German with English pronunciation.
    """
    code = language.strip().lower().replace("-", "_")
    mapped = LANGUAGE_MAP.get(code, code)
    supported = set(spec["languages"]) if spec else set()
    if mapped in supported or mapped == "en":
        return mapped
    base = mapped.split("_")[0]
    if base in supported:
        return base
    if mapped.startswith("en") or base == "en":
        return (spec or {}).get("default_language") or "en"
    return mapped


def _synth(text: str, voice: str, language: str) -> tuple[bytes, int]:
    spec = VOICES.get(voice)
    language = _resolve_language(language, spec)
    result = runtime.synthesize(SynthesisRequest(
        text=text, voice_id=voice, language=language, **DELIVERY
    ))
    pcm = (np.clip(np.asarray(result.audio, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    return pcm.tobytes(), int(result.sample_rate)


class Handler(AsyncEventHandler):
    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(_info().event())
            return True

        if Synthesize.is_type(event.type):
            request = Synthesize.from_event(event)
            voice = request.voice.name if request.voice and request.voice.name else DEFAULT_VOICE
            language = request.voice.language if request.voice and request.voice.language else DEFAULT_LANG
            LOG.info("Synthesize voice=%s language=%s chars=%d", voice, language, len(request.text))
            if MAX_CHARS > 0 and len(request.text) > MAX_CHARS:
                await self.write_event(Error(
                    text=f"text exceeds SCYLLASBAND_MAX_CHARS ({MAX_CHARS})",
                    code="text-too-long",
                ).event())
                return True
            try:
                async with _LOCK:
                    audio, rate = await asyncio.to_thread(_synth, request.text, voice, language)
            except Exception as exc:
                LOG.warning("synthesis failed: %s", exc)
                await self.write_event(Error(text=str(exc) or type(exc).__name__,
                                             code="synthesis-failed").event())
                return True
            await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
            await self.write_event(AudioChunk(audio=audio, rate=rate, width=2, channels=1).event())
            await self.write_event(AudioStop().event())
            return True

        return True


async def main() -> None:
    uri = os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10210")
    server = AsyncServer.from_uri(uri)
    LOG.info("Ready at %s (voice=%s %s)", uri, DEFAULT_VOICE, DEFAULT_LANG)
    task = asyncio.create_task(server.run(partial(Handler)))
    try:
        await task
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
