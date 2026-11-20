#!/usr/bin/env python3
"""Minimal Wyoming TTS server for scyllasband.

Drop-in replacement shape for the kokoro-wyoming container: HA's Wyoming
Protocol integration talks to this over TCP, with Describe/Synthesize events.
Streaming synthesis lets HA start speaking after the first passage of a long
reply instead of after all of it.
"""
import asyncio
import json
import logging
import os
import re
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scyllasband import ScyllasBandRuntime, SynthesisRequest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

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
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$")

runtime = ScyllasBandRuntime.from_bundle(BUNDLE)
runtime.warmup()
MANIFEST = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
VOICES = {v["id"]: v for v in MANIFEST["voices"]}
LANGUAGE_MAP = json.loads(
    (BUNDLE / "assets" / "g2p" / "language_map.json").read_text(encoding="utf-8")
)


def _info() -> Info:
    voices = [
        TtsVoice(
            name=v["id"],
            description=v.get("name") or v["id"],
            version=None,
            attribution=ATTR,
            installed=True,
            languages=[lang.lower().replace("-", "_") for lang in v["languages"]],
        )
        for v in MANIFEST["voices"]
    ]
    return Info(tts=[TtsProgram(
        name="scyllasband",
        description="Scylla's Band local TTS",
        attribution=ATTR,
        installed=True,
        voices=voices,
        version=version("scyllasband"),
        supports_synthesize_streaming=True,
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


def _synth_stream(text: str, voice: str, language: str) -> list[tuple[bytes, int]]:
    """Synthesize one passage, returning PCM bytes per generated audio chunk."""
    spec = VOICES.get(voice)
    resolved = _resolve_language(language, spec)
    record = dict(
        text=text, voice=voice, language=resolved,
        intensity=DELIVERY["intensity"], pace=DELIVERY["pace"], pitch=DELIVERY["pitch"],
    )
    out: list[tuple[bytes, int]] = []
    for event in runtime.synthesize_stream([record]):
        if event.type == "audio_chunk" and event.audio is not None:
            pcm = (np.clip(np.asarray(event.audio, dtype=np.float32), -1, 1) * 32767).astype("<i2")
            out.append((pcm.tobytes(), int(event.sample_rate)))
    return out


def _split_sentences(text: str) -> tuple[list[str], str]:
    """Return complete sentences and the trailing partial sentence."""
    parts = _SENTENCE_END.split(text)
    if len(parts) == 1:
        return [], text
    # The splitter removes whitespace but keeps the sentence-ending punctuation.
    complete = [part.strip() for part in parts[:-1] if part.strip()]
    return complete, parts[-1].strip()


class Handler(AsyncEventHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.stream_voice = DEFAULT_VOICE
        self.stream_language = DEFAULT_LANG
        self.stream_buffer = ""

    async def _write_audio(self, audio: bytes, rate: int) -> None:
        await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        await self.write_event(AudioChunk(audio=audio, rate=rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())

    async def _synthesize_passage(self, text: str, voice: str, language: str) -> bool:
        if not text.strip():
            return True
        if MAX_CHARS > 0 and len(text) > MAX_CHARS:
            await self.write_event(Error(
                text=f"text exceeds SCYLLASBAND_MAX_CHARS ({MAX_CHARS})",
                code="text-too-long",
            ).event())
            return False
        try:
            async with _LOCK:
                chunks = await asyncio.to_thread(_synth_stream, text, voice, language)
        except Exception as exc:
            LOG.warning("streaming synthesis failed: %s", exc)
            await self.write_event(Error(text=str(exc) or type(exc).__name__,
                                         code="synthesis-failed").event())
            return False
        for audio, rate in chunks:
            await self._write_audio(audio, rate)
        return True

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(_info().event())
            return True

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self.is_streaming = True
            self.stream_buffer = ""
            if start.voice is not None:
                self.stream_voice = start.voice.name or DEFAULT_VOICE
                self.stream_language = start.voice.language or DEFAULT_LANG
            else:
                self.stream_voice = DEFAULT_VOICE
                self.stream_language = DEFAULT_LANG
            LOG.info("stream start voice=%s language=%s", self.stream_voice, self.stream_language)
            return True

        if SynthesizeChunk.is_type(event.type):
            if not self.is_streaming:
                return True
            chunk = SynthesizeChunk.from_event(event)
            self.stream_buffer += chunk.text
            complete, self.stream_buffer = _split_sentences(self.stream_buffer)
            for sentence in complete:
                if not await self._synthesize_passage(sentence, self.stream_voice, self.stream_language):
                    self.stream_buffer = ""
                    return True
            return True

        if SynthesizeStop.is_type(event.type):
            if not self.is_streaming:
                return True
            await self._synthesize_passage(self.stream_buffer, self.stream_voice, self.stream_language)
            self.stream_buffer = ""
            self.is_streaming = False
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            # Sent for compatibility during a stream; the stream carries the text.
            if self.is_streaming:
                return True
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
            await self._write_audio(audio, rate)
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
