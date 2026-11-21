#!/usr/bin/env python3
"""Minimal Wyoming TTS server for scyllasband.

Drop-in replacement shape for the kokoro-wyoming container: HA's Wyoming
Protocol integration talks to this over TCP, with Describe/Synthesize events.
Streaming synthesis lets HA start speaking after the first passage of a long
reply instead of after all of it.
"""
import asyncio
import io
import json
import logging
import os
import re
import wave
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
HTTP_HOST = os.environ.get("OPENAI_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("OPENAI_HTTP_PORT", "8000"))
ATTR = Attribution(name="Scylla's Band", url="https://github.com/lowkeytea/scyllasband")
_LOCK = asyncio.Lock()
# Where a streamed reply may be cut: the runtime planner's sentence ends (punctuation, closing quotes, then space) and
# paragraph breaks. Each candidate is checked against the planner before it is used; see _ready_length.
_CUT = re.compile(r"[.!?…]+[\"'”’»)\]]*\s+|\n\s*\n")

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


def _record(text: str, voice: str, language: str) -> dict:
    return dict(text=text, voice=voice, language=language, **DELIVERY)


def _synth_stream(text: str, voice: str, language: str) -> list[tuple[bytes, int]]:
    """Synthesize one passage (``language`` already resolved), returning PCM bytes per generated audio chunk."""
    out: list[tuple[bytes, int]] = []
    for event in runtime.synthesize_stream([_record(text, voice, language)]):
        if event.type == "audio_chunk" and event.audio is not None:
            pcm = (np.clip(np.asarray(event.audio, dtype=np.float32), -1, 1) * 32767).astype("<i2")
            out.append((pcm.tobytes(), int(event.sample_rate)))
    return out


def _sentences(text: str, voice: str, language: str) -> list[str]:
    """The sentences the runtime would speak for ``text``, after its own text normalization."""
    return [chunk.text for chunk in runtime.plan_records([_record(text, voice, language)]).chunks]


def _ready_length(buffer: str, voice: str, language: str) -> int:
    """How much of the start of a streamed ``buffer`` can be spoken now (0 for none).

    Home Assistant sends a streamed reply in fragments as it is written, but the runtime normalizes and splits whole
    text: "Dr. Smith", "21.5" and "5 p.m. Tomorrow" are only read right with what follows them. So a cut is taken only
    where planning the two sides separately gives the same sentences as planning the whole buffer, and only with text
    after it, since the last sentence may still be growing. The latest such cut wins, so sentences that arrive together
    are spoken together.
    """
    whole = None
    for match in reversed(list(_CUT.finditer(buffer))):
        end = match.end()
        if not any(char.isalnum() for char in buffer[end:]):
            continue
        if whole is None:
            whole = _sentences(buffer, voice, language)
        if _sentences(buffer[:end], voice, language) + _sentences(buffer[end:], voice, language) == whole:
            return end
    return 0


class Handler(AsyncEventHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.stream_voice = DEFAULT_VOICE
        self.stream_language = DEFAULT_LANG
        self.stream_buffer = ""
        self.stream_chars = 0
        self.stream_failed = False

    async def _write_audio(self, audio: bytes, rate: int) -> None:
        await self.write_event(AudioStart(rate=rate, width=2, channels=1).event())
        await self.write_event(AudioChunk(audio=audio, rate=rate, width=2, channels=1).event())
        await self.write_event(AudioStop().event())

    async def _fail_stream(self, text: str, code: str) -> None:
        """Report one error for the stream and ignore the rest of its text; SynthesizeStop still ends it."""
        LOG.warning("stream failed: %s", text)
        await self.write_event(Error(text=text, code=code).event())
        self.stream_failed = True
        self.stream_buffer = ""

    async def _speak_passage(self, text: str) -> None:
        if not text.strip():
            return
        try:
            async with _LOCK:
                chunks = await asyncio.to_thread(_synth_stream, text, self.stream_voice, self.stream_language)
        except Exception as exc:
            await self._fail_stream(str(exc) or type(exc).__name__, "synthesis-failed")
            return
        for audio, rate in chunks:
            await self._write_audio(audio, rate)

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(_info().event())
            return True

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            voice = start.voice.name if start.voice and start.voice.name else DEFAULT_VOICE
            language = start.voice.language if start.voice and start.voice.language else DEFAULT_LANG
            self.is_streaming, self.stream_failed = True, False
            self.stream_buffer, self.stream_chars = "", 0
            self.stream_voice = voice
            LOG.info("stream start voice=%s language=%s", voice, language)
            try:
                # Check the voice and language once, before any text arrives.
                self.stream_language = runtime.resolve_language_for_voice(
                    voice, _resolve_language(language, VOICES.get(voice)))
            except Exception as exc:
                await self._fail_stream(str(exc) or type(exc).__name__, "synthesis-failed")
            return True

        if SynthesizeChunk.is_type(event.type):
            if not self.is_streaming or self.stream_failed:
                return True
            text = SynthesizeChunk.from_event(event).text
            self.stream_chars += len(text)
            if MAX_CHARS > 0 and self.stream_chars > MAX_CHARS:
                await self._fail_stream(f"text exceeds SCYLLASBAND_MAX_CHARS ({MAX_CHARS})", "text-too-long")
                return True
            self.stream_buffer += text
            end = _ready_length(self.stream_buffer, self.stream_voice, self.stream_language)
            if end:
                ready, self.stream_buffer = self.stream_buffer[:end], self.stream_buffer[end:]
                await self._speak_passage(ready)
            return True

        if SynthesizeStop.is_type(event.type):
            if not self.is_streaming:
                return True
            if not self.stream_failed:
                await self._speak_passage(self.stream_buffer)
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


_HTTP_REASONS = {
    200: "OK", 204: "No Content", 400: "Bad Request", 404: "Not Found",
    413: "Payload Too Large", 500: "Internal Server Error",
}


def _http_head(status: int, content_type: str, length: int) -> bytes:
    return (
        f"HTTP/1.1 {status} {_HTTP_REASONS[status]}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {length}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")


async def _http_send(writer: asyncio.StreamWriter, status: int,
                     content_type: str, payload: bytes) -> None:
    writer.write(_http_head(status, content_type, len(payload)) + payload)
    await writer.drain()


def _http_json(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _wav(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


async def _handle_http(reader: asyncio.StreamReader,
                       writer: asyncio.StreamWriter) -> None:
    """Serve a minimal OpenAI-compatible speech API over HTTP/1.1."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        lines = head.decode("latin-1").split("\r\n")
        method, target, _ = lines[0].split(" ", 2)
        headers = {}
        for line in lines[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if length < 0 or length > 1_000_000:
            await _http_send(writer, 413, "application/json",
                             _http_json({"error": "request too large"}))
            return
        body = await reader.readexactly(length) if length else b""
        path = target.split("?", 1)[0]

        if method == "OPTIONS":
            await _http_send(writer, 204, "text/plain", b"")
            return
        if method == "GET" and path in ("/v1/models", "/models"):
            models = [{"id": voice, "object": "model", "owned_by": "scyllasband"}
                      for voice in sorted(VOICES)]
            await _http_send(writer, 200, "application/json",
                             _http_json({"object": "list", "data": models}))
            return
        if method == "POST" and path in ("/v1/audio/speech", "/audio/speech"):
            try:
                request = json.loads(body)
            except Exception:
                await _http_send(writer, 400, "application/json",
                                 _http_json({"error": "invalid JSON body"}))
                return
            text = request.get("input")
            if not isinstance(text, str) or not text.strip():
                await _http_send(writer, 400, "application/json",
                                 _http_json({"error": "input must be non-empty text"}))
                return
            if MAX_CHARS > 0 and len(text) > MAX_CHARS:
                await _http_send(writer, 400, "application/json",
                                 _http_json({"error": f"text exceeds SCYLLASBAND_MAX_CHARS ({MAX_CHARS})"}))
                return
            voice = request.get("voice") or DEFAULT_VOICE
            if voice not in VOICES:
                await _http_send(writer, 400, "application/json",
                                 _http_json({"error": f"unknown voice: {voice}"}))
                return
            language = request.get("language") or VOICES[voice].get("default_language") or DEFAULT_LANG
            try:
                async with _LOCK:
                    pcm, rate = await asyncio.to_thread(_synth, text, voice, language)
            except Exception as exc:
                LOG.warning("http synthesis failed: %s", exc)
                await _http_send(writer, 400, "application/json",
                                 _http_json({"error": str(exc) or type(exc).__name__}))
                return
            await _http_send(writer, 200, "audio/wav", _wav(pcm, rate))
            return
        await _http_send(writer, 404, "application/json",
                         _http_json({"error": "not found"}))
    except Exception as exc:
        LOG.warning("http request failed: %s", exc)
    finally:
        writer.close()


async def main() -> None:
    uri = os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10210")
    server = AsyncServer.from_uri(uri)
    http_server = await asyncio.start_server(_handle_http, HTTP_HOST, HTTP_PORT)
    LOG.info("Ready at %s (voice=%s %s) and HTTP on %s:%d",
             uri, DEFAULT_VOICE, DEFAULT_LANG, HTTP_HOST, HTTP_PORT)
    tasks = [
        asyncio.create_task(server.run(partial(Handler))),
        asyncio.create_task(http_server.serve_forever()),
    ]
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
