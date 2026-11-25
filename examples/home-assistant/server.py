#!/usr/bin/env python3
"""Minimal Wyoming TTS server for scyllasband.

Drop-in replacement shape for the kokoro-wyoming container: HA's Wyoming
Protocol integration talks to this over TCP, with Describe/Synthesize events.
Streaming synthesis lets HA start speaking after the first passage of a long
reply instead of after all of it.
"""
import asyncio
import hashlib
import hmac
import io
import json
import logging
import math
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


def _number(value, name: str, low: float, high: float) -> float:
    """A number from a JSON request in [low, high]; anything else is the client's error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{name} must be a number from {low:g} to {high:g}")
    return float(value)


def _delivery_override(request: dict) -> dict:
    """Per-request voice delivery from an OpenAI-compatible payload.

    Missing keys keep the env-derived DELIVERY values, so clients that send
    only model/input/voice/response_format (Firefox Read Aloud) are unchanged.
    `speed` is OpenAI's rate factor (0.25 to 4, 1 normal), read on a log scale
    as the runtime's pace: 0.5 and 2 reach its slowest and fastest (-2, 2).
    """
    out = dict(DELIVERY)
    if "pitch" in request:
        out["pitch"] = _number(request["pitch"], "pitch", -2.0, 2.0)
    if "intensity" in request:
        out["intensity"] = _number(request["intensity"], "intensity", -2.0, 2.0)
    if "speed" in request:
        speed = _number(request["speed"], "speed", 0.25, 4.0)
        if speed != 1.0:   # 1 is normal: the container's own pace stays
            out["pace"] = max(-2.0, min(2.0, 2.0 * math.log2(speed)))
    return out


MAX_CHARS = int(os.environ.get("SCYLLASBAND_MAX_CHARS", "0"))
# The OpenAI-compatible speech endpoint is off unless OPENAI_HTTP_PORT is set. With OPENAI_API_KEY set, requests need
# "Authorization: Bearer <key>". OPENAI_MAX_CHARS caps one request's text (OpenAI's own limit, 4096, by default; 0 for
# none), so one long request cannot hold the synthesis lock Home Assistant's Wyoming requests wait on for long.
HTTP_HOST = os.environ.get("OPENAI_HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("OPENAI_HTTP_PORT") or 0)
API_KEY = os.environ.get("OPENAI_API_KEY", "")
HTTP_MAX_CHARS = int(os.environ.get("OPENAI_MAX_CHARS", "4096"))
HTTP_MAX_BODY = max(65536, 16 * HTTP_MAX_CHARS) if HTTP_MAX_CHARS > 0 else 1_000_000
HTTP_TIMEOUT = 30.0   # seconds to send the request head, and again the body
HTTP_FORMATS = {"wav": "audio/wav", "pcm": "audio/pcm"}   # pcm: 16-bit little-endian mono at the bundle's rate
HTTP_MAX_QUEUE = 3    # speech requests being made or waiting for the synthesis lock; more get 503 and Retry-After
_http_queue = 0
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


def _synth(text: str, voice: str, language: str, delivery: dict | None = None) -> tuple[bytes, int]:
    spec = VOICES.get(voice)
    language = _resolve_language(language, spec)
    result = runtime.synthesize(SynthesisRequest(
        text=text, voice_id=voice, language=language, **(delivery or DELIVERY)
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
    200: "OK", 204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found",
    405: "Method Not Allowed", 408: "Request Timeout", 411: "Length Required", 413: "Payload Too Large",
    431: "Request Header Fields Too Large", 500: "Internal Server Error", 503: "Service Unavailable",
}
_MODELS_PATHS = ("/v1/models", "/models")
_SPEECH_PATHS = ("/v1/audio/speech", "/audio/speech")


class _HttpError(Exception):
    def __init__(self, status: int, message: str, headers: str = "") -> None:
        super().__init__(message)
        self.status, self.headers = status, headers


class _ClientGone(Exception):
    """The client closed its connection before its audio was made."""


def _http_head(status: int, content_type: str, length: int, extra: str = "") -> bytes:
    return (
        f"HTTP/1.1 {status} {_HTTP_REASONS[status]}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {length}\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Authorization, Content-Type\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        f"{extra}"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")


async def _http_send(writer: asyncio.StreamWriter, status: int, content_type: str, payload: bytes, extra: str = "") -> None:
    writer.write(_http_head(status, content_type, len(payload), extra))
    writer.write(payload)
    try:
        await asyncio.wait_for(writer.drain(), HTTP_TIMEOUT)
    except asyncio.TimeoutError:
        writer.transport.abort()   # a client that does not read its reply does not keep it in memory
        raise _ClientGone() from None


def _http_error_body(status: int, message: str) -> bytes:
    """OpenAI's error shape, which its clients show as error.message."""
    kind = "server_error" if status >= 500 else "invalid_request_error"
    return json.dumps({"error": {"message": message, "type": kind}}).encode("utf-8")


def _wav(pcm: bytes, rate: int) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


async def _read_request(reader: asyncio.StreamReader) -> tuple[str, str, dict, bytes]:
    """Method, path, headers (lower-case names) and body of one HTTP/1.1 request, or an _HttpError."""
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HTTP_TIMEOUT)
    except asyncio.LimitOverrunError:
        raise _HttpError(431, "request headers too large") from None
    except asyncio.TimeoutError:
        raise _HttpError(408, "request head not received in time") from None
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/"):
        raise _HttpError(400, "malformed request line")
    method, target = parts[0], parts[1]
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    if "transfer-encoding" in headers:
        raise _HttpError(411, "send the body with Content-Length")
    raw_length = headers.get("content-length", "0")
    if not (raw_length.isascii() and raw_length.isdigit()) or len(raw_length) > 12:
        raise _HttpError(400, "invalid Content-Length")
    length = int(raw_length)
    if length > HTTP_MAX_BODY:
        raise _HttpError(413, "request too large")
    try:
        body = await asyncio.wait_for(reader.readexactly(length), HTTP_TIMEOUT) if length else b""
    except asyncio.TimeoutError:
        raise _HttpError(408, "request body not received in time") from None
    return method, target.split("?", 1)[0], headers, body


def _authorize(headers: dict) -> None:
    if not API_KEY:
        return
    scheme, _, token = headers.get("authorization", "").partition(" ")
    given = hashlib.sha256(token.strip().encode("utf-8")).digest()
    if scheme.lower() != "bearer" or not hmac.compare_digest(given, hashlib.sha256(API_KEY.encode("utf-8")).digest()):
        raise _HttpError(401, "missing or wrong API key", "WWW-Authenticate: Bearer\r\n")


async def _speech(body: bytes, reader: asyncio.StreamReader) -> tuple[str, bytes]:
    """The audio for one /v1/audio/speech request: (content type, bytes)."""
    try:
        request = json.loads(body)
    except (ValueError, RecursionError):
        raise _HttpError(400, "invalid JSON body") from None
    if not isinstance(request, dict):
        raise _HttpError(400, "the body must be a JSON object")
    text = request.get("input")
    if not isinstance(text, str) or not text.strip():
        raise _HttpError(400, "input must be non-empty text")
    for limit, name in ((HTTP_MAX_CHARS, "OPENAI_MAX_CHARS"), (MAX_CHARS, "SCYLLASBAND_MAX_CHARS")):
        if limit > 0 and len(text) > limit:
            raise _HttpError(400, f"input is longer than {limit} characters ({name})")
    response_format = request.get("response_format") or "wav"
    if not isinstance(response_format, str) or response_format not in HTTP_FORMATS:
        raise _HttpError(400, "response_format must be wav or pcm")
    voice = request.get("voice") or DEFAULT_VOICE
    if not isinstance(voice, str) or voice not in VOICES:
        raise _HttpError(400, f"unknown voice: {voice}; voices are {', '.join(sorted(VOICES))}")
    language = request.get("language") or VOICES[voice].get("default_language") or DEFAULT_LANG
    if not isinstance(language, str):
        raise _HttpError(400, "language must be text")
    try:
        delivery = _delivery_override(request)
    except ValueError as exc:
        raise _HttpError(400, str(exc)) from None
    global _http_queue
    if _http_queue >= HTTP_MAX_QUEUE:
        raise _HttpError(503, "busy: try again shortly", "Retry-After: 2\r\n")
    _http_queue += 1
    try:
        async with _LOCK:
            if reader.at_eof():
                raise _ClientGone()   # its body was read, so end of stream means the client left while it waited
            pcm, rate = await asyncio.to_thread(_synth, text, voice, language, delivery)
    except ValueError as exc:   # the runtime's own refusals, such as a language the voice does not speak
        raise _HttpError(400, str(exc) or "invalid request") from None
    except _ClientGone:
        raise
    except Exception:
        LOG.exception("http synthesis failed")
        raise _HttpError(500, "synthesis failed") from None
    finally:
        _http_queue -= 1
    return HTTP_FORMATS[response_format], (_wav(pcm, rate) if response_format == "wav" else pcm)


async def _discard(reader: asyncio.StreamReader) -> None:
    """Read what is left of a refused request for a moment, so closing does not reset the connection before the client
    has read the error."""
    deadline = asyncio.get_running_loop().time() + 2.0
    received = 0
    while received < HTTP_MAX_BODY:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return
        try:
            chunk = await asyncio.wait_for(reader.read(65536), remaining)
        except (asyncio.TimeoutError, ConnectionError):
            return
        if not chunk:
            return
        received += len(chunk)


async def _handle_http(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serve a minimal OpenAI-compatible speech API over HTTP/1.1, one request per connection."""
    try:
        try:
            method, path, headers, body = await _read_request(reader)
            if method == "OPTIONS":
                await _http_send(writer, 204, "text/plain", b"")   # CORS preflight, before any key is sent
            elif path in _MODELS_PATHS:
                if method != "GET":
                    raise _HttpError(405, "use GET", "Allow: GET, OPTIONS\r\n")
                _authorize(headers)
                models = [{"id": voice, "object": "model", "owned_by": "scyllasband"} for voice in sorted(VOICES)]
                await _http_send(writer, 200, "application/json", json.dumps({"object": "list", "data": models}).encode("utf-8"))
            elif path in _SPEECH_PATHS:
                if method != "POST":
                    raise _HttpError(405, "use POST", "Allow: POST, OPTIONS\r\n")
                _authorize(headers)
                content_type, audio = await _speech(body, reader)
                await _http_send(writer, 200, content_type, audio)
            else:
                raise _HttpError(404, "not found")
        except _HttpError as exc:
            await _http_send(writer, exc.status, "application/json", _http_error_body(exc.status, str(exc)), exc.headers)
            await _discard(reader)
    except (asyncio.IncompleteReadError, ConnectionError, _ClientGone):
        pass   # the client went away
    except Exception:
        LOG.exception("http request failed")
    finally:
        writer.close()


async def main() -> None:
    uri = os.environ.get("WYOMING_URI", "tcp://0.0.0.0:10210")
    server = AsyncServer.from_uri(uri)
    tasks = [asyncio.create_task(server.run(partial(Handler)))]
    if HTTP_PORT > 0:
        http_server = await asyncio.start_server(_handle_http, HTTP_HOST, HTTP_PORT)
        tasks.append(asyncio.create_task(http_server.serve_forever()))
        if API_KEY:
            LOG.info("OpenAI-compatible speech endpoint on %s:%d (API key required)", HTTP_HOST, HTTP_PORT)
        else:
            LOG.warning("OpenAI-compatible speech endpoint on %s:%d with no API key: anyone who can reach this port "
                        "can use it (set OPENAI_API_KEY)", HTTP_HOST, HTTP_PORT)
    LOG.info("Ready at %s (voice=%s %s)", uri, DEFAULT_VOICE, DEFAULT_LANG)
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(main())
