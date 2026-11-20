#!/usr/bin/env python3
"""End-to-end checks for the Wyoming TTS container.

Run against a built image from this directory:

  docker build -t scyllasband-wyoming:test .
  docker run --rm -d --name scyllasband-test -p 10212:10210 scyllasband-wyoming:test
  python3 test_server.py 127.0.0.1 10212   # needs the wyoming package locally
"""

import asyncio
import sys
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize, SynthesizeVoice
from wyoming.info import Describe, Info
from wyoming.error import Error


HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 10210
URI = f"tcp://{HOST}:{PORT}"


async def request(text, voice="orpheus", language="en_gb", timeout=30):
    async with AsyncClient.from_uri(URI) as client:
        await asyncio.wait_for(client.write_event(Synthesize(
            text=text, voice=SynthesizeVoice(name=voice, language=language)).event()), timeout)
        events = []
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout)
            if event is None:
                return "closed", events
            events.append(event)
            if event.type in ("audio-stop", "error"):
                return event.type, events


async def info():
    async with AsyncClient.from_uri(URI) as client:
        await client.write_event(Describe().event())
        while True:
            event = await asyncio.wait_for(client.read_event(), 30)
            if Info.is_type(event.type):
                return Info.from_event(event)


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    return ok


async def main():
    results = []
    i = await info()
    program = i.tts[0]
    results.append(check("Describe returns 14 voices", len(program.voices) == 14,
                         f"got {len(program.voices)}"))
    results.append(check("version comes from package metadata", program.version == "3.0.0",
                         f"got {program.version}"))

    for name, text, voice, lang in (
        ("unknown voice error", "Hello there.", "nobody", "en_gb"),
        ("punctuation-only error", "...", "orpheus", "en_gb"),
        ("emoji-only error", "😀", "orpheus", "en_gb"),
    ):
        kind, events = await request(text, voice, lang)
        err = next((e for e in events if Error.is_type(e.type)), None)
        results.append(check(name, kind == "error" and err is not None,
                             f"got {kind}"))
        if err is not None:
            print(f"     text={err.data.get('text')!r} code={err.data.get('code')!r}")

    # de_DE must map to de, not English: ink supports de and en_gb.
    kind, events = await request("Guten Tag.", "ink", "de_DE")
    results.append(check("de_DE synth succeeds", kind == "audio-stop", f"got {kind}"))

    # A locale the voice has no base for must error, not speak with English.
    kind, events = await request("Konnichiwa.", "ink", "ja_JP")
    results.append(check("unsupported non-English errors", kind == "error", f"got {kind}"))

    # English variants still fall back to the voice's own English.
    kind, events = await request("Hello there.", "ink", "en_US")
    results.append(check("en_US falls back for British voice", kind == "audio-stop", f"got {kind}"))

    return all(results)


async def cap_check():
    """Run with SCYLLASBAND_MAX_CHARS set on the server to check the length cap."""
    kind, events = await request("x" * 200)
    err = next((e for e in events if Error.is_type(e.type)), None)
    return check("over-length text rejected", kind == "error" and
                 err is not None and err.data.get("code") == "text-too-long", f"got {kind}")


if __name__ == "__main__":
    ok = asyncio.run(main())
    if "--cap" in sys.argv:
        ok = asyncio.run(cap_check()) and ok
    sys.exit(0 if ok else 1)
