#!/usr/bin/env python3
"""End-to-end checks for Wyoming streaming synthesis.

  python3 test_streaming.py [host] [port]
"""

import asyncio
import sys
from wyoming.client import AsyncClient
from wyoming.tts import Synthesize, SynthesizeStart, SynthesizeChunk, SynthesizeStop, SynthesizeStopped, SynthesizeVoice
from wyoming.info import Describe, Info

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 10210
URI = f"tcp://{HOST}:{PORT}"


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    return ok


async def main():
    results = []
    async with AsyncClient.from_uri(URI) as client:
        await client.write_event(Describe().event())
        while True:
            event = await asyncio.wait_for(client.read_event(), 20)
            if Info.is_type(event.type):
                info = Info.from_event(event)
                results.append(check("Info advertises streaming",
                                     info.tts[0].supports_synthesize_streaming is True,
                                     f"got {info.tts[0].supports_synthesize_streaming}"))
                break

        await client.write_event(SynthesizeStart(
            voice=SynthesizeVoice(name="orpheus", language="en_gb")).event())

        # A complete sentence should produce audio before the stream is stopped.
        await client.write_event(SynthesizeChunk(
            text="The first short sentence is spoken now.").event())
        early_audio = False
        deadline = asyncio.get_event_loop().time() + 20
        while asyncio.get_event_loop().time() < deadline:
            event = await asyncio.wait_for(client.read_event(), 20)
            if event.type in ("audio-start", "audio-chunk"):
                early_audio = True
            if event.type == "audio-stop":
                break
            if event.type == "error":
                results.append(check("early audio (error received)", False, str(event.data)))
                break
        results.append(check("audio starts before SynthesizeStop", early_audio))

        await client.write_event(SynthesizeChunk(
            text="A second sentence arrives later in the reply.").event())
        # HA also sends the full message for backwards compatibility.
        await client.write_event(Synthesize(
            text="The first short sentence is spoken now. A second sentence arrives later in the reply.",
            voice=SynthesizeVoice(name="orpheus", language="en_gb")).event())
        await client.write_event(SynthesizeStop().event())
        stopped = False
        deadline = asyncio.get_event_loop().time() + 20
        while asyncio.get_event_loop().time() < deadline:
            event = await asyncio.wait_for(client.read_event(), 20)
            if SynthesizeStopped.is_type(event.type):
                stopped = True
                break
            if event.type == "error":
                results.append(check("stream completion (error received)", False, str(event.data)))
                break
        results.append(check("SynthesizeStopped sent", stopped))
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
