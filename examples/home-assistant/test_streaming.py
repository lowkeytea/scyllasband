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

        # A complete sentence should produce audio before the stream is stopped, once the next one begins (until
        # then the server cannot tell whether it has ended: "21." may continue as "21.5").
        await client.write_event(SynthesizeChunk(
            text="The first short sentence is spoken now.").event())
        await client.write_event(SynthesizeChunk(text=" A second").event())
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
            text=" sentence arrives later in the reply.").event())
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

    # Home Assistant forwards a reply as it is written, so a fragment can end anywhere. Text the runtime reads as one
    # sentence must stay one passage.
    for name, chunks in (
        ("decimal split across chunks stays whole", ["The temperature is 21", ".", "5 degrees today."]),
        ("abbreviations stay whole", words("Dr. Smith will call at 5 p.m. today.")),
    ):
        passages, errors, stopped = await stream(chunks)
        results.append(check(name, passages == 1 and not errors and stopped,
                             f"got {passages} passages, errors={errors}"))
    passages, errors, stopped = await stream(words("Your meeting is at 3 p.m. Tomorrow is free."))
    results.append(check("sentence after an abbreviation is its own passage", passages == 2 and stopped,
                         f"got {passages} passages"))
    passages, errors, stopped = await stream(words("One. Two. Three."), voice="nobody")
    results.append(check("unknown voice: one error, stream still stopped", len(errors) == 1 and stopped,
                         f"got {len(errors)} errors, stopped={stopped}"))
    return all(results)


def words(text):
    """Split text into the word-sized fragments a streamed reply arrives in."""
    parts = text.split(" ")
    return parts[:1] + [" " + part for part in parts[1:]]


async def stream(chunks, voice="orpheus", language="en_gb"):
    """Stream ``chunks``; return (audio passages, error texts, whether SynthesizeStopped arrived)."""
    passages, errors = 0, []
    async with AsyncClient.from_uri(URI) as client:
        await client.write_event(SynthesizeStart(voice=SynthesizeVoice(name=voice, language=language)).event())
        for chunk in chunks:
            await client.write_event(SynthesizeChunk(text=chunk).event())
        await client.write_event(SynthesizeStop().event())
        while True:
            event = await asyncio.wait_for(client.read_event(), 60)
            if event is None:
                return passages, errors, False
            if event.type == "audio-start":
                passages += 1
            elif event.type == "error":
                errors.append(event.data.get("text"))
            elif SynthesizeStopped.is_type(event.type):
                return passages, errors, True


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(main()) else 1)
