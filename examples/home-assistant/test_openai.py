#!/usr/bin/env python3
"""Checks for the OpenAI-compatible HTTP endpoint in the same container.

  python3 test_openai.py [host] [port] [api-key]
"""

import http.client
import io
import json
import socket
import sys
import wave

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
KEY = sys.argv[3] if len(sys.argv) > 3 else ""


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    return ok


def request(method, path, body=None, headers=None, key=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    hdrs = dict(headers or {})
    key = KEY if key is None else key
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    data = json.dumps(body).encode() if body is not None else None
    if data is not None:
        hdrs["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=hdrs)
    resp = conn.getresponse()
    payload = resp.read()
    result = (resp.status, dict(resp.getheaders()), payload)
    conn.close()
    return result


def main():
    results = []
    status, headers, payload = request("GET", "/v1/models")
    ok = status == 200
    models = {}
    if ok:
        try:
            models = json.loads(payload)
        except Exception:
            ok = False
    ids = [m.get("id") for m in models.get("data", [])]
    results.append(check("GET /v1/models", ok and "orpheus" in ids,
                         f"status={status} voices={len(ids)}"))
    results.append(check("CORS on GET", headers.get("Access-Control-Allow-Origin") == "*"))

    status, headers, payload = request("OPTIONS", "/v1/audio/speech", key="")   # a preflight carries no key
    results.append(check("CORS preflight", status in (200, 204) and
                         "POST" in headers.get("Access-Control-Allow-Methods", ""),
                         f"status={status}"))

    status, headers, payload = request("POST", "/v1/audio/speech", {
        "model": "scyllasband", "input": "Hello from Scyllas Band.",
        "voice": "orpheus", "response_format": "wav",
    })
    ok = status == 200 and payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"
    frames = 0
    rate = 0
    if ok:
        with wave.open(io.BytesIO(payload)) as w:
            frames = w.getnframes()
            rate = w.getframerate()
    results.append(check("POST /v1/audio/speech returns WAV", ok and frames > 0,
                         f"status={status} frames={frames} rate={rate}"))
    results.append(check("CORS on POST", headers.get("Access-Control-Allow-Origin") == "*"))

    # Per-request delivery (pitch/speed/intensity). Firefox Read Aloud sends
    # none of these; its baseline request above must keep working unchanged.
    base = request("POST", "/v1/audio/speech", {
        "input": "Delivery check.", "voice": "orpheus", "response_format": "wav",
    })
    results.append(check("baseline (no delivery keys)", base[0] == 200 and base[2][:4] == b"RIFF",
                         f"status={base[0]}"))

    pitched = request("POST", "/v1/audio/speech", {
        "input": "Delivery check.", "voice": "orpheus", "response_format": "wav", "pitch": 2,
    })
    results.append(check("pitch accepted", pitched[0] == 200 and pitched[2][:4] == b"RIFF",
                         f"status={pitched[0]}"))

    fast = request("POST", "/v1/audio/speech", {
        "input": "Delivery check.", "voice": "orpheus", "response_format": "wav", "speed": 1.5,
    })
    ok = fast[0] == 200 and fast[2][:4] == b"RIFF"
    fast_frames = base_frames = 0
    if ok and base[0] == 200:
        with wave.open(io.BytesIO(fast[2])) as w:
            fast_frames = w.getnframes()
        with wave.open(io.BytesIO(base[2])) as w:
            base_frames = w.getnframes()
    results.append(check("speed changes pacing", ok and fast_frames != base_frames,
                         f"status={fast[0]} base={base_frames} fast={fast_frames}"))

    for bad, label in [(5, "pitch out of range"), ("loud", "pitch not numeric")]:
        status, _, _ = request("POST", "/v1/audio/speech", {
            "input": "Delivery check.", "voice": "orpheus", "response_format": "wav", "pitch": bad,
        })
        results.append(check(label + " rejected", status == 400, f"status={status}"))

    status, _, payload = request("POST", "/v1/audio/speech", {
        "input": "   ", "voice": "orpheus", "response_format": "wav",
    })
    results.append(check("empty input rejected", status == 400, f"status={status}"))
    message = json.loads(payload or b"{}").get("error", {})
    results.append(check("errors use OpenAI's shape", isinstance(message, dict) and bool(message.get("message")),
                         f"error={message}"))

    # Speed is OpenAI's 0.25..4 factor; values beyond the runtime's pace range stay at its ends.
    status, _, _ = request("POST", "/v1/audio/speech", {"input": "Delivery check.", "voice": "orpheus", "speed": 4.0})
    results.append(check("speed 4 accepted", status == 200, f"status={status}"))
    for bad in (5, "fast", True):
        status, _, _ = request("POST", "/v1/audio/speech", {"input": "Delivery check.", "voice": "orpheus", "speed": bad})
        results.append(check(f"speed {bad!r} rejected", status == 400, f"status={status}"))

    # Formats: wav or raw pcm; no encoder for the others.
    status, headers, payload = request("POST", "/v1/audio/speech", {"input": "Delivery check.", "voice": "orpheus",
                                                                      "response_format": "pcm"})
    results.append(check("pcm returns raw samples", status == 200 and headers.get("Content-Type") == "audio/pcm"
                         and len(payload) > 0 and len(payload) % 2 == 0 and payload[:4] != b"RIFF",
                         f"status={status} bytes={len(payload)}"))
    status, _, _ = request("POST", "/v1/audio/speech", {"input": "Hi.", "voice": "orpheus", "response_format": "mp3"})
    results.append(check("mp3 rejected", status == 400, f"status={status}"))

    # Requests that never reach synthesis.
    status, _, _ = request("POST", "/v1/audio/speech", {"input": "a" * 4097, "voice": "orpheus"})   # default OPENAI_MAX_CHARS
    results.append(check("input over 4096 characters rejected", status == 400, f"status={status}"))
    status, _, _ = request("POST", "/v1/audio/speech", {"input": "Hi.", "voice": "alloy"})
    results.append(check("unknown voice rejected", status == 400, f"status={status}"))
    status, _, _ = request("GET", "/v1/audio/speech")
    results.append(check("GET on the speech path is 405", status == 405, f"status={status}"))
    results.append(check("JSON array body rejected", raw_status(b"[]") == 400))
    results.append(check("malformed request line rejected", raw_status(None, b"GARBAGE\r\n\r\n") == 400))
    status, _, _ = request("POST", "/v1/audio/speech", {"input": "Hi.", "voice": "orpheus", "response_format": ["mp3"]})
    results.append(check("non-text response_format rejected", status == 400, f"status={status}"))
    results.append(check("deeply nested JSON rejected", raw_status(b"[" * 5000) == 400))
    results.append(check("non-ASCII Content-Length rejected",
                         raw_status(None, "POST /v1/audio/speech HTTP/1.1\r\nContent-Length: ²\r\n\r\n".encode("latin-1")) == 400))
    if KEY:
        status, _, _ = request("GET", "/v1/models", headers={"Authorization": f"bearer {KEY}"}, key="")
        results.append(check("lower-case bearer accepted", status == 200, f"status={status}"))
        status, _, _ = request("GET", "/v1/models", key="")
        results.append(check("no key rejected", status == 401, f"status={status}"))
        status, _, _ = request("POST", "/v1/audio/speech", {"input": "Hi.", "voice": "orpheus"}, key=KEY + "x")
        results.append(check("wrong key rejected", status == 401, f"status={status}"))
    return all(results)


def raw_status(body, request_bytes=None):
    """The status of a hand-written request (a body http.client would not send as JSON, or a broken request line)."""
    if request_bytes is None:
        auth = f"Authorization: Bearer {KEY}\r\n".encode() if KEY else b""
        request_bytes = (b"POST /v1/audio/speech HTTP/1.1\r\nHost: x\r\n" + auth +
                         b"Content-Length: %d\r\n\r\n" % len(body) + body)
    with socket.create_connection((HOST, PORT), timeout=30) as sock:
        sock.sendall(request_bytes)
        reply = sock.recv(64)
    try:
        return int(reply.split(b" ")[1])
    except (IndexError, ValueError):
        return 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
