#!/usr/bin/env python3
"""Checks for the OpenAI-compatible HTTP endpoint in the same container.

  python3 test_openai.py [host] [port] [api-key]
"""

import http.client
import io
import json
import struct
import sys
import wave

HOST = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
KEY = sys.argv[3] if len(sys.argv) > 3 else ""


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}".rstrip())
    return ok


def request(method, path, body=None, headers=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    hdrs = dict(headers or {})
    if KEY:
        hdrs["Authorization"] = f"Bearer {KEY}"
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

    status, headers, payload = request("OPTIONS", "/v1/audio/speech")
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

    status, _, payload = request("POST", "/v1/audio/speech", {
        "input": "   ", "voice": "orpheus", "response_format": "wav",
    })
    results.append(check("empty input rejected", status == 400, f"status={status}"))
    return all(results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
