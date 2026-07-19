#!/usr/bin/env python3
"""Repo-local Scylla's Band speak entrypoint."""

from __future__ import annotations

import sys

from scyllasband.cli import speak_main


if __name__ == "__main__":
    raise SystemExit(speak_main(sys.argv[1:]))
