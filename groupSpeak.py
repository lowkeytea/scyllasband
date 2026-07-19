#!/usr/bin/env python3
"""Repo-local Scylla's Band groupSpeak entrypoint."""

from __future__ import annotations

import sys

from scyllasband.cli import group_main


if __name__ == "__main__":
    raise SystemExit(group_main(sys.argv[1:]))
