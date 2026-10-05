#!/usr/bin/env python3
"""Repo-local Scylla's Band groupSpeak entrypoint."""

from __future__ import annotations

import sys

from scyllasband.bootstrap import ensure_environment


if __name__ == "__main__":
    ensure_environment(["group-speak", *sys.argv[1:]], relaunch=[__file__, *sys.argv[1:]])
    from scyllasband.cli import group_main

    raise SystemExit(group_main(sys.argv[1:]))
