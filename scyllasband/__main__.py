import sys

from scyllasband.bootstrap import ensure_environment

if __name__ == "__main__":
    ensure_environment(sys.argv[1:])
    from scyllasband.cli import main

    raise SystemExit(main())
