"""Command-line entry point for Puremacro dynamic-programming .mod files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .modfile import load_mod, solve_mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate or solve a Puremacro DP .mod file.")
    parser.add_argument("file", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate declarations without solving.")
    parser.add_argument("--method", choices=("vfi", "egm", "hjb"), help="Override the file's solver method.")
    parser.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                        help="Override one numeric calibration; may be repeated.")
    args = parser.parse_args(argv)
    params = {}
    try:
        for item in args.param:
            name, value = item.split("=", 1)
            if name in params:
                raise ValueError(f"duplicate parameter override {name!r}")
            params[name] = float(value)
        if args.check:
            if args.method:
                raise ValueError("--method applies to a solve; omit it with --check")
            model = load_mod(args.file, params=params)
            print(f"Validated declarations: {model!r}\nNo numerical solve was performed.")
        else:
            options = {"method": args.method} if args.method else {}
            result = solve_mod(args.file, params=params, **options)
            print(result.summary())
    except (ValueError, OSError, RuntimeError, TypeError, NotImplementedError) as exc:
        print(f"puremacro-dp: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
