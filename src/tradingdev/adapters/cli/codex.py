"""Send one Codex CLI research message and automatically save its conversation."""

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

from filelock import Timeout

from tradingdev.adapters.clients.codex_session import record_turn
from tradingdev.adapters.storage.filesystem import WorkspacePaths


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--message")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--conversation-id")
    parser.add_argument("--model")
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args(argv)
    try:
        prompt = args.message
        if args.prompt_file is not None:
            with args.prompt_file.expanduser().open(encoding="utf-8") as stream:
                prompt = stream.read(65537)
        result = asyncio.run(
            record_turn(
                workspace=WorkspacePaths(args.workspace),
                prompt=prompt,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                conversation_id=args.conversation_id,
                timeout=args.timeout,
            )
        )
    except (ValueError, OSError, sqlite3.Error, RuntimeError, Timeout) as exc:
        result = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, ensure_ascii=False))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
