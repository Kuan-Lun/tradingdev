"""List, read and export conversations captured by a TradingDev client adapter."""

import argparse
import json
import sqlite3
from pathlib import Path

from filelock import Timeout

from tradingdev.adapters.storage.filesystem import WorkspacePaths
from tradingdev.adapters.storage.sqlite import get_sqlite_store
from tradingdev.app.conversation_service import ConversationService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    commands = parser.add_subparsers(dest="operation", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--offset", type=int, default=0)
    listing.add_argument("--limit", type=int, default=20)
    read = commands.add_parser("read")
    read.add_argument("conversation_id")
    export = commands.add_parser("export")
    export.add_argument("conversation_id")
    export.add_argument("--expected-version", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        workspace = WorkspacePaths(args.workspace)
        service = ConversationService(workspace, get_sqlite_store(workspace))
        if args.operation == "list":
            result = service.list(args.offset, args.limit).model_dump(mode="json")
        elif args.operation == "read":
            result = service.read(args.conversation_id).model_dump(mode="json")
        else:
            result = service.export(
                args.conversation_id, args.expected_version
            ).model_dump(mode="json")
    except (ValueError, OSError, sqlite3.Error, Timeout) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1) from exc
    print(json.dumps({"success": True, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
