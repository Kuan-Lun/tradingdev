"""Bounded regular-file access for workspace reading artifacts."""

import os
import stat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class WorkspaceFiles:
    def __init__(self, root: Path) -> None:
        self.root = root

    def safe_path(self, path: Path) -> Path:
        if self.root.is_symlink() or self.root.resolve() != self.root:
            raise ValueError("Artifact workspace is not a canonical directory")
        relative = path.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            if part in {".", ".."} or current.is_symlink():
                raise ValueError("Artifact paths must not traverse symlinks or parents")
        return current

    def read(self, path: Path, limit: int) -> bytes:
        descriptor = os.open(
            self.safe_path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("Artifact source is not a bounded regular file")
            content = stream.read(limit + 1)
        if len(content) > limit:
            raise ValueError("Artifact exceeds its byte limit")
        return content
