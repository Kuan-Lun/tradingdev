"""Version-gated warning filters for tests that render native tear sheets."""

import json
import re
import subprocess
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_VERIFIED = {"quantstats": "0.0.86", "seaborn": "0.13.2", "matplotlib": "3.11.2"}
_pypi_get = partial(httpx.get, timeout=2.0, follow_redirects=False)


class PlotVersionWarning(UserWarning):
    """The native plot warning filters could not be verified."""


def _pypi_version(package: str, get: Callable[[str], httpx.Response]) -> str:
    try:
        response = get(f"https://pypi.org/pypi/{package}/json")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return f"HTTP {exc.response.status_code}"
    except (
        httpx.RequestError,
        httpx.InvalidURL,
        ImportError,
        OSError,
        ValueError,
    ) as exc:
        return type(exc).__name__

    try:
        info = response.json()["info"]
        latest = info["version"]
        if (
            str(info["name"]).lower() != package
            or not isinstance(latest, str)
            or not latest.strip()
        ):
            raise ValueError("invalid project metadata")
    except KeyError, TypeError, ValueError, RecursionError:
        return "invalid project metadata"
    return latest


def run_plot_version_check(*, timeout: float = 8) -> bool:
    try:
        child = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve())],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            encoding="utf-8",
            timeout=timeout,
            check=True,
        )
        latest = json.loads(child.stdout) if not child.stderr else None
        if (
            not isinstance(latest, dict)
            or latest.keys() != _VERIFIED.keys()
            or not all(isinstance(v, str) and v.strip() for v in latest.values())
        ):
            raise ValueError("invalid PyPI version map")
    except (OSError, subprocess.SubprocessError) as exc:
        result = type(exc).__name__
    except ValueError, RecursionError:
        result = "unexpected version check output"
    else:
        differences = []
        for package, expected in _VERIFIED.items():
            try:
                installed = version(package)
            except PackageNotFoundError:
                installed = "not installed"
            if installed != expected or latest[package] != expected:
                differences.append(
                    f"{package}: installed={installed}, "
                    f"PyPI={latest[package]}, verified={expected}"
                )
        if not differences:
            return True
        result = "; ".join(differences)
    with warnings.catch_warnings():
        warnings.simplefilter("always", PlotVersionWarning)
        warnings.warn(
            f"Keeping native QuantStats plot warnings ({result}). "
            "If PyPI is unreachable, check connectivity, TLS certificates and "
            "proxy settings (SOCKS requires socksio). Check pyproject.toml and "
            "uv.lock, then reverify the native report.",
            PlotVersionWarning,
            stacklevel=2,
        )
    return False


@contextmanager
def native_plot_warnings(verified: bool) -> Iterator[None]:
    if not verified:
        yield
        return

    from matplotlib import MatplotlibDeprecationWarning

    with warnings.catch_warnings():
        for module, category, message in (
            (
                "seaborn.matrix",
                PendingDeprecationWarning,
                "The set_bad function will be deprecated in a future version. "
                "Use cmap.with_extremes(bad=...) or Colormap(bad=...) instead.",
            ),
            (
                "seaborn.categorical",
                MatplotlibDeprecationWarning,
                "vert: bool was deprecated in Matplotlib 3.11 and will be removed "
                "in 3.13. Use orientation: {'vertical', 'horizontal'} instead.",
            ),
        ):
            warnings.filterwarnings(
                "ignore",
                message=rf"(?-i:{re.escape(message)})\Z",
                category=category,
                module=rf"{re.escape(module)}\Z",
            )
        yield


if __name__ == "__main__":
    print(json.dumps({name: _pypi_version(name, _pypi_get) for name in _VERIFIED}))
