"""Offline regressions for the native plotting tests' warning policy."""

import json
import os
import runpy
import subprocess
import warnings
from contextlib import redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING
from unittest.mock import patch

import httpx
import pytest
from matplotlib import MatplotlibDeprecationWarning

from tests import quantstats_warnings

if TYPE_CHECKING:
    from pathlib import Path

_SET_BAD = (
    "The set_bad function will be deprecated in a future version. "
    "Use cmap.with_extremes(bad=...) or Colormap(bad=...) instead."
)
_VERT = (
    "vert: bool was deprecated in Matplotlib 3.11 and will be removed in 3.13. "
    "Use orientation: {'vertical', 'horizontal'} instead."
)


def _emit(message: str, category: type[Warning], module: str) -> None:
    warnings.warn_explicit(message, category, "plot.py", 1, module=module)


@pytest.mark.parametrize(
    ("installed", "upstream", "pypi_difference"),
    [
        ("0.13.2", "0.13.2", None),
        ("0.13.1", "0.13.2", "0.13.2"),
        ("0.13.2", "0.13.3", "0.13.3"),
        ("0.13.3", "0.13.3", "0.13.3"),
        ("0.13.2", httpx.ConnectTimeout("offline"), "ConnectTimeout"),
        ("0.13.2", ImportError("missing socksio"), "ImportError"),
        ("0.13.2", httpx.Response(503), "HTTP 503"),
        ("0.13.2", httpx.Response(200, json={}), "invalid project metadata"),
    ],
)
def test_only_verified_versions_suppress_native_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    installed: str,
    upstream: str | httpx.Response | Exception,
    pypi_difference: str | None,
) -> None:
    local = {"quantstats": "0.0.86", "matplotlib": "3.11.2", "seaborn": installed}
    for name, release in local.items():
        dist = tmp_path / f"{name}-{release}.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(f"Name: {name}\nVersion: {release}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    pending = {f"https://pypi.org/pypi/{name}/json": name for name in local}

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        name = pending.pop(str(request.url))
        if name == "seaborn":
            if isinstance(upstream, Exception):
                raise upstream
            if isinstance(upstream, httpx.Response):
                return upstream
            latest = upstream
        else:
            latest = local[name]
        return httpx.Response(200, json={"info": {"name": name, "version": latest}})

    with (
        warnings.catch_warnings(record=True) as caught,
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
    ):
        warnings.simplefilter("always")
        stdout = StringIO()
        with patch.object(httpx, "get", client.get), redirect_stdout(stdout):
            runpy.run_path(quantstats_warnings.__file__, run_name="__main__")
        child = subprocess.CompletedProcess([], 0, stdout.getvalue(), "")
        with patch("subprocess.run", return_value=child):
            verified = quantstats_warnings.run_plot_version_check()
        with quantstats_warnings.native_plot_warnings(verified):
            _emit(_SET_BAD, PendingDeprecationWarning, "seaborn.matrix")
            _emit(_VERT, MatplotlibDeprecationWarning, "seaborn.categorical")

    assert not pending
    if pypi_difference is None:
        assert not caught
    else:
        summary, *plot_warnings = [str(item.message) for item in caught]
        assert plot_warnings == [_SET_BAD, _VERT]
        detail = f"installed={installed}, PyPI={pypi_difference}, verified=0.13.2"
        assert f"seaborn: {detail}" in summary
        assert "pyproject.toml" in summary and "uv.lock" in summary
        if isinstance(upstream, ImportError):
            assert "proxy settings (SOCKS requires socksio)" in summary


def test_pypi_get_uses_environment_proxy() -> None:
    with (
        patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.test:8080"}, clear=True),
        patch("socket.getaddrinfo", side_effect=OSError("offline")) as resolve,
        pytest.raises(httpx.ConnectError),
    ):
        quantstats_warnings._pypi_get("https://pypi.org/pypi/seaborn/json")
    assert resolve.call_args.args[:2] == ("proxy.test", 8080)


@pytest.mark.parametrize(
    ("script", "timeout", "reason"),
    [
        ("import time; time.sleep(2)", 0.1, "TimeoutExpired"),
        ("pass", 8, "unexpected version check output"),
        ('print(\'{"seaborn": "0.13.2"}\')', 8, "unexpected version check output"),
        (
            'print(\'{"quantstats":"0.0.86","seaborn":null,"matplotlib":"3.11.2"}\')',
            8,
            "unexpected version check output",
        ),
        (
            'import sys; print("private proxy detail", file=sys.stderr); '
            f"print({json.dumps(quantstats_warnings._VERIFIED)!r})",
            8,
            "unexpected version check output",
        ),
        ("import sys; sys.exit('private proxy detail')", 8, "CalledProcessError"),
    ],
)
def test_failed_version_probe_keeps_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    timeout: float,
    reason: str,
) -> None:
    child = tmp_path / "probe.py"
    child.write_text(script, encoding="utf-8")
    monkeypatch.setattr(quantstats_warnings, "__file__", str(child))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", UserWarning)
        assert quantstats_warnings.run_plot_version_check(timeout=timeout) is False
        with pytest.raises(UserWarning, match="ordinary warning"):
            _emit("ordinary warning", UserWarning, "seaborn.categorical")
    (advisory,) = caught
    assert advisory.category is quantstats_warnings.PlotVersionWarning
    message = str(advisory.message)
    assert reason in message and "reverify" in message
    assert "private proxy detail" not in message


def test_request_programming_errors_propagate() -> None:
    def broken_get(url: str) -> httpx.Response:
        raise TypeError("incorrect request arguments")

    with pytest.raises(TypeError, match="incorrect request arguments"):
        quantstats_warnings._pypi_version("seaborn", broken_get)


def test_session_probe_runs_only_on_first_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    consumer = tmp_path / "test_plot_consumer.py"
    consumer.write_text(
        "import pytest\n"
        "from tests.conftest import verified_quantstats_versions\n\n"
        "@pytest.mark.parametrize('render', [1, 2])\n"
        "def test_render(render, verified_quantstats_versions):\n"
        "    assert verified_quantstats_versions() is False\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    args = [
        "--disable-plugin-autoload",
        "--import-mode=importlib",
        "-p",
        "no:cacheprovider",
        str(consumer),
    ]
    with patch.object(
        quantstats_warnings, "run_plot_version_check", return_value=False
    ) as probe:
        assert pytest.main([*args, "--setup-only"]) == pytest.ExitCode.OK
        probe.assert_not_called()
        assert pytest.main(args) == pytest.ExitCode.OK
        probe.assert_called_once_with()


def test_filters_preserve_near_matches_and_end_with_render() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with quantstats_warnings.native_plot_warnings(True):
            _emit(_SET_BAD.lower(), PendingDeprecationWarning, "seaborn.matrix")
            _emit(_SET_BAD + " More.", PendingDeprecationWarning, "seaborn.matrix")
            _emit(_VERT, MatplotlibDeprecationWarning, "seaborn.categorical.extra")
            _emit(_VERT, UserWarning, "seaborn.categorical")
        _emit(_VERT, MatplotlibDeprecationWarning, "seaborn.categorical")

    assert [str(item.message) for item in caught] == [
        _SET_BAD.lower(),
        _SET_BAD + " More.",
        _VERT,
        _VERT,
        _VERT,
    ]
