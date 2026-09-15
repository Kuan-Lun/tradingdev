"""Locate the canonical quality configuration in a checkout or installed wheel."""

from pathlib import Path


def quality_config_path() -> Path:
    """Return the same policy used by repository checks and strategy validation."""
    package = Path(__file__).resolve().parents[1]
    installed_policy = package / "_quality" / "pyproject.toml"
    if installed_policy.is_file():
        return installed_policy
    checkout_policy = package.parent.parent / "pyproject.toml"
    if package.parent.name == "src" and checkout_policy.is_file():
        return checkout_policy
    raise FileNotFoundError(
        "TradingDev quality policy is missing from the installation"
    )
