"""Opt-in tests against real LLM clients and their offline evidence verifiers."""

import pytest

# Keep comparison values in the AssertionError text used by workflow diagnostics.
pytest.register_assert_rewrite(
    "tests.e2e.workflow_lifecycle",
    "tests.e2e.workflow_execution",
    "tests.e2e.workflow_evidence",
)
