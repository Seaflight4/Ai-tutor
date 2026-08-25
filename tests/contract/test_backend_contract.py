"""Contract test runner — runs every conformance case against every backend.

The conformance cases live in `conformance.py` as plain functions taking a
`backend`. This file parametrizes them across the three backend fixtures
(`in_memory`, `sqlite`, `supabase_mock`) so each case runs three times.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tests.contract import conformance

# Collect every conformance test function (those named `test_*`).
_CASES: list[Callable[..., Any]] = [
    getattr(conformance, name)
    for name in sorted(dir(conformance))
    if name.startswith("test_") and callable(getattr(conformance, name))
]

_BACKENDS = ["in_memory", "sqlite", "supabase_mock"]


def _case_id(case: Callable[..., Any], backend: str) -> str:
    return f"{case.__name__}[{backend}]"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.__name__)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_contract(case: Callable[..., Any], backend: str, request) -> None:
    """Run one conformance case against one backend fixture."""
    backend_obj = request.getfixturevalue(backend)
    case(backend_obj)
