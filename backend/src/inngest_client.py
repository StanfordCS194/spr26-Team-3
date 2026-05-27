"""Inngest client. Functions are registered in each feature's
`inngest_functions.py` and imported here so the worker entrypoint can serve
them all together.
"""
from __future__ import annotations

import inngest

from src.config import get_settings

_settings = get_settings()

inngest_client = inngest.Inngest(
    app_id=_settings.inngest_app_id,
    is_production=False,
)

# Functions registered in features modules; aggregated at startup.
# (imports are at the bottom of this module to avoid circular imports
#  — see src/features/*/inngest_functions.py)
FUNCTIONS: list[inngest.Function] = []


def register(fn: inngest.Function) -> inngest.Function:
    FUNCTIONS.append(fn)
    return fn
