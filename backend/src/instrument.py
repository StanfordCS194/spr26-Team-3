"""Sentry init. Imported via `--import` flag in uvicorn for early init."""
from __future__ import annotations

import sentry_sdk

from src.config import get_settings

_settings = get_settings()
if _settings.sentry_dsn:
    sentry_sdk.init(dsn=_settings.sentry_dsn, traces_sample_rate=0.1)
