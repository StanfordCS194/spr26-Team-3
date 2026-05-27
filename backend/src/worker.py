"""Inngest worker entrypoint. Same Python process as the API would also work
— Inngest doesn't strictly require a separate worker — but splitting them
mirrors Haleum's deploy story and prevents long jobs from competing for the
API's request-handling thread pool.

Run with: `python -m src.worker`
"""
from __future__ import annotations

import asyncio

import inngest.fast_api
from fastapi import FastAPI

from src.inngest_client import FUNCTIONS, inngest_client

# Importing feature modules registers Inngest functions into FUNCTIONS.
from src.features.reconstruction import inngest_functions as _reconstruct_fns  # noqa: F401
from src.features.training import inngest_functions as _training_fns  # noqa: F401


def make_worker_app() -> FastAPI:
    """A minimal FastAPI app that only serves Inngest's discovery endpoint.
    The Inngest Dev Server hits this to pick up function definitions and
    invokes them via HTTP — that's how all SDKs (Python, TS, Go) work.
    """
    app = FastAPI(title="WorldScan worker")
    inngest.fast_api.serve(app, inngest_client, FUNCTIONS)
    return app


async def main() -> None:
    import uvicorn

    config = uvicorn.Config(make_worker_app(), host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
