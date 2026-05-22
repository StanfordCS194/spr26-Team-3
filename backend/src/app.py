"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

import inngest.fast_api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.features.builds.router import router as builds_router
from src.features.projects.router import router as projects_router
from src.features.reconstruction.router import router as reconstruction_router
from src.features.replay.router import router as replay_router
from src.features.training.router import router as training_router
from src.features.validation.router import router as validation_router
from src.inngest_client import FUNCTIONS, inngest_client

# Importing feature modules registers Inngest functions into FUNCTIONS.
from src.features.builds import inngest_functions as _build_fns  # noqa: F401
from src.features.reconstruction import inngest_functions as _reconstruct_fns  # noqa: F401
from src.features.replay import inngest_functions as _replay_fns  # noqa: F401
from src.features.training import inngest_functions as _training_fns  # noqa: F401
from src.features.validation import inngest_functions as _validation_fns  # noqa: F401
from src import inngest_failures as _inngest_failures  # noqa: F401

_settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _settings.data_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(
    title="WorldScan API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# Feature routers — every screen in the SPA maps to one of these.
app.include_router(projects_router, prefix="/api/projects", tags=["projects"])
app.include_router(reconstruction_router, prefix="/api", tags=["reconstruction"])

# Static files: meshes, point clouds, thumbnails — read-only by frontend.
from fastapi.staticfiles import StaticFiles  # noqa: E402
_data_dir = _settings.data_dir
_data_dir.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(_data_dir)), name="data")
app.include_router(validation_router, prefix="/api/projects", tags=["validation"])
app.include_router(builds_router, prefix="/api/projects", tags=["builds"])
app.include_router(training_router, prefix="/api/projects", tags=["training"])
app.include_router(replay_router, prefix="/api/projects", tags=["replay"])

# Inngest serve route — the dev server polls this to discover functions.
inngest.fast_api.serve(app, inngest_client, FUNCTIONS)
