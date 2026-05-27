FROM python:3.13-slim

# system deps: ffmpeg for frame extraction, libgl for trimesh, libglib for opencv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 git \
 && rm -rf /var/lib/apt/lists/*

# uv for fast dependency resolution
RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml ./
# rl_env is mounted at runtime via docker-compose; install backend deps only
RUN uv sync --no-install-workspace --no-dev || true

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
