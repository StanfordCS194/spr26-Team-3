# WorldScan dev tasks. Run `just` for a list.

# default: print the menu
default:
    @just --list

# bring up the full stack: postgres, inngest-dev, api, worker, frontend
dev:
    docker compose up --watch

# tear everything down (keeps the postgres volume)
down:
    docker compose down

# tear everything down AND drop the postgres volume
nuke:
    docker compose down -v

# run all tests (pytest + vitest)
test:
    cd backend && uv run pytest
    yarn workspace @worldscan/frontend run test

# lint everything (ruff + black + eslint + prettier)
lint:
    cd backend && uv run ruff check . && uv run black --check .
    yarn lint

# regenerate the TypeScript client from the running backend's OpenAPI
typegen:
    curl -s http://localhost:8000/openapi.json > shared/openapi.json
    yarn workspace @worldscan/shared run codegen

# apply pending Alembic migrations
db-migrate:
    cd backend && uv run alembic upgrade head

# create a new Alembic revision from current model state
db-revision MSG:
    cd backend && uv run alembic revision --autogenerate -m "{{MSG}}"

# drop the dev DB, recreate, migrate, seed
db-reset:
    docker compose exec postgres psql -U worldscan -c "DROP DATABASE IF EXISTS worldscan;"
    docker compose exec postgres psql -U worldscan -c "CREATE DATABASE worldscan;"
    cd backend && uv run alembic upgrade head

# headless smoke test (no docker, no UI) — the rl_env CLI
demo:
    .venv/bin/python -m rl_env demo
