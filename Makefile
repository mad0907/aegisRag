.PHONY: venv install up down logs init-db ingest api test eval verify-audit fmt phoenix webui

# python3 is often too new for crewai's published releases (see README) — pin to 3.11.
venv:
	/opt/homebrew/bin/python3.11 -m venv .venv || python3.11 -m venv .venv

install: venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/python scripts/patch_ragas.py

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

init-db:
	.venv/bin/python -m aegisrag.database.db

ingest:
	.venv/bin/python -m aegisrag.ingestion.pipeline

api:
	.venv/bin/uvicorn aegisrag.api.main:app --reload --port 8080

test:
	.venv/bin/pytest

eval:
	.venv/bin/python -m aegisrag.evaluation.run

verify-audit:
	.venv/bin/python -m aegisrag.audit.hashchain

# Convenience only — Phoenix and OpenWebUI aren't "run" by make, they're already up as part of
# `make up` (docker-compose); these just open the browser to the right URL.
phoenix:
	open http://localhost:6006

webui:
	open http://localhost:3000

fmt:
	.venv/bin/ruff check --fix src tests
