.PHONY: run run-prod install install-dev test test-unit test-api lint format ingest queries clean

# ── Development server ────────────────────────────────────────────────────────
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-prod:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── Dependencies ──────────────────────────────────────────────────────────────
install:
	pip3 install -r requirements.txt --break-system-packages

install-dev:
	pip3 install -r requirements-dev.txt --break-system-packages

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=app --cov-report=term-missing

test-unit:
	pytest tests/test_ingest tests/test_retrieval tests/test_processing tests/test_generation -v

test-api:
	pytest tests/test_api -v

# ── Code quality ─────────────────────────────────────────────────────────────
lint:
	ruff check app/ tests/
	black --check app/ tests/

format:
	black app/ tests/
	isort app/ tests/

# ── Demo workflow ─────────────────────────────────────────────────────────────
ingest:
	python3 scripts/ingest_demo_data.py

queries:
	python3 scripts/run_example_queries.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov/ .bm25_corpus.pkl
