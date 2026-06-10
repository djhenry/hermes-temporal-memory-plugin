.PHONY: install install-hermes test test-unit test-integration test-live test-all benchmark lint clean

HERMES_DIR  := vendor/hermes-agent
HERMES_REPO := https://github.com/NousResearch/hermes-agent.git
PYTHON      := python3
PIP         := $(PYTHON) -m pip
PYTEST      := $(PYTHON) -m pytest

# ── Setup ──────────────────────────────────────────────────────────────────

$(HERMES_DIR)/.git:
	git clone --depth 1 $(HERMES_REPO) $(HERMES_DIR)

install-hermes: $(HERMES_DIR)/.git
	$(PIP) install -e $(HERMES_DIR) --no-deps -q

install: install-hermes
	$(PIP) install -e ".[dev]" -q

# ── Environment ────────────────────────────────────────────────────────────

# Start Neo4j via Docker and validate the full live stack
setup-neo4j:
	docker compose up -d
	bash scripts/setup_env.sh --neo4j

# Install Kuzu (embedded, no Docker) and validate — dev/offline only
setup-kuzu:
	bash scripts/setup_env.sh --kuzu

# ── Tests ──────────────────────────────────────────────────────────────────

# Unit tests — no hermes-agent or graphiti-core needed
test-unit:
	$(PYTEST) tests/test_memory_index.py -v

# Integration tests with mocked Graphiti client — requires hermes-agent (make install)
test-integration:
	$(PYTEST) tests/integration/ -v -m "not live"

# Live integration tests — requires graphiti-core[falkordblite] + extraction LLM key
test-live:
	$(PYTEST) tests/integration/ -v -m "live"

test: test-unit test-integration

# ── Benchmark ──────────────────────────────────────────────────────────────

# Compare retrieval recall: hermes-graphiti vs Hermes built-in FTS5 baseline
benchmark:
	$(PYTHON) benchmarks/benchmark.py

benchmark-verbose:
	$(PYTHON) benchmarks/benchmark.py --verbose

benchmark-advanced:
	$(PYTHON) benchmarks/advanced_benchmark.py

# ── Quality ────────────────────────────────────────────────────────────────

lint:
	ruff check plugins/ tests/

# ── Housekeeping ───────────────────────────────────────────────────────────

clean:
	rm -rf vendor/ *.egg-info dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
