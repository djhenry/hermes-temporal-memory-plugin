.PHONY: install install-hermes test test-unit test-integration test-live test-all lint clean

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
	$(PIP) install -e ".[sqlite,dev]" -q

# ── Tests ──────────────────────────────────────────────────────────────────

# Unit tests — no hermes-agent or graphiti-core needed
test-unit:
	$(PYTEST) tests/test_memory_index.py -v

# Integration tests with mocked Graphiti client — requires hermes-agent (make install)
test-integration:
	$(PYTEST) tests/integration/ -v -m "not live"

# Live integration tests — requires graphiti-core[sqlite] + extraction LLM key
test-live:
	$(PYTEST) tests/integration/ -v -m "live"

test: test-unit test-integration

# ── Quality ────────────────────────────────────────────────────────────────

lint:
	ruff check plugins/ tests/

# ── Housekeeping ───────────────────────────────────────────────────────────

clean:
	rm -rf vendor/ *.egg-info dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
