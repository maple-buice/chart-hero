PYTHON ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PRECOMMIT := $(VENV)/bin/pre-commit
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

.PHONY: help
help:
	@echo "Common targets:"
	@echo "  make venv           # Create Python 3.11 venv"
	@echo "  make install        # Install deps + package editable"
	@echo "  make pre-commit     # Install git hooks"
	@echo "  make test           # Run pytest"
	@echo "  make lint           # Ruff lint"
	@echo "  make format         # Ruff format"
	@echo "  make train-quick    # Quick sanity training run"

.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Run: source $(VENV)/bin/activate"

.PHONY: install
install: venv
	$(PIP) install -U pip wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: pre-commit
pre-commit: install
	$(PRECOMMIT) install
	$(PRECOMMIT) install --hook-type pre-push

.PHONY: test
test:
	$(PYTEST) -q || true

.PHONY: lint
lint:
	$(RUFF) check .

.PHONY: format
format:
	$(RUFF) format .

.PHONY: train-quick
train-quick:
	$(PY) -m chart_hero.model_training.train_transformer \
		--config local \
		--no-wandb \
		--quick-test \
		--data-dir ./tests/assets/dummy_data/processed \
		--experiment-tag dev_quick
