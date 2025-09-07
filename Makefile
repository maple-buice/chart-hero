PYTHON ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
PRECOMMIT := $(VENV)/bin/pre-commit
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

# Defaults for dataset building (can be overridden on the command line)
ROOT ?= /Volumes/Media/CloneHero
OUT ?= datasets/processed_highres
LIMIT ?= 25
INDEX_DIR ?= artifacts/clonehero_charts_json

# Training/inference convenience variables
TAG ?= $(shell date "+%Y%m%d_%H%M%S")
WANDB ?= 0
# Resolve W&B flag
WANDB_FLAG := $(if $(filter 1 yes true,$(WANDB)),--use-wandb,--no-wandb)

MODELS_DIR := models/local_transformer_models
# Newest last.ckpt under timestamped subdirs
LATEST_LAST := $(shell ls -t $(MODELS_DIR)/*/last.ckpt 2>/dev/null | head -n 1)
DEFAULT_MODEL := $(if $(wildcard $(MODELS_DIR)/last.ckpt),$(MODELS_DIR)/last.ckpt,$(MODELS_DIR)/best_model.ckpt)
INFER_MODEL ?= $(if $(LATEST_LAST),$(LATEST_LAST),$(DEFAULT_MODEL))
CRUEL_SUMMER ?= https://youtu.be/SU8Jx80fCmg?si=LGzRTq-vx6xsylmZ
REDWINE_SUPERNOVA ?= https://youtu.be/KSbwHJMPq8w?si=zKQSXmGtMWH8n1DU
# Optional: provide multiple links separated by spaces
LINKS ?= $(CRUEL_SUMMER) $(REDWINE_SUPERNOVA)
# Optional inference preset; leave empty by default.
# Set PRESET=conservative or PRESET=aggressive when desired.
PRESET ?=
# Build flag only when PRESET is non-empty
PRESET_FLAG := $(if $(strip $(PRESET)),--preset $(PRESET),)

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
	@echo "  make dataset-highres ROOT=/Volumes/Media/CloneHero LIMIT=50 OUT=datasets/processed_highres  # Build hi-res dataset subset"
	@echo "  make train-highres TAG=myrun WANDB=1  # Train with local_highres on processed_highres"
	@echo "  make infer LINK='https://youtu.be/...?...' PRESET=conservative  # Run inference with newest model"

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

# Build a high-resolution dataset from Clone Hero folders.
# Variables:
# - ROOT: Clone Hero songs root (default /Volumes/Media/CloneHero)
# - OUT:  output dataset directory (default datasets/processed_highres)
# - LIMIT: number of songs to select (default 50)
.PHONY: dataset-highres
dataset-highres:
	$(PY) -m chart_hero.train.build_dataset \
		--roots "$(ROOT)" \
		--out-dir "$(OUT)" \
		--config local_highres \
		--json-index-dir "$(INDEX_DIR)" \
		--limit-songs $(LIMIT) \
		--min-align-score 0.05 \
		--dedupe

.PHONY: train-highres
train-highres:
	$(PY) scripts/train_highres.py \
		$(WANDB_FLAG) \
		--experiment-tag $(TAG)

.PHONY: infer
infer:
	@for L in $(LINKS); do \
		echo "Running inference for $$L with model $(INFER_MODEL)"; \
		$(PY) ./src/chart_hero/ \
			--export-clonehero \
			--to-clonehero \
			-l "$$L" \
			--model-path="$(INFER_MODEL)" \
			$(PRESET_FLAG) || exit $$?; \
	done

# Basic evaluation against a known-good notes.mid
# Required vars: AUDIO=/path/to/song.ogg MID=/path/to/notes.mid
# Optional: NMS=11 AG=0.55 HF=0.32 THR= (global threshold) DISABLE_CALIB=1
.PHONY: eval
eval:
	@if [ -z "$(AUDIO)" ] || [ -z "$(MID)" ]; then \
		echo "Usage: make eval AUDIO=/path/song.ogg MID=/path/notes.mid [NMS=11 AG=0.55 HF=0.32]"; \
		exit 2; \
	fi
	@echo "Evaluating $(AUDIO) vs $(MID) using model $(INFER_MODEL)";
	$(PY) -m chart_hero.eval.evaluate_chart \
		--audio "$(AUDIO)" \
		--mid "$(MID)" \
		--model "$(INFER_MODEL)" \
		--patch-stride 1 \
		$(if $(strip $(NMS)),--nms-k $(NMS),) \
		$(if $(strip $(AG)),--activity-gate $(AG),) \
		$(if $(strip $(HF)),--cymbal-hf-gate $(HF),) \
		$(if $(strip $(THR)),--threshold $(THR),) \
		$(if $(filter 1 yes true,$(DISABLE_CALIB)),--disable-calibrated,) \
		|| true
