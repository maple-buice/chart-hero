Development Setup

- Prereqs: Python 3.11 arm64 on Apple Silicon for MPS (or use CPU/CUDA).

Quick start

- Create venv: make venv && source .venv/bin/activate
- Install deps: make install
- Install hooks: make pre-commit
- Run tests: make test
- Lint/format: make lint && make format

Notes

- Set AUDD API env if you use song identification: export AUDD_API_TOKEN=... (see .env.example)
- For a quick sanity training run on dummy data: make train-quick
- PyTest coverage is configured for src/chart_hero; logs go to logs/pytest-logs.log
