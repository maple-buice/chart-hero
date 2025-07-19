# Guide: Future Project Improvements

This document outlines the plan for significant improvements to the project's tooling and infrastructure. These steps are designed to enhance reproducibility, code quality, and developer efficiency.

## Prerequisite: Poetry Migration

Before proceeding with the steps below, the project's dependency management must be migrated from `requirements.txt` to Poetry. The detailed steps for this are documented in `POETRY_MIGRATION.md`.

---

## Step 1: Consolidate Configuration

**Goal:** Use `pyproject.toml` as the single source of truth for all tool configurations, reducing clutter and improving maintainability.

**Actions:**

1.  **Migrate `pytest` Configuration:** Move the settings from `pytest.ini` into `pyproject.toml` under the `[tool.pytest.ini_options]` section.
2.  **Migrate `ruff` Configuration:** Move the `ruff` settings into `pyproject.toml` under the `[tool.ruff]` section.
3.  **Remove Old Files:** Once migrated, delete the `pytest.ini` file.

---

## Step 2: Implement Static Type Checking

**Goal:** Add static type checking to the project to catch a wide range of potential bugs before runtime.

**Actions:**

1.  **Add `mypy`:** Add `mypy` as a development dependency in `pyproject.toml`.
    ```bash
    poetry add --group dev mypy
    ```
2.  **Configure `mypy`:** Add a `[tool.mypy]` section to `pyproject.toml` to configure paths and checking strictness.
3.  **Add to Pre-Commit:** Add a `mypy` hook to `.pre-commit-config.yaml` to run type checks automatically before each commit.
4.  **Add Type Hints:** Incrementally add type hints to the function signatures throughout the codebase.

---

## Step 3: Implement Data & Model Versioning

**Goal:** Version large data and model files using DVC (Data Version Control) to make experiments fully reproducible and keep the Git repository lightweight.

**Actions:**

1.  **Add `dvc`:** Add `dvc` as a development dependency.
    ```bash
    poetry add --group dev dvc
    ```
2.  **Initialize DVC:** Set up DVC in the project. This will create a few configuration files.
    ```bash
    dvc init
    ```
3.  **Track Directories:** Tell DVC to start tracking the directories containing large files.
    ```bash
    dvc add datasets
    dvc add models
    ```
4.  **Configure Remote Storage (Optional but Recommended):** Set up a DVC remote (e.g., S3, Google Drive, or a local cache) to store the versioned data.
    ```bash
    dvc remote add -d myremote s3://my-bucket/chart-hero-dvc
    ```

---

## Step 4: Codify the Development Environment

**Goal:** Create a reproducible, containerized development environment using VS Code Dev Containers.

**Actions:**

1.  **Create Configuration Directory:** Create a `.devcontainer` directory in the project root.
2.  **Create `devcontainer.json`:** Inside `.devcontainer`, create a `devcontainer.json` file that specifies:
    *   A base Docker image (e.g., a standard Python 3.12 image).
    *   A list of VS Code extensions to automatically install (e.g., Python, Ruff, Mypy).
    *   A `postCreateCommand` to automatically run `poetry install` after the container is built.
3.  **Commit:** Commit the `.devcontainer` directory to the repository. This will enable any developer with VS Code and Docker to instantly build and connect to a consistent development environment.
