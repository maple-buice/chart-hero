GitHub Projects (v2) Setup
==========================

Project URL
- Using: https://github.com/users/maple-buice/projects/1

What’s included
- Issue templates: `.github/ISSUE_TEMPLATE/*.yml` (feature, bug, chore, test)
- Auto-add to Project: `.github/workflows/project-auto-add.yml` (adds issues/PRs)
- Roadmap/Tech-debt sync: `.github/workflows/sync-roadmap.yml` + `.github/scripts/sync_roadmap.py`
  - Parses `docs/ROADMAP.md` and `docs/TECH_DEBT.md` and creates issues with labels

Required secret
- Create a fine-grained PAT with at least:
  - Repository permissions: Issues (Read/Write)
  - Organization/User permissions: Projects (Read/Write) for your user project
- Add it to this repo secrets as `GH_PROJECT_TOKEN`.

Enable automation
1) Add secret `GH_PROJECT_TOKEN` (Settings → Secrets and variables → Actions).
2) Trigger “Sync ROADMAP/TECH_DEBT to issues” workflow manually once.
3) New/updated issues will be auto-added to the Project by the “Add issues/PRs to Project” workflow.

Custom fields (optional)
- If you add custom fields to your Project (e.g., Priority, Area), we can extend the workflows to set them using a small GitHub Script step.

Notes
- The `GITHUB_TOKEN` cannot access user Projects v2; a PAT is required.
- The sync script is idempotent by title; it won’t duplicate existing issues.
