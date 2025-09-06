GitHub Projects (v2) Setup
==========================

Project URL
- Using: https://github.com/users/maple-buice/projects/1

What’s included
- Issue templates: `.github/ISSUE_TEMPLATE/*.yml` (feature, bug, chore, test)
- Auto-add to Project: `.github/workflows/project-auto-add.yml` (adds issues/PRs)
- Roadmap/Tech-debt sync: `.github/workflows/sync-roadmap.yml` + `.github/scripts/sync_roadmap.py`
  - Parses `docs/ROADMAP.md` and `docs/TECH_DEBT.md` and creates issues with labels

Required secret (for adding to a User Project v2)
- Create a Personal Access Token (classic), not fine-grained:
  - Go to Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Generate new token (classic) with scopes:
    - `project` (Full control of user projects) — required for Projects v2
    - `repo` (at least issues: write) — only if you want to use the token for issue ops; our sync uses `GITHUB_TOKEN` instead
- Add it to this repo’s Actions secrets as `GH_PROJECT_TOKEN`.

Enable automation
1) Add secret `GH_PROJECT_TOKEN` (Settings → Secrets and variables → Actions).
2) Trigger “Sync ROADMAP/TECH_DEBT to issues” workflow manually once.
3) Open a new issue or PR — it will be auto-added to the Project by the “Add issues/PRs to Project” workflow.
   - Note: the add-to-project workflow no longer supports manual (workflow_dispatch) runs. Manual runs cause a GraphQL error (contentId null) because there is no issue/PR context.

Custom fields (optional)
- If you add custom fields to your Project (e.g., Priority, Area), we can extend the workflows to set them using a small GitHub Script step.

Notes
- The `GITHUB_TOKEN` cannot modify user Projects v2; a classic PAT with `project` scope is required for project automation.
- The roadmap sync workflow uses `GITHUB_TOKEN` to create issues; it does not require your PAT.
- The sync script is idempotent by title; it won’t duplicate existing issues.
