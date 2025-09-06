GitHub Projects (v2) Setup
==========================

Project URL
- Using: https://github.com/users/maple-buice/projects/1

What’s included
- Issue templates: `.github/ISSUE_TEMPLATE/*.yml` (feature, bug, chore, test)
- Auto-add to Project: `.github/workflows/project-auto-add.yml` (adds issues/PRs)
  - Also sets Project fields:
    - `Status` → `Todo` by default
    - `Priority` → inferred from Issue Form field `Priority: P0|P1|P2|P3` (if present)
    - `Status` → `Done` on issue close or merged PR
    - `Area` → matches any label that equals an Area option (case-insensitive)
- Roadmap/Tech-debt sync: `.github/workflows/sync-roadmap.yml` + `.github/scripts/sync_roadmap.py`
  - Parses `docs/ROADMAP.md` and `docs/TECH_DEBT.md` and creates issues with labels
  - Now also closes open roadmap issues when corresponding items are checked `[x]` in the doc
  - Triggers on push to `docs/ROADMAP.md` and `docs/TECH_DEBT.md`
- Label sync: `.github/workflows/label-sync.yml` + `.github/labels.yml`
  - Ensures standard labels exist (type + area) and are ready for Project field mapping
 - Project setup: `.github/workflows/project-setup.yml`
   - Creates Project v2 single-select fields (Status, Priority, Area)
   - Notes missing options to add via UI when GraphQL option update is unavailable

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

Roadmap formatting (for reliable sync)
- Use ATX headers (e.g., `## Near-Term (High Priority)`).
- One item per top-level bullet using `- [ ]` or `- [x]`.
- Optional stable IDs in parentheses (e.g., `(R-001)`) inside the bullet: these make the sync idempotent across renames.
- Example:
  - `- [ ] (R-001) Switch export to notes.mid`
  - `- [x] (R-002) Add PART VOCALS exporter`

Custom fields (optional)
- If you add custom fields to your Project (e.g., Priority, Area), we can extend the workflows to set them using a small GitHub Script step.

Notes
- The `GITHUB_TOKEN` cannot modify user Projects v2; a classic PAT with `project` scope is required for project automation.
- The roadmap sync workflow uses `GITHUB_TOKEN` to create issues; it does not require your PAT.
- The sync script is idempotent by title; it won’t duplicate existing issues.

4) Run “Ensure Project fields” to create fields (uses `GH_PROJECT_TOKEN`). If the workflow warns about missing options, add them in the Project UI (Status: Todo/In Progress/Done; Priority: P0–P3; Area: lyrics/vocals/export/downloader/inference).

Project field prerequisites
- Create single-select fields in your Project:
  - `Status` with options: `Todo`, `In Progress`, `Done` (created by setup workflow). The add-to-project workflow sets `Todo` on new items and `Done` on close/merge.
  - `Priority` with options: `P0`, `P1`, `P2`, `P3` (created by setup workflow). The add-to-project workflow reads the Issue body for `Priority: P?` and sets the matching option.
  - `Area` with options matching labels you use (e.g., `lyrics`, `vocals`, `export`, `downloader`, `inference`) — created by setup workflow. The add-to-project workflow maps labels to the matching Area option.

Standard labels
- Types: `feature`, `bug`, `chore`, `testing`, `roadmap`, `tech-debt`
- Areas: `lyrics`, `vocals`, `export`, `downloader`, `inference`
Run the “Sync labels” workflow (or push `.github/labels.yml`) to create/update them.
