#!/usr/bin/env python3
import os
import re
import sys
import json
import time
from typing import Iterable, Optional, Dict

import requests


REPO = os.environ.get("GITHUB_REPOSITORY")
# Token used for issue operations
TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
# Optional token permitted to modify Projects v2 (classic PAT with 'project' scope)
PROJECT_TOKEN = os.environ.get("GH_PROJECT_TOKEN") or os.environ.get("PROJECT_TOKEN")
PROJECT_URL = os.environ.get("PROJECT_URL")
API = "https://api.github.com"

HEADERS = {}
if TOKEN:
    HEADERS = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.github+json",
    }

PROJ_HEADERS = {}
if PROJECT_TOKEN:
    PROJ_HEADERS = {
        "Authorization": f"Bearer {PROJECT_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def ensure_labels(names: list[str]) -> None:
    for name in names:
        r = requests.get(f"{API}/repos/{REPO}/labels/{name}", headers=HEADERS)
        if r.status_code == 200:
            continue
        requests.post(
            f"{API}/repos/{REPO}/labels",
            headers=HEADERS,
            json={"name": name, "color": "0e8a16" if name == "roadmap" else "5319e7"},
        )


def list_open_issues() -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        r = requests.get(
            f"{API}/repos/{REPO}/issues",
            headers=HEADERS,
            params={"state": "open", "per_page": 100, "page": page},
        )
        r.raise_for_status()
        items = r.json()
        if not items:
            break
        out.extend(items)
        page += 1
    return out


def make_issue(title: str, body: str, labels: list[str]) -> dict:
    r = requests.post(
        f"{API}/repos/{REPO}/issues",
        headers=HEADERS,
        json={"title": title, "body": body, "labels": labels},
    )
    r.raise_for_status()
    return r.json()


def _parse_project_url(url: str) -> Optional[tuple[str, int]]:
    if not url:
        return None
    m = re.search(r"github\.com/(?:users|orgs)/([^/]+)/projects/(\d+)", url)
    if not m:
        return None
    return m.group(1), int(m.group(2))


def get_project_id(project_url: str) -> Optional[str]:
    if not PROJECT_TOKEN:
        return None
    parsed = _parse_project_url(project_url)
    if not parsed:
        return None
    owner, number = parsed
    query = """
    query($login:String!, $number:Int!){
      user(login:$login){ projectV2(number:$number){ id } }
      organization(login:$login){ projectV2(number:$number){ id } }
    }
    """
    r = requests.post(
        f"{API}/graphql",
        headers=PROJ_HEADERS,
        json={"query": query, "variables": {"login": owner, "number": number}},
    )
    if r.status_code != 200:
        return None
    js = r.json()
    try:
        uid = js["data"]["user"]["projectV2"]["id"] if js["data"].get("user") else None
        if uid:
            return uid
        oid = (
            js["data"]["organization"]["projectV2"]["id"]
            if js["data"].get("organization")
            else None
        )
        return oid
    except Exception:
        return None


def add_item_to_project(project_id: str, content_node_id: str) -> bool:
    if not PROJECT_TOKEN or not project_id or not content_node_id:
        return False
    mutation = """
    mutation($projectId:ID!, $contentId:ID!){
      addProjectV2ItemById(input:{projectId:$projectId, contentId:$contentId}){ item { id } }
    }
    """
    r = requests.post(
        f"{API}/graphql",
        headers=PROJ_HEADERS,
        json={
            "query": mutation,
            "variables": {"projectId": project_id, "contentId": content_node_id},
        },
    )
    try:
        r.raise_for_status()
        js = r.json()
        return bool(
            js.get("data", {}).get("addProjectV2ItemById", {}).get("item", {}).get("id")
        )
    except Exception:
        return False


def close_issue(number: int) -> None:
    r = requests.patch(
        f"{API}/repos/{REPO}/issues/{number}",
        headers=HEADERS,
        json={"state": "closed"},
    )
    r.raise_for_status()


ITEM_RE = re.compile(r"^-\s*(\[(?P<mark>[ xX])\])?\s*(?P<text>.+?)\s*$")
ID_RE = re.compile(r"\((?P<id>R-\d{3,})\)")


def parse_bullets(md: str, section: str) -> list[dict]:
    """Collect top-level bullets ('- ') under a section.

    Supports:
    - ATX headers: '# Section'
    - Setext-style headers: 'Section' followed by '---'/'===' line
    - Plain section lines: a line exactly matching the section text (case-insensitive)
    Terminates when the next header or next non-indented, non-bullet line appears.
    """
    lines = md.splitlines()
    items: list[dict] = []
    in_sec = False
    sec_lower = section.strip().lower()
    any_hdr = re.compile(r"^#{1,6}\\s+")

    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        # Enter section on ATX header
        if re.match(r"^#{1,6}\\s+" + re.escape(section) + r"$", s, flags=re.I):
            in_sec = True
            i += 1
            continue
        # Enter section on Setext header or plain line equal to section
        if not in_sec and s.lower() == sec_lower:
            # If next line is setext underline, skip it
            if i + 1 < len(lines) and re.match(r"^\s*(=+|-+)\s*$", lines[i + 1]):
                i += 2
            else:
                i += 1
            in_sec = True
            continue

        if in_sec:
            # Stop at next ATX header
            if any_hdr.match(s):
                break
            # Stop at next non-indented, non-bullet content line (new section title)
            if s and not ln.startswith(" ") and not s.startswith("- "):
                break
            # Collect only top-level bullets (no leading spaces)
            if ln.startswith("- "):
                m = ITEM_RE.match(s)
                if not m:
                    i += 1
                    continue
                text = m.group("text").strip()
                checked = (m.group("mark") or " ").strip().lower() == "x"
                sid: Optional[str] = None
                idm = ID_RE.search(text)
                if idm:
                    sid = idm.group("id").upper()
                    text = ID_RE.sub("", text).strip()
                items.append({"text": text, "checked": checked, "id": sid})
        i += 1
    return items


def sync():
    if not REPO or not TOKEN:
        print(
            "Missing GITHUB_REPOSITORY or token (GH_TOKEN/GITHUB_TOKEN)",
            file=sys.stderr,
        )
        sys.exit(1)

    ensure_labels(["roadmap", "tech-debt", "testing"])
    existing = list_open_issues()
    existing_titles = {it["title"] for it in existing}
    existing_by_title = {it["title"]: it for it in existing}

    roadmap_md = read("docs/ROADMAP.md")
    tech_md = read("docs/TECH_DEBT.md")

    roadmap_sections = [
        "Near-Term (High Priority)",
        "Modeling & Inference (Next)",
        "Training Data & Labels",
        "Advanced Modeling",
        "Tooling & Integration",
        "Attribution & Licensing",
        "Backlog / Ideas",
        "Lyrics & Vocals",
        "YouTube Music Premium Cookies",
    ]

    created = 0
    # Build desired items (keyed by ID or normalized text)
    desired: Dict[str, dict] = {}
    for sec in roadmap_sections:
        for item in parse_bullets(roadmap_md, sec):
            key = item.get("id") or item["text"].lower()
            desired[key] = {**item, "section": sec}

    # Create issues for unchecked items; close open issues for checked items
    for key, it in desired.items():
        disp = it["text"]
        sec = it["section"]
        sid = it.get("id")
        checked = bool(it.get("checked"))
        title = f"Roadmap{f' [{sid}]' if sid else ''}: {disp}"
        if title not in existing_titles and not checked:
            issue = make_issue(
                title,
                body=(
                    f"Auto-synced from ROADMAP.md section: {sec}\n\n"
                    f"Source: `docs/ROADMAP.md`\n\n"
                    f"- [ ] Track progress here and close when done."
                ),
                labels=["roadmap"],
            )
            # Add the newly created issue to Project if configured
            try:
                if PROJECT_URL:
                    pid = get_project_id(PROJECT_URL)
                    node_id = issue.get("node_id")
                    if pid and node_id:
                        ok = add_item_to_project(pid, node_id)
                        if not ok:
                            print("Warning: failed to add issue to Project")
            except Exception:
                pass
            created += 1
            time.sleep(0.3)
        elif title in existing_titles and checked:
            issue = existing_by_title[title]
            if issue.get("state") == "open":
                close_issue(issue["number"])

    # Tech debt lines under "Unit Testing TODOs (new features)" and below
    for item in parse_bullets(tech_md, "Unit Testing TODOs (new features)"):
        disp = item["text"]
        title = f"Tech debt (tests): {disp}"
        if title not in existing_titles:
            make_issue(
                title,
                body="Auto-synced from TECH_DEBT.md. Write tests and link PRs here.",
                labels=["tech-debt", "testing"],
            )
            created += 1
            time.sleep(0.5)

    print(f"Created {created} issues")


if __name__ == "__main__":
    sync()
