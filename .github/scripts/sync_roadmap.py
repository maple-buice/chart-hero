#!/usr/bin/env python3
import os
import re
import sys
import json
import time
from typing import Iterable

import requests


REPO = os.environ.get("GITHUB_REPOSITORY")
# Prefer explicit GH_TOKEN, else fall back to Actions' GITHUB_TOKEN
TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
API = "https://api.github.com"

HEADERS = {}
if TOKEN:
    HEADERS = {
        "Authorization": f"Bearer {TOKEN}",
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


def make_issue(title: str, body: str, labels: list[str]) -> None:
    r = requests.post(
        f"{API}/repos/{REPO}/issues",
        headers=HEADERS,
        json={"title": title, "body": body, "labels": labels},
    )
    r.raise_for_status()


def parse_bullets(md: str, section: str) -> list[str]:
    # crude parse: collect '- ' bullets under a section header until next header
    lines = md.splitlines()
    items: list[str] = []
    in_sec = False
    hdr_re = re.compile(r"^#{1,6}\\s+" + re.escape(section) + r"$", re.I)
    any_hdr = re.compile(r"^#{1,6}\\s+")
    for ln in lines:
        if hdr_re.match(ln.strip()):
            in_sec = True
            continue
        if in_sec and any_hdr.match(ln.strip()):
            break
        if in_sec and ln.strip().startswith("- "):
            items.append(ln.strip()[2:].strip())
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
    for sec in roadmap_sections:
        for item in parse_bullets(roadmap_md, sec):
            title = f"Roadmap: {item}"
            if title not in existing_titles:
                make_issue(
                    title,
                    body=f"Auto-synced from ROADMAP.md section: {sec}\n\nSource: `docs/ROADMAP.md`\n\n- [ ] Track progress here and close when done.",
                    labels=["roadmap"],
                )
                created += 1
                time.sleep(0.5)

    # Tech debt lines under "Unit Testing TODOs (new features)" and below
    for item in parse_bullets(tech_md, "Unit Testing TODOs (new features)"):
        title = f"Tech debt (tests): {item}"
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
