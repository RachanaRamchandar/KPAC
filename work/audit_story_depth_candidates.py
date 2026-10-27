#!/usr/bin/env python3
"""Audit unselected story listings for canonical groups already near 12–15 rows."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
INDEX = "https://www.kathakalipadam.com/index.php/en/padam-list/by-story"
EXISTING = {"67", "71", "72", "78", "83", "84", "102"}


def key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return re.sub(r"_+", "_", value)


def fetch(url: str) -> str:
    response = requests.get(url, timeout=45, headers={"User-Agent": "KathakaliPadam research audit"})
    response.raise_for_status()
    return response.text


def story_specs() -> list[dict]:
    soup = BeautifulSoup(fetch(INDEX), "html.parser")
    found = {}
    for anchor in soup.select('a[href*="sstory="]'):
        href = urljoin(INDEX, anchor.get("href", ""))
        query = parse_qs(urlparse(href).query)
        story_id = query.get("sstory", [""])[0]
        text = " ".join(anchor.get_text(" ", strip=True).split())
        match = re.match(r"(.+?)\s*\((\d+)\)$", text)
        if not match or story_id in EXISTING:
            continue
        label, count = match.group(1), int(match.group(2))
        if count < 30:
            continue
        found[story_id] = {
            "story_id": story_id,
            "story_label": label,
            "badge_count": count,
            "url": re.sub(r"limit=\d+", f"limit={count}", href),
        }
    return list(found.values())


def audit(spec: dict) -> tuple[dict, list[dict]]:
    soup = BeautifulSoup(fetch(spec["url"]), "html.parser")
    titles = [a.get_text(" ", strip=True) for a in soup.select('table.category a[href*="/all-uploads/"]')]
    counts = Counter(key(title) for title in titles)
    groups = []
    for title_key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= 8:
            groups.append({
                "story_id": spec["story_id"],
                "story_label": spec["story_label"],
                "padam_title_key": title_key,
                "listing_recordings": count,
                "gap_to_12_before_audio_validation": max(0, 12 - count),
                "gap_to_15_before_audio_validation": max(0, 15 - count),
                "source_listing_url": spec["url"],
            })
    summary = {
        **spec,
        "rendered_rows": len(titles),
        "groups_ge_8": sum(count >= 8 for count in counts.values()),
        "groups_ge_12": sum(count >= 12 for count in counts.values()),
        "rows_in_groups_ge_8": sum(count for count in counts.values() if count >= 8),
        "largest_exact_normalized_group": max(counts.values(), default=0),
    }
    return summary, groups


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    specs = story_specs()
    summaries, groups = [], []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(audit, spec): spec for spec in specs}
        for future in as_completed(futures):
            summary, detail = future.result()
            summaries.append(summary)
            groups.extend(detail)
            print(summary["story_label"], summary["rendered_rows"], summary["groups_ge_12"], flush=True)
    summaries.sort(key=lambda row: (-row["groups_ge_12"], -row["rows_in_groups_ge_8"], row["story_label"]))
    groups.sort(key=lambda row: (-row["listing_recordings"], row["story_label"], row["padam_title_key"]))
    write(OUT / "kathakalipadam_remaining_story_depth_audit.csv", summaries)
    if groups:
        write(OUT / "kathakalipadam_remaining_repeat_group_candidates.csv", groups)
    print(f"stories={len(summaries)} groups_ge_8={len(groups)} groups_ge_12={sum(row['listing_recordings'] >= 12 for row in groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
