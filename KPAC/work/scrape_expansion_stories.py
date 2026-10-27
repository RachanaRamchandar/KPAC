#!/usr/bin/env python3
"""Scrape repeat-rich Kathakalipadam story listings and detail metadata."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "corpus_expansion"
INVENTORY_CSV = WORK / "story_inventory_raw.csv"
DETAIL_JSON = WORK / "story_details_raw.json"
BASE = "https://www.kathakalipadam.com"
USER_AGENT = "Mozilla/5.0 (KathakaliPadam computational research corpus expansion)"

STORIES = [
    {"story_id": "84", "story_label": "Nalacharitham Nalaam Divasam", "limit": 119},
    {"story_id": "72", "story_label": "Kuchela Vritham", "limit": 98},
    {"story_id": "67", "story_label": "Karna Shapadham", "limit": 97},
]


def key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return re.sub(r"_+", "_", value)


def session() -> requests.Session:
    client = requests.Session()
    retry = Retry(total=4, connect=4, read=4, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
    client.mount("https://", HTTPAdapter(max_retries=retry))
    client.headers.update({"User-Agent": USER_AGENT})
    return client


def listing_url(spec: dict[str, object]) -> str:
    return (
        f"{BASE}/index.php/en/padam-list/by-story?option=com_musicsearch&limitstart=0"
        f"&sragam=-1&sartist=-1&sstory={spec['story_id']}&limit={spec['limit']}"
    )


def fetch(client: requests.Session, url: str) -> str:
    response = client.get(url, timeout=45)
    response.raise_for_status()
    return response.text


def parse_listing(html: str, spec: dict[str, object], source_url: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.category")
    if table is None:
        raise RuntimeError(f"Listing table not found for {spec['story_label']}")
    rows = table.select("tr")
    output: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        link = row.select_one('a[href*="/all-uploads/"]')
        if link is None:
            continue
        metadata = rows[index + 1].select("td") if index + 1 < len(rows) else []
        metadata_text = [cell.get_text(" ", strip=True) for cell in metadata]
        if len(metadata_text) < 4:
            metadata_text += [""] * (4 - len(metadata_text))
        title = link.get_text(" ", strip=True)
        output.append({
            "story_id": str(spec["story_id"]),
            "story_label_expected": str(spec["story_label"]),
            "listing_index": str(len(output) + 1),
            "title_listing": title,
            "title_key_listing": key(title),
            "main_musician_listing": metadata_text[0],
            "supporting_musician_listing": metadata_text[1],
            "story_listing": metadata_text[2],
            "ragam_listing": metadata_text[3],
            "detail_url": urljoin(BASE, link.get("href", "")),
            "source_listing_url": source_url,
        })
    expected = int(spec["limit"])
    # The site's story-count badges include a small number of unpublished or
    # compilation rows that are not rendered in the listing table.
    if len(output) < expected - 3:
        raise RuntimeError(f"Expected approximately {expected} rows for {spec['story_label']}, parsed {len(output)}")
    return output


def label_value(article: str, label: str, next_labels: list[str]) -> str:
    escaped = "|".join(re.escape(item) for item in next_labels)
    match = re.search(rf"{re.escape(label)}\s*:\s*(.*?)(?=\n(?:{escaped})\s*:|\Z)", article, flags=re.I | re.S)
    if not match:
        return ""
    value = re.sub(r"\s+", " ", match.group(1)).strip(" &\xa0")
    return value


def parse_detail(html: str, row: dict[str, str]) -> dict[str, str]:
    soup = BeautifulSoup(html, "lxml")
    article_node = soup.select_one("article")
    article = article_node.get_text("\n", strip=True) if article_node else ""
    audio_candidates = []
    for node in soup.select("audio[src], source[src], a[href]"):
        candidate = node.get("src") or node.get("href") or ""
        if re.search(r"\.(?:mp3|wav|m4a|ogg|flac)(?:\?|$)", candidate, flags=re.I):
            audio_candidates.append(urljoin(BASE, candidate))
    artists = label_value(article, "ARTISTS", ["RAGAM", "THALAM"])
    artist_parts = [part.strip() for part in re.split(r"\s*&\s*", artists) if part.strip()]
    padam = label_value(article, "PADAM", ["STORY", "ARTISTS", "RAGAM", "THALAM"]).replace("&nbsp", "").strip()
    story = label_value(article, "STORY", ["ARTISTS", "RAGAM", "THALAM"])
    story = re.sub(r"\s*\(\s*If the long player.*$", "", story, flags=re.I).strip()
    ragam = label_value(article, "RAGAM", ["THALAM"])
    thalam_match = re.search(r"THALAM\s*:\s*([^\n]+)", article, flags=re.I)
    thalam = thalam_match.group(1).strip() if thalam_match else ""
    return {
        **row,
        "page_title": soup.title.get_text(" ", strip=True) if soup.title else "",
        "padam_detail": padam,
        "story_detail": story,
        "main_musician_detail": artist_parts[0] if artist_parts else "",
        "supporting_musician_detail": artist_parts[1] if len(artist_parts) > 1 else "",
        "artists_detail": artists,
        "ragam_detail": ragam,
        "thalam_detail": thalam,
        "audio_url": audio_candidates[0] if audio_candidates else "",
        "audio_candidate_count": str(len(set(audio_candidates))),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", action="store_true", help="Fetch every detail page after listing extraction")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    client = session()
    inventory: list[dict[str, str]] = []
    for spec in STORIES:
        url = listing_url(spec)
        parsed = parse_listing(fetch(client, url), spec, url)
        inventory.extend(parsed)
        repeated_rows = sum(count for count in Counter(row["title_key_listing"] for row in parsed).values() if count >= 2)
        print(f"{spec['story_label']}: {len(parsed)} listing rows; {repeated_rows} rows in exact repeated-title groups", flush=True)
    write_csv(INVENTORY_CSV, inventory)
    print(f"Wrote {INVENTORY_CSV} ({len(inventory)} rows)", flush=True)

    if args.details:
        detail_rows = []
        for index, row in enumerate(inventory, start=1):
            detail_rows.append(parse_detail(fetch(client, row["detail_url"]), row))
            if index % 20 == 0 or index == len(inventory):
                print(f"Detail pages {index}/{len(inventory)}", flush=True)
            time.sleep(0.08)
        DETAIL_JSON.write_text(json.dumps(detail_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {DETAIL_JSON}; audio URLs found {sum(bool(row['audio_url']) for row in detail_rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
