#!/usr/bin/env python3
"""Build the repeat-rich seven-story Kathakalipadam corpus expansion."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
WORK = ROOT / "work" / "corpus_expansion"
OLD_CSV = OUT / "kathakalipadam_canonicalized_expanded_425.csv"
RAW_JSON = WORK / "story_details_raw.json"
FINAL_CSV = OUT / "kathakalipadam_repeat_rich_expanded.csv"
CANONICAL_AUDIT = OUT / "kathakalipadam_repeat_rich_canonicalization_audit.csv"
STORY_AUDIT = OUT / "kathakalipadam_expansion_story_audit.csv"
EXCLUSIONS = OUT / "kathakalipadam_expansion_exclusions.csv"
README = OUT / "README_kathakalipadam_repeat_rich_expansion.md"
MIN_NEW_CLASS_RECORDINGS = 3


def key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return re.sub(r"_+", "_", value)


def compact(value: str) -> str:
    return key(value).replace("_", "")


def clean_unknown(value: str) -> str:
    value = (value or "").strip()
    return "" if key(value) in {"", "unknown", "none"} else value


def bool_text(value: bool) -> str:
    return "1" if value else "0"


def display(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split("_") if part)


def strip_ragam_suffix(title: str, ragam: str) -> tuple[str, bool]:
    if not title or not ragam:
        return title, False
    parts = re.split(r"\s*[-–—]\s*", title.strip())
    if len(parts) > 1 and compact(parts[-1]) == compact(ragam):
        return " - ".join(parts[:-1]).strip(), True
    return title, False


# Conservative, story-local orthographic/transliteration aliases. Phrase
# extensions remain separate even when they are musically related.
ALIASES: dict[str, dict[str, str]] = {
    "Nalacharitham Nalaam Divasam": {
        "aaa_kadha_kettu": "akkadha_kettu",
        "akkadha_ketto": "akkadha_kettu",
        "akkatha_ketto": "akkadha_kettu",
        "dharanipanmaar": "dharanipanmar",
        "dheranipanmaar": "dharanipanmar",
        "eeryathe_ellaam": "eeryathe_ellam",
        "eryathe_ellam": "eeryathe_ellam",
        "kulavadhoonaam": "kulavadhoonam",
        "kulavadhunam": "kulavadhoonam",
        "mandire_chennaal": "mandire_chennal",
        "naishadhan_ivan_thaan": "naishadhan_ivan_than",
        "nishadhan_ivan_than": "naishadhan_ivan_than",
        "nishandhan_ivan_than": "naishadhan_ivan_than",
        "olivilundo": "olivil_undo",
        "parnadan_sakethatthil": "parnadan_sakethathil",
        "parnadan_sakathathil": "parnadan_sakethathil",
        "parnnadan_sakethaththil": "parnadan_sakethathil",
        "poomaathinoththa": "poomathinotha",
        "poomathinoththa": "poomathinotha",
        "poomathinottha": "poomathinotha",
        "svalpa_punyayayen": "swalpa_punyayayen",
        "svalpapunyayaye": "swalpa_punyayayen",
        "swalpapunyayayen": "swalpa_punyayayen",
        "swlapa_punyayayen": "swalpa_punyayayen",
        "theernu_sandeham_ellam": "theernu_sandehamellam",
        "theernnu_sandehamellaam": "theernu_sandehamellam",
        "theernnu_santheham_ellam": "theernu_sandehamellam",
        "abhilasham_kondu_thanne": "abhilasham_kontu_thanne",
        "premanuragineee_njan": "premanuragini_njan",
    },
    "Kuchela Vritham": {
        "ajithahare": "ajitha_hare",
        "ajithahare_jaya": "ajitha_hare_jaya",
        "ajithaa_hare_jaya": "ajitha_hare_jaya",
        "daanavaari": "danavari",
        "danavaari": "danavari",
        "enthoru_chitramidam": "enthoru_chithramidam",
        "enthoru_chithramitham": "enthoru_chithramidam",
        "indhana_sambaadana": "indhana_sambadana",
        "inthanasampadanam": "indhana_sambadana",
        "kalayaami": "kalayami",
        "kalayamisumathe": "kalayami_sumathe",
        "kalyanaalayaaa": "kalyanalaya",
        "karunyanidhe_kantha": "karunya_nidhe_kantha",
        "manjulamgee": "manjulangi",
        "pushkaravilochana": "pushkara_vilochana",
        "pushkaravilochanaa": "pushkara_vilochana",
    },
    "Karna Shapadham": {
        "aaryaputra_kelkka": "aryaputhra_kelkka",
        "aryaputhra_kelka": "aryaputhra_kelkka",
        "aryaputra_kelkka": "aryaputhra_kelkka",
        "eute_ponnomane": "ennute_ponnomane",
        "enthiha_man_manase": "enthiha_manmanase",
        "hantha_daivme": "hantha_daivame",
        "kaathara_vilochane": "kathara_vilochane",
        "kalina_kai_thozhitheetunnen": "kalina_kaithozhutheetunnen",
        "karnnaa_dayalo": "karnna_dayalo",
        "karna_mathi_yidam": "karnna_mathiyidam",
        "karnna_mathi_idam": "karnna_mathiyidam",
        "karnnaa_mathiyidam": "karnna_mathiyidam",
        "kadhayellam_arivayi": "kathayellam_arivayi",
        "katha_ellam_arivayi": "kathayellam_arivayi",
        "kathayellaam_arivaayi": "kathayellam_arivayi",
        "kathayellam_arivaayi": "kathayellam_arivayi",
        "pranasakaha_ninnutaya": "pranasakha_ninnutaya",
        "sranagathare": "sharanaagathare",
        "unnee_ennute": "unne_ennute",
        "valsalyavaridhe": "valsalya_varidhe",
        "vathsalya_varidhe": "valsalya_varidhe",
        "valsalyavaridhe": "valsalya_varidhe",
        "valsalya_varidhe_karna": "vatsalya_varidhe_karna",
        "vatsalya_vaaridhe_karnaa": "vatsalya_varidhe_karna",
    },
}


def canonicalize(story: str, title: str, ragam: str) -> dict[str, str]:
    stripped, removed = strip_ragam_suffix(title, ragam)
    base = key(stripped)
    canonical = ALIASES.get(story, {}).get(base, base)
    methods = []
    if removed:
        methods.append("ragam_suffix_removed")
    if canonical != base:
        methods.append("curated_orthographic_alias")
    if not methods:
        methods.append("mechanical_normalization")
    return {
        "padam_title_key": key(title),
        "padam_canonical_key": canonical,
        "padam_canonical_id": f"{key(story)}__{canonical}",
        "padam_canonical_label": display(canonical),
        "canonicalization_method": "+".join(methods),
        "canonicalization_confidence": "medium" if canonical != base else "high",
        "canonicalization_note": (
            "Curated story-local spelling/transliteration merge; raw title retained."
            if canonical != base
            else "No semantic merge; punctuation, case, and spacing normalized."
        ),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with OLD_CSV.open(encoding="utf-8", newline="") as handle:
        existing = list(csv.DictReader(handle))
        base_fields = list(existing[0])
    raw = json.loads(RAW_JSON.read_text(encoding="utf-8"))

    candidate_rows = []
    exclusions = []
    for index, item in enumerate(raw, start=len(existing) + 1):
        story = item["story_label_expected"].strip()
        title = item["title_listing"].strip()
        ragam = clean_unknown(item.get("ragam_listing", ""))
        scope = "full_story_compilation" if compact(title) == compact(story) else "padam"
        audio_url = item.get("audio_url", "").strip()
        canonical = canonicalize(story, title, ragam)
        filename = unquote(Path(urlparse(audio_url).path).name) if audio_url else ""
        page_id = re.search(r"/all-uploads/(?:80-embeded-padams/)?(\d+)", item["detail_url"])
        row = {
            "record_id": f"{index:03d}",
            "source_page_record_id": page_id.group(1) if page_id else "",
            "recording_scope": scope,
            "padam_title_raw": title,
            **canonical,
            "main_musician_raw": item.get("main_musician_listing", "").strip(),
            "main_musician": clean_unknown(item.get("main_musician_listing", "")),
            "supporting_musician_raw": item.get("supporting_musician_listing", "").strip(),
            "supporting_musician": clean_unknown(item.get("supporting_musician_listing", "")),
            "story_raw": story,
            "story_label": story,
            "ragam_raw": item.get("ragam_listing", "").strip(),
            "ragam_label": ragam,
            "thalam_raw": item.get("thalam_detail", "").strip(),
            "thalam_label": clean_unknown(item.get("thalam_detail", "")),
            "missing_supporting_musician": bool_text(not bool(clean_unknown(item.get("supporting_musician_listing", "")))),
            "missing_ragam": bool_text(not bool(ragam)),
            "missing_thalam": bool_text(not bool(clean_unknown(item.get("thalam_detail", "")))),
            "missing_audio": bool_text(not bool(audio_url)),
            "audio_url": audio_url,
            "audio_filename": filename,
            "audio_format": Path(filename).suffix.lower().lstrip("."),
            "detail_url": item["detail_url"],
            "source_listing_url": item["source_listing_url"],
            "source_retrieved_on": "2026-08-09",
            "split_group_id": f"recording_{index:03d}",
            "modeling_note": "Candidate from repeat-rich story; retained only if canonical class has at least three distinct exposed audio URLs.",
            "dataset_iteration": "repeat_rich_story_expansion",
            "source_story_selected_for_repeat_coverage": "1",
            "new_story_class_audio_count": "",
        }
        if scope != "padam":
            exclusions.append({"record_id": row["record_id"], "detail_url": row["detail_url"], "reason": "full-story compilation"})
        elif not audio_url:
            exclusions.append({"record_id": row["record_id"], "detail_url": row["detail_url"], "reason": "no exposed audio URL"})
        else:
            candidate_rows.append(row)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        by_class[row["padam_canonical_id"]].append(row)
    distinct_url_counts = {
        canonical_id: len({row["audio_url"] for row in members})
        for canonical_id, members in by_class.items()
    }

    selected_new = []
    seen_class_url = set()
    for row in candidate_rows:
        count = distinct_url_counts[row["padam_canonical_id"]]
        row["new_story_class_audio_count"] = str(count)
        pair = (row["padam_canonical_id"], row["audio_url"])
        if count < MIN_NEW_CLASS_RECORDINGS:
            exclusions.append({"record_id": row["record_id"], "detail_url": row["detail_url"], "reason": f"new canonical class has only {count} distinct audio URL(s)"})
        elif pair in seen_class_url:
            exclusions.append({"record_id": row["record_id"], "detail_url": row["detail_url"], "reason": "duplicate audio URL within canonical class"})
        else:
            seen_class_url.add(pair)
            selected_new.append(row)

    extra_fields = ["dataset_iteration", "source_story_selected_for_repeat_coverage", "new_story_class_audio_count"]
    final_fields = base_fields + extra_fields
    final_rows = []
    for row in existing:
        enriched = dict(row)
        enriched.update({
            "dataset_iteration": "existing_canonical_corpus",
            "source_story_selected_for_repeat_coverage": "0",
            "new_story_class_audio_count": "",
        })
        final_rows.append(enriched)
    final_rows.extend(selected_new)
    write_csv(FINAL_CSV, final_rows, final_fields)

    audit_rows = []
    for canonical_id, members in sorted(by_class.items()):
        count = distinct_url_counts[canonical_id]
        audit_rows.append({
            "padam_canonical_id": canonical_id,
            "story_label": members[0]["story_label"],
            "padam_canonical_label": members[0]["padam_canonical_label"],
            "distinct_audio_url_count": count,
            "distinct_main_musicians": len({row["main_musician"] for row in members if row["main_musician"]}),
            "raw_title_keys": " | ".join(sorted({row["padam_title_key"] for row in members})),
            "methods_used": " | ".join(sorted({row["canonicalization_method"] for row in members})),
            "selected_repeat_rich_class": bool_text(count >= MIN_NEW_CLASS_RECORDINGS),
        })
    write_csv(CANONICAL_AUDIT, audit_rows)

    story_rows = []
    for story in sorted({row["story_label"] for row in candidate_rows}):
        raw_story = [item for item in raw if item["story_label_expected"] == story]
        selected_story = [row for row in selected_new if row["story_label"] == story]
        story_rows.append({
            "story_label": story,
            "rendered_listing_rows": len(raw_story),
            "rows_with_exposed_audio": sum(bool(item.get("audio_url")) for item in raw_story),
            "selected_repeat_rich_recordings": len(selected_story),
            "selected_canonical_classes": len({row["padam_canonical_id"] for row in selected_story}),
            "classes_with_at_least_8_recordings": sum(
                1 for canonical_id, count in distinct_url_counts.items()
                if canonical_id.startswith(f"{key(story)}__") and count >= 8
            ),
            "source_listing_url": raw_story[0]["source_listing_url"],
        })
    write_csv(STORY_AUDIT, story_rows)
    write_csv(EXCLUSIONS, sorted(exclusions, key=lambda row: row["record_id"]))

    class_distribution = Counter(distinct_url_counts.values())
    README.write_text(
        "# Repeat-rich Kathakalipadam corpus expansion\n\n"
        f"The previous 425-row corpus is retained. Three reserve stories were scraped in full, but only new canonical classes with at least {MIN_NEW_CLASS_RECORDINGS} distinct exposed audio URLs were admitted. New singleton and two-recording padams remain in the exclusion audit rather than the modeling CSV.\n\n"
        f"- Previous rows retained: {len(existing)}\n"
        f"- New story listing rows audited: {len(raw)}\n"
        f"- New repeat-rich recordings admitted: {len(selected_new)}\n"
        f"- New repeat-rich canonical classes admitted: {len({row['padam_canonical_id'] for row in selected_new})}\n"
        f"- Final CSV rows: {len(final_rows)}\n"
        f"- New-story distinct-audio class-size distribution: {dict(sorted(class_distribution.items()))}\n\n"
        "Canonicalization is story-local and conservative. It merges punctuation, spacing, and documented transliteration variants, while phrase extensions remain distinct. Audio-byte duplicate validation is performed in the subsequent audio validation stage.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "existing_rows": len(existing),
        "raw_new_rows": len(raw),
        "new_selected_recordings": len(selected_new),
        "new_selected_classes": len({row["padam_canonical_id"] for row in selected_new}),
        "final_rows": len(final_rows),
        "new_class_size_distribution": dict(sorted(class_distribution.items())),
        "story_audit": story_rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
