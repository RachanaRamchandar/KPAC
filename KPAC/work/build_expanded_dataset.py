from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
OLD_CSV = ROOT / "outputs" / "kathakalipadam_combined_stories_190.csv"
NEW_JSON = ROOT / "work" / "new_stories.json"
FALLBACK_JSON = ROOT / "work" / "new_story_listing_fallbacks.json"
OUT_CSV = ROOT / "outputs" / "kathakalipadam_canonicalized_expanded_425.csv"
AUDIT_CSV = ROOT / "outputs" / "kathakalipadam_canonicalization_audit.csv"
SELECTION_CSV = ROOT / "outputs" / "kathakalipadam_story_selection_audit.csv"
README = ROOT / "outputs" / "README_kathakalipadam_canonicalized_expanded.md"


def key(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold()
    value = re.sub(r"[^\w]+", "_", value, flags=re.UNICODE).strip("_")
    return re.sub(r"_+", "_", value)


def compact(value: str) -> str:
    return key(value).replace("_", "")


def clean_unknown(value: str) -> str:
    value = (value or "").strip()
    if key(value) in {"", "unknown"}:
        return ""
    return value


def bool_text(value: bool) -> str:
    return "1" if value else "0"


def display_from_key(value: str) -> str:
    if any("\u0d00" <= ch <= "\u0d7f" for ch in value):
        return value.replace("_", " ")
    return " ".join(part.capitalize() for part in value.split("_") if part)


# Curated only where the differences are orthographic/transliteration variants.
# Phrase extensions (for example anthike / anthike_vannitenam) are intentionally absent.
ALIASES: dict[str, dict[str, str]] = {
    "Duryodhana Vadham": {
        "paripahimam": "paripahi_mam",
        "paripahi_mam": "paripahi_mam",
        "dharmma_nandana_veeraa": "dharma_nandana_veera",
        "dharmanandana_veera": "dharma_nandana_veera",
        "njathi_vatsala": "jnathi_valsala",
        "jnathi_valsala": "jnathi_valsala",
    },
    "Nalacharitham Moonam Divasam": {
        "ghora_vipinam": "ghora_vipinam",
        "ghoravipinam": "ghora_vipinam",
        "khora_vipinam": "ghora_vipinam",
        "khoravipinam": "ghora_vipinam",
        "mariman_kanni": "mariman_kanni",
        "marimankanni": "mariman_kanni",
        "marimaan_kanni": "mariman_kanni",
        "marimaankanni": "mariman_kanni",
        "swaira_vachanam": "swaira_vachanam",
        "swairavachanam": "swaira_vachanam",
        "svairavachanam": "swaira_vachanam",
        "souravachanam": "swaira_vachanam",
        "manyamathe": "manyamathe",
        "maanyamathe": "manyamathe",
        "lokapalanmaare": "lokapalanmaare",
        "lokapalanmare": "lokapalanmaare",
        "lokapalakanmare": "lokapalanmaare",
        "loka_palakanmare": "lokapalanmaare",
        "lokanpalanmmare": "lokapalanmaare",
        "vijana_batha": "vijane_batha",
        "vijane_batha": "vijane_batha",
        "vijane_batha_mahathi": "vijane_batha_mahathi",
        "vijanee_batha_mahati": "vijane_batha_mahathi",
        "indu_mouli_harame": "indu_mouli_harame",
        "indu_moulee_harame": "indu_mouli_harame",
        "indumoulee_harame": "indu_mouli_harame",
        "indu_mouleeharame": "indu_mouli_harame",
        "nishadhendra": "naishadhendra",
        "naishadhendra": "naishadhendra",
        "kathunna_vanashikhi": "kathunna_vanashikhi",
        "kathunna_vanasikhi": "kathunna_vanashikhi",
        "varika_baahuka": "varika_baahuka",
        "varika_bahuka": "varika_baahuka",
        "chinthithamachiral": "chinthitham_achiraal",
        "chinthitham_achiraal": "chinthitham_achiraal",
        "chindithamachiral": "chinthitham_achiraal",
        "kadraveya_kulathilaka": "kadraveya_kulathilaka",
        "kadraveya_kula_thilaka": "kadraveya_kulathilaka",
        "kardraveyakulathilaka": "kadraveya_kulathilaka",
        "rithuparna_dharanipala": "rithuparna_dharanipala",
        "rithuparnna_dharanipala": "rithuparna_dharanipala",
        "rithuparna_dharaneepala": "rithuparna_dharanipala",
        "avalethoru_kamini": "aval_ethoru_kamini",
        "aval_ethoru_kamini": "aval_ethoru_kamini",
        "avalethoru_kamini_abhogi": "aval_ethoru_kamini_abhogi",
        "aval_ethoru_kamini_abhogi": "aval_ethoru_kamini_abhogi",
    },
    "Santhana Gopalam": {
        "ha_ha_karomi": "ha_ha_karomi",
        "haha_karomi": "ha_ha_karomi",
        "haa_haa_karomi": "ha_ha_karomi",
        "sreeman_sakhe": "sreeman_sakhe",
        "sreemansakhe": "sreeman_sakhe",
        "sreemal_sakhe": "sreeman_sakhe",
        "ma_kuru_sahasam": "ma_kuru_sahasam",
        "makuru_sahasam": "ma_kuru_sahasam",
        "maa_kuru_sahasam": "ma_kuru_sahasam",
        "maakuru_sahasam": "ma_kuru_sahasam",
        "nadha_bhaval": "nadha_bhaval",
        "nadhabhaval": "nadha_bhaval",
        "nadha_bhaval_charana": "nadha_bhaval_charana",
        "nadhabhavalcharana": "nadha_bhaval_charana",
        "nadhabhavalcharanaa": "nadha_bhaval_charana",
        "natha_bhavacharana": "nadha_bhaval_charana",
        "naathaa_bhavacharana": "nadha_bhaval_charana",
        "vidhimatham_nirasicheedamo": "vidhimatham_nirasicheedamo",
        "vidhimatham_nirasicheedaamo": "vidhimatham_nirasicheedamo",
        "vidhimatham_nirasichitamo": "vidhimatham_nirasicheedamo",
        "jayikka_jayikka_krishna": "jayikka_jayikka_krishna",
        "jayikka_jayikka_krishnaa": "jayikka_jayikka_krishna",
        "jeevitha_nayaka": "jeevitha_nayaka",
        "jeevitha_naayaka": "jeevitha_nayaka",
        "karuna_varidhe": "karuna_vaaridhe",
        "karuna_vaaridhe": "karuna_vaaridhe",
        "vidhimatham": "vidhimatham",
        "vidhmatham": "vidhimatham",
        "vidhikrutha": "vidhikrutha",
        "vidhikritha": "vidhikrutha",
        "namasthe_bhoosuramoule": "namasthe_bhoosuramoule",
        "namaste_bhoosuramoule": "namasthe_bhoosuramoule",
        "moodda_athi": "mooda_athi",
        "mooda_athi": "mooda_athi",
        "brahmanendra_koode": "brahmanendra_koode",
        "brahmanendrakoode": "brahmanendra_koode",
    },
    "Keechaka Vadham": {
        "kandivar_kuzhali": "kandivar_kuzhali",
        "kandivarkuzhali": "kandivar_kuzhali",
        "kandivaarkuzhali": "kandivar_kuzhali",
        "kantivar_kuzhali": "kandivar_kuzhali",
        "kandivar_kuzhalee": "kandivar_kuzhali",
        "kandivarkuzhalee": "kandivar_kuzhali",
        "kandivaarkuzhalee": "kandivar_kuzhali",
        "harinakshi": "harinakshi",
        "harinaakshi": "harinakshi",
        "harinakshee": "harinakshee",
        "harinaakshee": "harinakshee",
        "harinakshee_jana": "harinakshee_jana",
        "harinaksheejana": "harinakshee_jana",
        "harinaakshee_jana": "harinakshee_jana",
        "saadaram": "saadaram",
        "sadaram": "saadaram",
        "saadaram_nee_chonnoru": "saadaram_nee_chonnoru",
        "sadaram_nee_chonnoru": "saadaram_nee_chonnoru",
        "sodara": "sodara",
        "sodaraa": "sodara",
        "kantha": "kantha",
        "kanthaa": "kantha",
        "kantha_kripalo": "kantha_kripalo",
        "kaanthaa_kripalo": "kantha_kripalo",
        "kaanthaa_kripaalo": "kantha_kripalo",
        "kanthaa_krupalo": "kantha_kripalo",
        "sodaree_rajnhee": "sodaree_rajnhee",
        "sodaree_rajnee": "sodaree_rajnhee",
        "sodaree_raajnjee": "sodaree_rajnhee",
        "sodaraa_srunu": "sodaraa_srunu",
        "sodara_srunu": "sodaraa_srunu",
        "sodaraa_srinu": "sodaraa_srunu",
        "sasimukhi_varika": "sasimukhi_varika",
        "sasimukhee_varika": "sasimukhi_varika",
        "maninimar": "maninimar",
        "maaninimaar": "maninimar",
    },
}


DISPLAY = {
    (story, canonical): display_from_key(canonical)
    for story, mapping in ALIASES.items()
    for canonical in set(mapping.values())
}


def strip_ragam_suffix(title: str, ragam: str) -> tuple[str, bool]:
    if not title or not ragam:
        return title, False
    parts = re.split(r"\s*[-–—]\s*", title.strip())
    if len(parts) < 2:
        return title, False
    if compact(parts[-1]) == compact(ragam):
        return " - ".join(parts[:-1]).strip(), True
    return title, False


def canonicalize(story: str, title: str, ragam: str, scope: str) -> dict[str, str]:
    if scope != "padam":
        return {
            "padam_canonical_key": "",
            "padam_canonical_id": "",
            "padam_canonical_label": "",
            "canonicalization_method": "not_applicable_compilation",
            "canonicalization_confidence": "",
            "canonicalization_note": "Compilation retained for provenance but excluded from padam classification.",
        }
    stripped, removed = strip_ragam_suffix(title, ragam)
    base = key(stripped)
    canonical = ALIASES.get(story, {}).get(base, base)
    methods = []
    if removed:
        methods.append("ragam_suffix_removed")
    if canonical != base:
        methods.append("curated_orthographic_alias")
    elif base != key(title):
        methods.append("normalized_after_suffix_removal")
    else:
        methods.append("mechanical_normalization")
    confidence = "medium" if "curated_orthographic_alias" in methods else "high"
    note = (
        "Curated spelling/transliteration merge; raw title retained."
        if "curated_orthographic_alias" in methods
        else "No semantic merge; punctuation, case, and spacing normalized."
    )
    return {
        "padam_canonical_key": canonical,
        "padam_canonical_id": f"{key(story)}__{canonical}",
        "padam_canonical_label": DISPLAY.get((story, canonical), display_from_key(canonical)),
        "canonicalization_method": "+".join(methods),
        "canonicalization_confidence": confidence,
        "canonicalization_note": note,
    }


with OLD_CSV.open(encoding="utf-8", newline="") as handle:
    old_rows = list(csv.DictReader(handle))
with NEW_JSON.open(encoding="utf-8") as handle:
    scraped = json.load(handle)
with FALLBACK_JSON.open(encoding="utf-8") as handle:
    listing_fallbacks = {row["detail_url"]: row for row in json.load(handle)}

rows: list[dict[str, str]] = []
for row in old_rows:
    row = dict(row)
    row["missing_audio"] = bool_text(not bool(row.get("audio_url")))
    row.update(canonicalize(row["story_label"], row["padam_title_raw"], row["ragam_label"], row["recording_scope"]))
    if row["recording_scope"] == "padam":
        row["modeling_note"] = "Use padam_canonical_id as the target label; preserve recording-level splits."
    rows.append(row)

for offset, item in enumerate(scraped, start=len(rows) + 1):
    source_url = item["source_listing_url"]
    fallback = listing_fallbacks[item["detail_url"]]
    story = (item.get("story") or fallback.get("listing_story") or "").strip()
    if not story:
        story = "Santhana Gopalam" if "sstory=102" in source_url else "Keechaka Vadham"
    title = (item.get("padam") or item.get("title") or item.get("page_title") or "").strip()
    ragam_raw = (item.get("ragam") or fallback.get("listing_ragam") or "").strip()
    main_raw = (item.get("main") or "").strip()
    support_raw = (item.get("support") or "").strip()
    thalam_raw = (item.get("thalam") or "").strip()
    audio_url = (item.get("audio_url") or "").strip()
    title_key = key(title)
    scope = "scene_compilation" if "keechaka_vadham_scene" in title_key else "padam"
    filename = unquote(Path(urlparse(audio_url).path).name) if audio_url else ""
    audio_format = Path(filename).suffix.lower().lstrip(".") if filename else ""
    page_id_match = re.search(r"/all-uploads/(?:80-embeded-padams/)?([^/?#]+)", item["detail_url"])
    source_page_id = page_id_match.group(1).split("-")[0] if page_id_match else ""
    row = {
        "record_id": f"{offset:03d}",
        "source_page_record_id": source_page_id,
        "recording_scope": scope,
        "padam_title_raw": title,
        "padam_title_key": title_key,
        "main_musician_raw": main_raw,
        "main_musician": clean_unknown(main_raw),
        "supporting_musician_raw": support_raw,
        "supporting_musician": clean_unknown(support_raw),
        "story_raw": story,
        "story_label": story,
        "ragam_raw": ragam_raw,
        "ragam_label": clean_unknown(ragam_raw),
        "thalam_raw": thalam_raw,
        "thalam_label": clean_unknown(thalam_raw),
        "missing_supporting_musician": bool_text(not bool(clean_unknown(support_raw))),
        "missing_ragam": bool_text(not bool(clean_unknown(ragam_raw))),
        "missing_thalam": bool_text(not bool(clean_unknown(thalam_raw))),
        "missing_audio": bool_text(not bool(audio_url)),
        "audio_url": audio_url,
        "audio_filename": filename,
        "audio_format": audio_format,
        "detail_url": item["detail_url"],
        "source_listing_url": source_url,
        "source_retrieved_on": "2026-08-09",
        "split_group_id": f"recording_{offset:03d}",
        "modeling_note": (
            "Compilation retained for provenance; exclude from padam classification."
            if scope != "padam"
            else "Metadata retained but audio URL is missing; exclude until restored."
            if not audio_url
            else "Use padam_canonical_id as the target label; preserve recording-level splits."
        ),
    }
    row.update(canonicalize(story, title, ragam_raw, scope))
    rows.append(row)

fieldnames = [
    "record_id", "source_page_record_id", "recording_scope", "padam_title_raw", "padam_title_key",
    "padam_canonical_key", "padam_canonical_id", "padam_canonical_label",
    "canonicalization_method", "canonicalization_confidence", "canonicalization_note",
    "main_musician_raw", "main_musician", "supporting_musician_raw", "supporting_musician",
    "story_raw", "story_label", "ragam_raw", "ragam_label", "thalam_raw", "thalam_label",
    "missing_supporting_musician", "missing_ragam", "missing_thalam", "missing_audio",
    "audio_url", "audio_filename", "audio_format", "detail_url", "source_listing_url",
    "source_retrieved_on", "split_group_id", "modeling_note",
]
with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

groups: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in rows:
    if row["recording_scope"] == "padam":
        groups[row["padam_canonical_id"]].append(row)

audit_fields = [
    "padam_canonical_id", "story_label", "padam_canonical_key", "padam_canonical_label",
    "recording_count", "distinct_audio_url_count", "records_with_audio", "raw_title_keys",
    "raw_title_examples", "methods_used", "minimum_confidence", "eligible_for_3plus_classification",
]
confidence_order = {"": 0, "low": 1, "medium": 2, "high": 3}
audit_rows = []
for canonical_id, members in sorted(groups.items()):
    methods = sorted({m["canonicalization_method"] for m in members})
    min_conf = min((m["canonicalization_confidence"] for m in members), key=lambda x: confidence_order[x])
    urls = {m["audio_url"] for m in members if m["audio_url"]}
    audit_rows.append({
        "padam_canonical_id": canonical_id,
        "story_label": members[0]["story_label"],
        "padam_canonical_key": members[0]["padam_canonical_key"],
        "padam_canonical_label": members[0]["padam_canonical_label"],
        "recording_count": len(members),
        "distinct_audio_url_count": len(urls),
        "records_with_audio": sum(bool(m["audio_url"]) for m in members),
        "raw_title_keys": " | ".join(sorted({m["padam_title_key"] for m in members})),
        "raw_title_examples": " | ".join(sorted({m["padam_title_raw"] for m in members})),
        "methods_used": " | ".join(methods),
        "minimum_confidence": min_conf,
        "eligible_for_3plus_classification": bool_text(len(urls) >= 3),
    })
with AUDIT_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=audit_fields)
    writer.writeheader()
    writer.writerows(audit_rows)

candidate_metrics = [
    ("Santhana Gopalam", 131, 25, 86, 1, "Highest repeated-record coverage among audited candidates."),
    ("Keechaka Vadham", 104, 17, 61, 1, "Second-highest repeated-record coverage among audited candidates."),
    ("Nalacharitham Nalaam Divasam", 118, 19, 58, 0, "Strong reserve candidate; not added in this expansion."),
    ("Kuchela Vritham", 96, 17, 53, 0, "Reserve candidate; not added in this expansion."),
    ("Karna Shapadham", 97, 18, 45, 0, "Reserve candidate; not added in this expansion."),
    ("Nalacharitham Onnam Divasam", 137, 15, 42, 0, "High volume but lower exact-repeat coverage."),
    ("Nalacharitham Randam Divasam", 95, 14, 39, 0, "Lower exact-repeat coverage in this audit."),
]
with SELECTION_CSV.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["story_label", "rendered_listing_rows", "exact_repeated_title_classes", "rows_in_exact_repeated_title_classes", "selected", "selection_note"])
    writer.writerows(candidate_metrics)

story_counts = Counter(r["story_label"] for r in rows)
padam_rows = [r for r in rows if r["recording_scope"] == "padam"]
eligible_groups = [g for g in groups.values() if len({r["audio_url"] for r in g if r["audio_url"]}) >= 3]
eligible_records = sum(len({r["audio_url"] for r in g if r["audio_url"]}) for g in eligible_groups)
alias_changed = sum("curated_orthographic_alias" in r["canonicalization_method"] for r in padam_rows)
missing_audio = sum(not r["audio_url"] for r in rows)

README.write_text(f"""# Kathakalipadam canonicalized expanded dataset

## Deliverables

- `kathakalipadam_canonicalized_expanded_425.csv`: analysis table with raw labels, canonical labels, provenance, and audio links.
- `kathakalipadam_canonicalization_audit.csv`: one row per canonical padam, showing every merged raw title and recording counts.
- `kathakalipadam_story_selection_audit.csv`: comparison used to select the two added stories.

## Coverage

- Total archive rows: {len(rows)}
- Individual padam rows: {len(padam_rows)}
- Non-padam compilations: {len(rows) - len(padam_rows)}
- Story counts: {dict(story_counts)}
- Rows without an exposed audio URL: {missing_audio}
- Rows changed by a curated orthographic/transliteration alias: {alias_changed}
- Canonical padams with at least three distinct exposed audio URLs: {len(eligible_groups)}
- Distinct audio recordings inside those 3+ classes: {eligible_records}

## Canonicalization policy

The source spelling is never overwritten: `padam_title_raw` and `padam_title_key` remain available. The modeling label is `padam_canonical_id`, which is story-qualified so similarly named padams in different stories are not merged. `padam_canonical_key` merges only documented punctuation, spacing, and conservative transliteration variants. For example, `ghora_vipinam` and `ghoravipinam` map to the same canonical padam. Phrase extensions such as `anthike` versus `anthike_vannitenam` remain separate.

Each row records the method, confidence, and note used for canonicalization. Medium-confidence curated merges should be checked by a Kathakali domain expert before publication.

## Story selection

Seven high-volume story listings were profiled. Santhana Gopalam and Keechaka Vadham were selected because they contributed the largest numbers of records already belonging to repeated exact-title groups (86 and 61 respectively). This favors repeated, independent performances rather than additional singleton labels.

## Modeling use

Use only `recording_scope == padam` and non-empty `audio_url` for audio classification. Use `padam_canonical_id` as the target. Keep `split_group_id` at recording level and use artist-aware or performance-aware grouped evaluation where metadata allows it. The canonicalization audit is the authoritative record of label merges.
""", encoding="utf-8")

print(json.dumps({
    "rows": len(rows),
    "padam_rows": len(padam_rows),
    "story_counts": dict(story_counts),
    "missing_audio": missing_audio,
    "curated_alias_rows": alias_changed,
    "canonical_classes": len(groups),
    "eligible_3plus_classes": len(eligible_groups),
    "eligible_distinct_audio_records": eligible_records,
}, ensure_ascii=False, indent=2))
