# Prospective practitioner canonical-label adjudication protocol

This protocol addresses the current limitation that KPAC canonical labels were not independently adjudicated by multiple Kathakali practitioners. It must not be cited as evidence that such adjudication has already happened.

## Practitioner panel

Recruit at least two independent practitioners with documented familiarity with Kathakali music, texts, or performance practice. Record expertise and conflicts without publishing unnecessary personal data.

## Independent adjudication packet

For each proposed canonical group, provide the story, raw catalogue title, normalized form, source-page context, available text or metadata evidence, and a stable anonymous item identifier. Do not expose another practitioner’s judgment during the first pass.

Each practitioner selects one disposition:

1. same padam and safe to merge;
2. distinct padam and must remain separate;
3. uncertain—needs listening, text, or performance context;
4. catalogue/source error;
5. outside the practitioner’s expertise.

Practitioners also provide a short rationale and confidence (`high`, `medium`, or `low`).

## Adjudication

Agreement is accepted provisionally. Disagreements go to a joint adjudication meeting or a third practitioner. Never overwrite the v1 label silently. Store the old value, proposed value, practitioners’ dispositions, rationale, evidence, adjudicator, date, and release version.

## Evaluation consequences

If a decision changes class identity or membership in the locked benchmark, release a new benchmark version and do not replace the v1 result. Report performance on both locks when making longitudinal comparisons. Do not use current outer-test errors as evidence for changing labels without independent musical justification.

## Suggested adjudication table

`adjudication_item_id, story_label, record_id, raw_title, current_canonical_key, proposed_canonical_key, practitioner_id, disposition, confidence, rationale, evidence_uri, adjudication_date, adjudication_status, release_version`
