# Empty collection templates, not an acquired corpus

These CSV templates belong to the initial design draft and are not the input
schema of the integrated workflow. The authoritative inputs are
`cycle2/bundle_template.json`, `cycle2/seal_template.json`, `cycle2/protocol.json`
and `cycle2/rubric.md`. Do not use these draft CSVs to bypass that workflow's
independent-review and annotation requirements.

No contemporary documents or independent human annotations have been obtained.
These CSVs contain headers only. Do not fill them with inferred labels,
assistant-authored ratings, fabricated consent, or the old pilot articles.
All 24 pilot documents are now development-only and remain in their original
directory for auditability.

`corpus.csv` is for development text only: `train`, `tune`, `calibration`.
Every row needs a provenance cluster, pseudonymous source and author, topic,
genre, creation date, documented origin, text-specific reuse or consent evidence,
collection method, and PII review. Optional unknown metadata must say `unknown`.
Do not infer proficiency or protected traits. Generated records need model and
prompt-family IDs. Quality (`0`, `1`, `uncertain`) comes only from independent
annotation/adjudication. Mixed/unknown origins belong to separate challenges.

An origin attestation is evidence to audit, not a detector guess. A public URL
or permissive code license does not itself authorize releasing article text.
Origin/permission references can identify secure records without exposing them.

`annotations.csv` uses `primary` for two independent human raters and
`adjudicator` for a third human resolving any dimension/critical-error conflict.
Dimension scores: `0`, `1`, `2`, `uncertain`; critical error: `0`, `1`, `uncertain`.
Attestations: `true`/`false`. A model is not a human rater. Retain original ratings.
Use pseudonymous IDs, a rubric hash, rationale and evidence reference; record
`not_applicable` where justified rather than fabricate supporting evidence.

`custody.json` is deliberately unfilled. An independent human custodian must
retain final and challenge texts/labels separately and provide authentic hash
commitments and attestations. No final set has been constituted or bundled.
The workflow cannot establish identity, consent, independence or honesty from
Boolean fields. Independent human oversight is indispensable.