# Sampling, custody, and corpus specification: cycle2-v1

This is a proposed collection plan, not a description of an acquired corpus.
The targets are 3,200 development, 1,600 calibration, 1,200 primary final, and
400 challenge documents. No power claim follows from those targets. Admission
minima in `protocol.json` are necessary checks, not sufficient evidence of
representativeness or a guarantee of narrow subgroup intervals.

## Population and selection

The initial frame is permitted English public-web-style prose created between
January 1, 2024 and September 10, 2026. Record both publication/creation date
and collection date, with supporting provenance. The intended genres include
reported information, explanation/tutorial, argument/commentary, and discussion
or personal narrative. Topics should span at least six broad subject areas,
while `topic` identifiers capture finer subjects for disjoint evaluation.

Before collection, a human custodian must specify an actual eligible-source
roster, access rules, a random selection procedure within declared strata, and
replacement/exclusion rules. Archive the sampling manifest and extraction
failure counts. Without that roster and selection record, a collection can be
called curated or stratified but not representative of a defined population.
Balancing origin-by-quality cells is useful for discrimination experiments;
it destroys natural prevalence unless the selection probabilities are known
and properly accounted for. This package has no population prevalence sample.

Human-origin evidence must come from documented drafting histories, supervised
or consented authorship, or a comparably defensible provenance record. A
publication date alone does not establish contemporary human-only authorship.
Generated examples require actual model/version, prompt, settings where exposed,
date, provider terms/reuse evidence, and an independent generation record.
Match topics, intended audiences, tasks, and length ranges without deliberately
making one origin class more repetitive or more polished. Keep mixed authorship
and unknown provenance explicit rather than forcing labels.

Record applicable rights at the text level. A dataset repository's stated
license is not, by itself, an audit of all inherited source permissions. Do not
collect contact details or names when coarse pseudonymous grouping keys suffice.
Keep consent and identity records outside the released corpus under an
appropriate custodian. Release aggregate findings and approved artifacts only.

## Separation and challenge design

Construct leakage clusters before computing detector scores. Connect source,
human author, prompt/source-document lineage, and identified near duplicates;
every related record stays in one cluster. The code rejects sources or authors
assigned to multiple clusters, normalized duplicates without approved challenge
lineage, and cross-partition cluster/source/author overlap. Near-duplicate
discovery and the correctness of entity resolution still require a documented
preparation audit. Declaring arbitrary unique IDs does not establish independence.

Development and calibration must be separate. The primary final set is also
source-, author-, cluster-, and topic-disjoint from both. The independent
custodian reserves it before model development and controls access to its text
and labels. A content-hash commitment is supplied before final model freezing;
the bundle is released for a single final evaluation only after that freeze.
Do not describe the currently empty seal template as a reserved dataset.

Prepare the following independently sourced conditions, with at least 20
documents each as a minimum admission floor:

- `challenge_topic`: held-out topics, explicitly disjoint from development and calibration.
- `challenge_source`: unseen sources, with the same cluster/author separation.
- `challenge_generator`: documented generators absent from development/calibration.
- `challenge_format`: annotated formatting variants of new final parents; preserve
  token content and origin, record `parent_id`, and keep variants with their parent.
- `challenge_translation`: documented translator/model involvement, correct
  authorship-pattern labels, and independent quality review of the translated text.
- `challenge_mixed`: actual mixed authorship with `ai: null`; assess quality and
  coverage without pretending the binary authorship task has valid ground truth.
- `challenge_short`: texts below 80 tokenizer words; report abstention explicitly,
  not an accuracy computed only on longer cases.

All final/challenge conditions are evaluated together after one freeze. These
controls cannot support every possible distribution-shift claim; report their
actual composition. The required primary generator-family count is three.
Proficiency and formatting conditions need deliberate recruitment/collection,
not retrospectively inferred subgroup labels.

## Bundle format

Start with `bundle_template.json` and supply genuine records. The top-level
`role` is `development`, `calibration`, or `evaluation`. `sampling_frame` and
`selection_method` describe the actual frame and selection. `review` records
independent human review and a traceable evidence reference. Calibration also
requires an explicit review of independence assumptions.

Each element of `documents` contains:

```text
id, text, ai, cluster, source, topic, author_group, genre, proficiency,
formatting, generator_family, authorship_pattern, origin_evidence,
rights_basis, rights_evidence, privacy_review_evidence,
publication_date, collection_date, condition
```

Use `ai: 0` with `authorship_pattern: human`, `ai: 1` with
`authorship_pattern: fully_generated`, and JSON `null` for mixed/unknown origin.
Use `not_applicable` for a generator's human-author grouping field, and for the
generator-family field on human writing. Use `unknown` for uncollected
proficiency metadata. These placeholders are not acceptable substitutes for
missing rights or provenance evidence. Translation challenges additionally
require `translation_provenance`; formatting variants require `parent_id`.

`panel` identifies the same two independent human raters throughout the audit
batch. `annotators` contains pseudonymous IDs, `human: true`,
`independent_of_authors_and_generator: true`, and an `evidence_reference` for
each actual rater/adjudicator. False or unverified assertions must not be
converted to true simply to pass the software gate.

Each annotation contains:

```text
document_id, annotator_id, role, blinded, evidence_reference,
factual, information, coherence, central_error, uncertain
```

`role` is `independent` or `adjudication`; dimension scores are integers 0–2.
The final low-quality label is derived from these independent records using
the rubric, not read from an unsupported author-provided quality label.
Uncertain or disputed cases without a valid adjudication remain unresolved.

## Freeze and interpretation

`method_snapshot.json` records the local candidate specification's hashes.
It is not a human audit, an external preregistration, or a final model freeze.
The actual final freeze is created only after successful corpus admission,
fixed-model training, independent calibration, and receipt of a genuine
custodian seal. Changes after final access require a new protocol version and
a genuinely new final set.

For future prevalence work, independently verify sampling probabilities and
the joint screen's sensitivity/FPR in the target population. Propagate both
sampling and error-rate uncertainty; reject unstable corrections when the
screen is nearly uninformative. Annual estimates also require comparable frames,
composition, extraction, and contemporaneous error audits. No such estimates
are justified by the present curated pilot or the unfilled collection plan.