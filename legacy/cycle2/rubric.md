# Quality rubric: cycle2-v1

Version date: September 10, 2026. This fixes the proposed rules before a new
final evaluation. It has not yet passed an independent human annotation pilot.
Any revision requires a new version and cannot be applied retrospectively to
obtain preferred test results.

## Independent procedure

Use two fixed, identified-by-pseudonym human raters across an audit batch.
They work independently, cannot be the text authors, and are blinded to origin,
generation metadata, detector scores, split role, and each other's ratings.
A distinct human adjudicator resolves disagreements. Record independent ratings
before adjudication; agreement is computed on those ratings, never on consensus.
Do not use this assistant, another automated judge, or multiple invocations of
one model as independent human annotators.

Provide the task and genre where needed to judge relevance, and a permitted
evidence packet for factual checking. Keep identity/provenance records with a
separate custodian. The rater registry needs an independence attestation and
an evidence reference, not names, contact details, or demographic identifiers.
The software checks record consistency, not whether an asserted human review
really happened; a responsible investigator must verify that independently.

## Dimensions

Score each dimension 0 (no material defect), 1 (limited defect), or 2
(substantial defect):

- **Factual and evidential adequacy:** checkable central claims contradict
  evidence, or essential asserted support cannot be substantiated. Preserve the
  evidence reference and passage; uncertainty about a claim is not proof it is false.
- **Information value:** substantial redundancy or generic padding leaves the
  stated question unanswered or adds little relevant information. Repeated
  technical terminology and necessary restatement are not automatically defects.
- **Relevance and coherence:** major contradictions, irrelevant passages, or
  failure to serve the document's stated purpose. Language proficiency and
  stylistic preference are not substitutes for this judgment.

Set `central_error` only when a verified central factual error defeats the
document's purpose. Fiction, personal narrative, and opinion require appropriate
genre guidance; lack of academic citations alone is not a quality failure.

The binary low-quality label is 1 if at least two dimensions score 2, or if
`central_error` is true. Otherwise it is 0. Mark the entire decision uncertain
when evidence is inadequate; do not force a binary answer. An adjudicator may
also leave a case uncertain. Unresolved cases remain in the coverage report
but are excluded from binary training and label-dependent metrics.

## Audit and acceptance

Audit at least 100 documents with both low-quality classes represented. Report
raw binary agreement, Cohen's kappa for the fixed rater pair, dimension-level
quadratically weighted kappa, uncertainty frequency, and adjudication frequency.
Bootstrap leakage clusters 2,000 times for the binary kappa interval; report the
number of defined replicates. Constant-label kappa is undefined, not 1.

The proposed admission criteria are raw binary agreement at least 0.80, lower
95% cluster-bootstrap kappa bound at least 0.60, and at least 95% of documents
resolved. These are design choices, not universal quality standards. Failure
requires reporting and a new annotation-development cycle, not removing hard
cases silently or lowering the criteria after seeing final performance.

The present project has no ratings meeting this procedure. Its actual audit
statistics are therefore unavailable. External ratings on proficiency or
helpfulness cannot be renamed as this rubric's labels.