# Second cycle: development evidence and gated evaluation

**The confirmatory study is not complete.** No qualifying contemporary corpus,
independent human ratings, or independently held final/challenge set has been
obtained. All 24 historical pilot documents, including the former test articles,
are now development-only. No missing final accuracy, agreement, or prevalence
estimate has been invented.

## Authoritative methodology and schemas

Use `protocol.json`, `rubric.md`, `bundle_template.json` and `seal_template.json`
with `cycle2_workflow.py` and `cycle2_stats.py`. These are the authoritative
prospective workflow and input formats. The earlier CSV templates in `data/`,
`rubric.json` and `sampling.json` are retained as explicitly superseded drafts;
their targets and schemas must not be mixed with the current protocol.

`exploratory_plan.json` is a separate fixed schedule for the supplementary
repeated-split diagnostics. It is not an alternative final-evaluation protocol.
`method_snapshot.json` records current canonical and diagnostic file hashes.
This is a local methodological commitment, not external preregistration or an
independent human audit. A trained-model freeze is a later, separate artifact.

The authoritative protocol fixes the original ten-feature classifier at L2
penalty 0.1 for confirmatory comparison. Exploratory ablations do not select a
new claimed winner. Development and calibration bundles must pass independent
annotation/provenance/rights/privacy admission and be source/author/cluster
disjoint. Calibration uses one negative representative per cluster, chosen by
document ID before looking at scores. The protocol requires at least 120 such
negative clusters per head. The Neyman–Pearson order statistic controls a
5% target FPR with 2.5% violation tolerance per head **only under independent,
exchangeable calibration and future negatives**. Its theoretical minimum is
72, but that is not the protocol's recruitment minimum. It does not guarantee
joint-screen FPR or performance under distribution shift.

The rubric requires two fixed independent human raters, blinded to authorship
and model output, with a distinct human adjudicator. Admission uses at least
100 paired documents, raw agreement >=0.80, a lower cluster-bootstrap kappa
bound >=0.60, and at least 95% quality resolution. These are design choices,
not universal validity criteria. The code computes agreement on initial
judgments, not consensus; constant-label kappa remains undefined. Review
attestations are necessary records, not proof of real consent or independence.

Current collection targets, final-stratum requirements and challenge conditions
are specified in `protocol.json`, not reported as achieved sample sizes or a
completed power analysis. A curated balanced corpus is not an internet probability
sample. A data custodian must keep final texts and labels away from developers
and supply a hash commitment before a model/threshold freeze. No such final set
exists here. All final evaluations remain pending.

## Executed development analyses

Two complementary exploratory analyses are retained; their resampling schemes
must not be conflated or selected according to which result looks better.

`cycle2_diagnostics.py` performs leave-one-topic-pair-out assessment with separate
within-fold validation, refitting each model and selecting its threshold without
the assessment pair. Under this scheme, the original configuration has FNR
0.25 and the pilot-selected configuration has FNR about 0.333; both have FPR 0
on the reused pilot. Their mean within-fold AUROC is 1.0. Scores from different
fitted models are not pooled to calculate a global AUROC. A 99-permutation
within-pair control gives a Monte Carlo value of 0.01 for the mean-fold statistic.
It tests association under that randomization, not causal identification of
authorship rather than source or era. Detailed outputs are in `artifacts/`.

`research_cycle.py diagnose` performs 100 seed-fixed 6/3/3 topic-pair partitions,
with five feature/regularization configurations and three cutoff policies. This
produces 500 fits and 1,500 assessment rows, plus 100 within-pair permutation
fits using the original split. These are **600 fits on the same 24 documents**,
not 600 independent experiments. This second permutation statistic gives a
tail fraction about 0.099 because it uses a different, much smaller assessment
statistic. It is not interchangeable with the mean-fold test above.

For the supplementary partitions, the selected eight-feature configuration
with the legacy cutoff has mean FNR 0.240 and FPR 0.000. A negative-maximum
cutoff reduces mean FNR to 0.000 but raises mean FPR to 0.260. The midpoint
comparison gives mean FNR about 0.007 and FPR 0.040. These are exploratory
averages over reused data, not deployment estimates or a basis for replacing
the fixed confirmatory configuration after inspecting final outcomes.

Length-only features have median assessment AUROC 1.0 in those tiny supplementary
assessment sets, despite legacy-threshold mean FNR about 0.657. The relevant
lesson is that a nuisance-feature subset can explain much of the ranking in
this corpus—not that length reliably identifies AI writing.

Both analyses reproduce the original six-document failure exactly. Retrospective
alternative thresholds are retained as diagnostics, never substituted for the
original results. For the selected model, the negative-maximum rule flags all
three generated former-test articles but also one historical article; a midpoint
separates those six. This is not a new held-out success.

Whitespace and uppercase probes leave the selected model's 24 scores unchanged.
Adding a hyphen before each sentence flips four generated-positive decisions
to unflagged; the largest absolute score change is about 0.01849. Replacing
digit sequences with `number` changes some scores but no flags. The latter
changes content and is not a meaning-preserving challenge. No independently
annotated translation, proficiency, or mixed-authorship challenge was run.

The immediate failure mechanism is reproduced: the selected operating point
does not transfer to the assessment score range. The underlying causes cannot
be separated. Historical era, register, publication source and the generating
assistant are confounded with origin. Source-disjoint accuracy is explicitly
`not_identifiable`; no amount of re-splitting creates missing sources.

## Uncertainty and unavailable results

The 100-partition percentile ranges describe split sensitivity conditional on
the 24 documents, not population confidence intervals. The leave-pair-out
reports additionally include topic-cluster bootstrap intervals conditional on
the fitted predictions; these exclude training uncertainty. A degenerate
zero-error bootstrap is not evidence of zero population risk. Exact binomial
intervals are labeled as requiring independence, which shared sources violate.
Do not count repeatedly assessed documents as independent observations.

The actual independent-annotation audit in `artifacts/annotation_audit.json`
reports no data and null agreement. No final model, final results or independent
challenge results have been produced. Old files under `results/` record the
earlier CSV-template readiness audit; they are not a second corpus or a passed
annotation review. `source_audit.md` documents reviewed resources and unsuccessful
bulk-access probes. Documentation access did not produce a qualified dataset.

No prevalence or annual trend is estimated. Those require a separate probability
sample, real inclusion probabilities, transportable independently audited
joint-screen error rates, and uncertainty propagation. The balanced historical
pilot cannot supply these inputs.

## Reproduction and prospective workflow

Run from the workspace root:

```bash
python cycle2_diagnostics.py --output cycle2/artifacts
python research_cycle.py diagnose
python research_cycle.py snapshot
python cycle2_workflow.py audit cycle2/bundle_template.json cycle2/artifacts/annotation_audit.json
```

The audit of the empty template **intentionally exits with code 2** and reports
missing prerequisites. It is not a failed experiment with a zero score. A
snapshot may not overwrite a differing methodological commitment; use a
versioned amendment when methods change.

Once genuine admitted data and an independent custodian exist, the implemented
workflow supports:

```bash
python cycle2_workflow.py fit development.json calibration.json cycle2/candidate
python cycle2_workflow.py freeze cycle2/candidate custodian_seal.json
python cycle2_workflow.py evaluate cycle2/candidate evaluation.json
```

These input files are **not bundled**. Fitting records development/calibration
audits and hashes. Freezing requires a genuine custody record. Evaluation checks
the commitment, unchanged methods/model/data, source/author separation, required
final cells, and challenge coverage. It reports FPR, FNR, other metrics,
label/length coverage, condition/subgroup results and uncertainty without changing
thresholds. Unknown/mixed origins do not become human labels. Actual human
verification remains indispensable even when a supplied JSON passes checks.

## What is needed to finish

A responsible investigator must arrange permitted contemporary collection and
generation by independent operators, two independent human raters and an
adjudicator, and separate final-data custody. Topic/source/generator/proficiency
coverage and actual rights evidence must be audited. Only then can the fitted
model and thresholds be frozen and a new final evaluation run. The existing
pilot cannot be recycled as that final set.

The system is for manual-review research only: no automatic removal, misconduct
accusations or public labeling. Release aggregates and permitted data, not raw
consent records or unnecessary identifiers. An AI assistant prepared and revised
the study materials, authored the pilot's generated articles, and implemented
diagnostics. It did not supply independent human ratings or independent
confirmatory examples. Publication suitability depends on the eventual evidence
and review, not a promised outcome.