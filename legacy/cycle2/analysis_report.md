# Exploratory second-cycle findings

All 24 original pilot articles are now development material, including the six
previously called test articles. No result below is a new untouched-test result.
The original model files and historical test report were not modified.

## Reproduction and the immediate failure mechanism

Scoring the original six articles with the saved models reproduces their
recorded confusion counts: three true negatives, three false negatives,
zero false positives, and zero true positives. Both models rank generated
articles above historical articles but place all of them below their cutoffs.

The original threshold rule prefers the highest cutoff among settings with
the same validation recall and FPR. On these separated validation scores,
that selects the lowest generated validation score—the upper end of a large
gap that contains many observationally equivalent cutoffs. The validation
sample does not identify which member of that gap will transfer.

Holding weights fixed and deriving alternative cutoffs from the original
validation scores gives the following retrospective development diagnostic:

| Model / rule | Cutoff | Old test now development FPR | FNR | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Baseline / original conservative | 0.944453 | 0 | 1 | 0.500 |
| Baseline / validation-gap midpoint | 0.503116 | 0 | 0 | 1.000 |
| Baseline / just above validation negative maximum | 0.061779 | 0.333 | 0 | 0.833 |
| Selected / original conservative | 0.539989 | 0 | 1 | 0.500 |
| Selected / validation-gap midpoint | 0.500150 | 0 | 0 | 1.000 |
| Selected / just above validation negative maximum | 0.460311 | 0.333 | 0 | 0.833 |

The midpoint calculation is **not** adopted as a newly validated detector.
Inspecting these six outcomes cannot establish its generalization. It shows
that the failure is threshold-dependent with these fixed weights, not a lack
of ranking separation. The low-end cutoff's false positive demonstrates the
trade-off hidden by the tiny validation set. Nothing here isolates whether
era, source, register, limited training diversity, or another characteristic
caused the score distributions.

## Topic-disjoint development folds

The fixed seed 20260910 shuffles the 12 topic pairs. Each of 12 folds holds out
one pair, reserves the next three pairs cyclically for validation, and fits
on the other eight pairs. Thus every article receives exactly one held-out
development prediction; related human/generated pairs never cross a fold.
Normalization is fitted only on that fold's training articles. The original
validation threshold rule is recomputed within each fold.

| Fixed configuration | FPR | FNR | Accuracy | Mean within-fold AUROC |
| --- | ---: | ---: | ---: | ---: |
| Original ten features, L2=0.1 | 0.000 | 0.250 | 0.875 | 1.000 |
| Pilot-selected eight features, L2=10 | 0.000 | 0.333 | 0.833 | 1.000 |
| Two length features only, L2=0.1 | 0.167 | 0.667 | 0.583 | 0.833 |
| Without repetition features, L2=0.1 | 0.000 | 0.250 | 0.875 | 1.000 |

These estimates are not comparable to the old final result as evidence of
improvement: the training composition and evaluation scheme changed, and all
articles are already familiar development material. Raw scores from different
fits are not pooled into one AUROC; the reported AUROC is the mean of the
within-fold values. FPR/FNR use the pooled fold-specific threshold decisions.

Original-model cutoffs range from 0.6420 to 0.9373 (median 0.8015). The
eight-feature cutoffs range from 0.5089 to 0.5310 (median 0.5165). Scores are
on different scales under different penalties, so smaller numerical movement
does not establish greater robustness. The eight-feature configuration still
misses four generated articles, compared with three for the original model.
The length-only configuration sometimes selects the explicit no-positive
sentinel above 1; the full precision is retained in JSON.

The 2,000-replicate topic-pair bootstrap gives an FNR interval of [0, 0.50]
for the original model and approximately [0.083, 0.583] for the eight-feature
model. These are conditional, small-sample development intervals; they do not
include refitting uncertainty. Zero observed false positives can produce a
degenerate bootstrap interval. That is not evidence of zero population risk;
the separate exact-binomial intervals also explicitly require independence.

## Shortcut and formatting controls

Uppercasing and collapsing whitespace leave scores unchanged for both saved
models, as expected from their features. Adding sentence-level bullet markers
changes 5 of 24 baseline decisions and 4 of 24 selected-model decisions. Maximum
absolute score changes are approximately 0.1541 and 0.01849 respectively.
The words and their order are preserved, but punctuation-rate features change.
This is evidence of sensitivity to this particular formatting intervention,
not a complete audit of semantic equivalence or realistic deployment shifts.

Length alone is insufficient to reproduce the full model's development
performance; removing repetition does not change its aggregate decisions.
Neither observation proves that the remaining features identify authorship.
In particular, a feature can exploit source/register differences without
distinguishing the intended construct in a contemporary population.

A paired-label permutation sanity check independently swaps origin labels
within topic pairs, refits the original configuration in every fold, and
recomputes mean fold AUROC. None of 99 permuted statistics reaches the observed
value, giving a Monte Carlo p-value of 0.01 with the usual plus-one correction.
This is exploratory, conditional on within-pair label exchangeability, and
tests association rather than its cause. Source/era confounding is not removed
by rejecting a chance-label explanation.

## What cannot be determined

Source-disjoint generalization is not identifiable here: the single historical
issue is aligned with human origin and the single assistant with generated
origin. Holding out one of those sources removes an origin class from the
other side of the comparison. The protocol records this as unavailable rather
than manufacturing folds with missing classes or treating articles as publishers.

No independent quality labels exist under the proposed rubric. The admission
audit therefore has zero documents/pairs and null agreement statistics. The
dataset review identifies possible auxiliary resources but no accepted corpus;
three terminal access probes returned proxy 403 errors. Even successful access
would not make proficiency or helpfulness scores equivalent to this rubric.

The current evidence supports a narrow conclusion: the original operating
point is unstable in this pilot, and a formatting feature can alter decisions.
It does not establish a reliable contemporary detector, the causes of all
observed shifts, a validated quality classifier, or internet-wide prevalence.
The new final and independently generated challenge evaluations remain blocked.