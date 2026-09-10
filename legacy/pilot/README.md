# Authorship pilot: sources, tuning, and limits

This is a deliberately small experiment, completed on September 10, 2026.
It tests the authorship classifier, **not the quality classifier or a joint
slop detector**. Its disappointing operating-point result is retained.

## Data provenance

The 12 historical passages come from *Scientific American*, Volume XLIII,
No. 25, December 18, 1880. Project Gutenberg lists the digital edition as
eBook 21081, released April 15, 2007, and public domain in the United States.
The catalogue and full-text locations are recorded in `protocol.json`:

```text
https://www.gutenberg.org/ebooks/21081
https://www.gutenberg.org/cache/epub/21081/pg21081-images.html
```

`human_articles.json` contains excerpts of the historical article bodies, not
the catalogue's automatically generated summaries. Titles and passage locations
identify each excerpt. They were transcribed from the browser-returned source
text, with paragraph whitespace and nonbreaking spaces normalized. Direct
terminal downloads were denied by the environment's network proxy. No automated
full-source extraction or original-print image audit is claimed. The edition
credits Verity White, Juliet Sutherland, and the Online Distributed Proofreading
Team. Historical wording, including obsolete scientific claims, is retained as
data rather than endorsed as current scientific or practical advice.

These dates provide a strong basis for assigning historical human origin, but
do not rule out transcription errors. All passages share a publication and an
issue. Unnamed authors may also recur. Do not present the split as publisher-,
author-, or issue-disjoint.

`generated_articles.json` contains 12 articles written once by the assistant
during this session, before fitting or examining any detector score. They match
the historical topics but do not reproduce the historical prose. A fixed seed
assigns one of three writing styles to each topic. That assignment is random;
topic selection and language-model generation are **not** claimed to be random
samples. The source texts were visible to the assistant, so generation is
source-informed rather than blind. The exact backend model identifier,
temperature, and generation seed were unavailable. The style seed is not a
generation seed. The articles are experimental outputs, not fact-checked news.

## Split and selection protocol

`protocol.json` was written before fitting. It is a local protocol, not an
externally timestamped preregistration. The assistant could see the test topics
and texts; only their detector scores were withheld from model selection.

- Training: six topic pairs, 12 documents.
- Validation: three topic pairs, six documents.
- Final test: three topic pairs, six documents.

Both members of a pair stay in the same split. Group checks and normalized exact
duplicate checks run before fitting. Quality labels remain blank throughout.
The historical samples contain 124–182 code-tokenized words; generated samples
contain 143–158. This overlap is useful, but does not remove the confounding of
historical versus modern register with the origin label.

Tuning proceeds in three steps:

1. Fit the original ten-feature baseline at L2 penalty 0.1.
2. Compare penalties 0.01, 1, and 10 with the same features.
3. Repeat all four settings without log token count and mean sentence length.

Every candidate has training-only normalization and an authorship threshold
selected on validation data to maximize recall at empirical FPR at most 5%.
With only three historical validation passages, that constraint permits zero
false positives; it does not establish a reliable population FPR.

Select by validation recall, AUROC, lower FPR, fewer features, then stronger
regularization. The last two rules resolve all ties here: every candidate has
validation recall 1.0, AUROC 1.0, and FPR 0.0. The selected model therefore uses
eight active features and penalty 10. This is a preference among tied candidates,
not demonstrated validation improvement. Other length-related features remain;
the ablation does not create a fully length-invariant model.

## Actual held-out result

| Model | TP | FP | TN | FN | Accuracy | Recall | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 0 | 0 | 3 | 3 | 0.5 | 0 | 1.0 |
| Selected model | 0 | 0 | 3 | 3 | 0.5 | 0 | 1.0 |
| Always negative | 0 | 0 | 3 | 3 | 0.5 | 0 | 0.5 |

Precision is undefined because no article is flagged. The selected threshold is
approximately 0.5400. Generated test scores range from 0.5146 to 0.5372, above
all historical scores but below that threshold. A perfect AUROC can therefore
coexist with a useless operating point. **Tuning does not improve the held-out
decisions in this run.** No threshold is adjusted after testing.

Zero false positives among three historical passages offers little reassurance.
Even an idealized independent-binomial calculation gives a one-sided 95% upper
error-rate bound of about 63.2%. Independence is questionable within one issue,
so this calculation illustrates limited information rather than establishing
a modern-web confidence interval. No quality, slop-prevalence, cross-generator,
fairness, or contemporary-domain result is inferred.

## Artifacts and reproduction

Run the three stages from the workspace root, using a fresh output directory:

```bash
python run_pilot.py prepare --output pilot/reproduction
python run_pilot.py select --output pilot/reproduction
python run_pilot.py evaluate --output pilot/reproduction
```

`artifacts/corpus.csv` and `artifacts/manifest.json` preserve prepared documents,
source locations, prompts, counts, split assignments, and text hashes.
`artifacts/selection.json` preserves every candidate's validation metrics and
fingerprints of the input data, protocol, implementation, and frozen models.
`artifacts/test_results.json` and `artifacts/test_predictions.json` preserve
aggregate and individual test outcomes. `artifacts/results_tables.tex` and
`artifacts/pilot_results.pdf` are generated from those recorded results. The
paper includes the tables and draws the same figure directly in LaTeX, so its
build does not require the standalone plot PDF. The model JSON files are
compatible with `detector.py`:

```bash
python detector.py predict pilot/artifacts/selected_model.json article.txt
```

Reproduction reuses frozen generated text, not a live model call. Floating-point
details may vary with library versions. A rerun of these fixed articles is not
new evidence, even in a fresh directory.

## Next iteration

The observed failure can inform a new development cycle. First collect current,
provenance-controlled human writing and outputs from multiple documented
generators. Obtain quality labels through independent blinded annotation rather
than asking this assistant to label its own output. Allocate substantially more
human validation examples and a new, untouched final set. Prespecify threshold
selection and examine its stability across development folds before comparing
new models. The old test set can become development material only if its new
role is explicit. Do not tune repeatedly on this test set and call the resulting
improvement a held-out success.