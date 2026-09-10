# Repository-to-manuscript audit — September 10, 2026

This audit precedes the V11 manuscript revision. It does not train another model,
relabel documents, or create an independent evaluation. The pre-revision paper
is retained as `docs/main_pre_v11.tex`. Historical datasets and weights are unchanged.

## Evidence and reproducibility

Run `python docs/audit_repository.py` from the repository root. The resulting
`docs/audit_results.json` hashes 171 input files, inventories every processed CSV,
checks 133 metric blocks against their confusion counts, reconstructs qualifying
HC3 pairs from all three raw JSONL files, and reconstructs the combined V11
membership. Both reconstructions match; all checked metric arithmetic matches.
AUROC cannot be reconstructed from confusion counts and is not included in that
arithmetic verification. The full raw benchmark is compared with the processed
texts and labels. Hashes establish identity, not authenticity or permissions.

The audit environment has Python 3.13.15, NumPy 2.4.1, SciPy 1.17.0 and pandas
2.3.3, but no scikit-learn or joblib. No dependencies were installed. Saved-model
inference, fitted coefficients, per-document scores and historical AUROCs were
not independently reproduced. Do not describe this audit as a training replay.
The uploaded dependency record describes a different environment.

Non-executing inspection with `pickletools.genops` of serialized V11 metadata
confirms word 1–3-gram TF-IDF, 50,000 features, min_df=2, sublinear_tf=True,
lowercase=True, L2 normalization, and no accent stripping. The character
vectorizer has char_wb 2–6 grams, 80,000 features and min_df=2. LinearSVC stores
C=3.0, squared_hinge, L2 penalty, class_weight=None, random_state=None,
max_iter=1000 and 130,000 input coordinates. Embedded estimator metadata records
scikit-learn 1.9.0. This is metadata inspection, not execution or weight validation.
The current and release portable-model files are byte-identical.

## Reconstructed development sequence

| Stage | Implementation/data | Evidence and interpretation |
|---|---|---|
| Handcrafted baseline | `legacy/detector.py` | Ten standardized surface features and separate logistic heads; no completed independently labeled quality model. |
| 1880 pilot | `legacy/run_pilot.py`, `legacy/pilot/` | 12 historical + 12 assistant-generated passages; original test TN=3, FP=0, FN=3, TP=0 despite AUROC=1. |
| Second-cycle diagnostics | `legacy/cycle2*`, `docs/research_cycle.py` | Reuses all 24 passages; fold and perturbation results are exploratory. The prospective annotation audit remains null. |
| V2 | `legacy/detector_v2.py`, `models/archive/v2_results.json` | Word/character logistic model; 12/6/6 pilot rows; perfect archived metrics, not independent confirmation. |
| V3 | `training/build_v3_corpus.py`, `dataset/v3_corpus.csv` | 30/14/16 rows; 30 human excerpts balanced with 30 fixed generated articles. Human token joining strips punctuation; AI text retains it. |
| Modern benchmark | `data/benchmark/human_detectors.json`, grouped CSV builder | 300 rows, 150 human/AI article pairs, five generation conditions, 30 numeric prompt groups; 180/60/60 rows. It is NOT HC3 despite older filenames. |
| Benchmark logistic models | `hc3_v2_results.json`, `hc3_best_results.json` | Comparison accuracy 0.9333 and 0.9667; 21-setting three-fold grouped search for the latter. |
| V4 | `optimize_hc3_v4.py`, `train_best_hc3_v4.py` | 48 settings; character 3–5-gram SVM, C=10; comparison FP=2/30. |
| V5 | V5 search/train scripts and JSONs | 135 settings; character 2–6-gram SVM, C=10; comparison FP=1/30, FN=0/30. README reports a miss on article.txt. |
| V6 | Short corpus builder/train script | 474 training chunks, 160 validation chunks; full-document comparison FP=9/30, FN=0/30. |
| V7 | Ensemble script and search JSON | 21 mixtures; selected V5 weight=1, V6 weight=0, so no ensemble gain. |
| V8 | Normalized training script/JSON | Normalization plus character SVM; comparison FP=1/30, FN=0/30; external transfer still problematic according to README. |
| V9 / HC3 development | `make_dev_v9.py`, `build_hc3_dev.py` | 227 length-filtered benchmark candidates; 11,593 HC3 answers of at least 80 whitespace words. No V9 fitted result is supplied. |
| V10 pairing | `build_hc3_paired_v10.py` | First eligible answer of each origin per question; 3,278 pairs; shuffled pair-level 80/20 split. No separate V10 fitted result is supplied. |
| V11 | Combined builder/trainer and current artifacts | 2,248 training / 620 validation; balanced classes; validation-selected threshold −0.364734947384451. |

## Findings requiring corrections, not cosmetic rewriting

1. **Group separation does not establish text separation.** V11 has zero shared
   prompt IDs between splits, but one identical human medical answer occurs in
   training at `medicine:416` and `medicine:695`, and in validation at
   `medicine:1124`. It is byte-identical as a text field, not just a fuzzy match.
   This duplicate is already present in paired V10. There are two surplus rows
   for this three-row equivalence class. Do not call V11 leakage-free. Do not
   remove these rows and retain the old scores as if they describe cleaned data.

2. **The modern benchmark is not 30 underlying articles.** There are 150 distinct
   `(source,title,issue)` pairs, each contributing one row of each label. Thirty
   numeric `prompt_id` groups each collect ten rows across five generation
   conditions. No article pair crosses the stored splits. The earlier
   `analysis/analyze_human_detectors.py` and `fix_hc3_groups.py` expect ten rows per
   article tuple; that expectation does not describe the uploaded data. The
   actual grouped builder uses numeric prompt IDs. IDs are not unique document
   identifiers; use provenance plus row/content fingerprints.

3. **V11 subgroup results are not retained in its JSON.** The training script
   prints them, but the upload contains no corresponding stdout or row scores.
   The supplied finance FPR≈0.057 and recall≈0.9962 are incompatible with the
   combined validation's 150 human and 150 AI finance rows. The uncapped paired
   validation has 526 per label, permitting 30/526 and 524/526; this arithmetic
   compatibility does not establish which model or run produced the numbers.
   Medicine's 126 negatives and open QA's four negatives occur in both versions,
   so those denominators cannot resolve attribution. Do not mix these figures
   into the aggregate V11 result. Recompute a source/origin breakdown from the
   archived model and exact combined CSV in the recorded environment.

4. **Only one external article is uploaded.** `samples/article.txt` contains
   175 whitespace-delimited words. The other three named files are absent.
   README records four V11 predictions and earlier transfer failures, but no
   per-file score/provenance ledger accompanies them. Preserve them as reported
   development observations, not an independently verified four-document test.
   A mixed-authorship example does not have a binary authorship ground truth
   merely because the model returns AI. Sample content is not evidence about
   Internet prevalence or traffic.

5. **Quality annotation is still uncompleted.** The raw benchmark contains five
   annotator fields with origin guesses/confidence/comments and other detectors'
   outputs. Those are not the independent low-quality rubric labels. Processed
   `low_quality` columns are blank. Do not turn guesses into provenance or quality.

6. **Formatting differs by label.** Across all 300 benchmark rows, human/AI
   document counts are 0/23 for double asterisks, 30/60 for `<br>` variants,
   144/65 for curly double quotes, and 11/86 for straight double quotes.
   Any HTML-like tag occurs in 83/72. These are retrospective descriptive counts,
   not causal attribution or an independent robustness test. The prior feature
   weight audit is described in README, but no dedicated weight-audit output is
   supplied. Preserve that distinction.

7. **Search and comparison are exploratory.** V5 computes fold thresholds and
   fold decision metrics on the same validation folds. Its grouped CV fitting
   separation does not make those operating-point metrics unbiased. V5/V6/V7/V8
   repeatedly inspect the same 60-row comparison set. V11 incorporates benchmark
   training/validation material; that benchmark is not independent external
   validation of V11. No new untouched final test result is supplied.

8. **Normalization is a limited regex procedure.** It removes HTML-like tags,
   `**`, and isolated `*`, not every Markdown construct. It standardizes curly
   quotes/apostrophes, en/em dashes and whitespace. There is no HTML parser,
   factual verification, URL fetching, quality head, probability calibration,
   or minimum-length abstention in the V11 CLI. Each TF-IDF block is L2-normalized;
   concatenation is not an extra whole-vector normalization step.

9. **Data provenance and permissions are incomplete.** The benchmark has source
   links and inconsistent date encodings, but no bundled collection/generation
   protocol or complete rights ledger. Raw HC3 files have questions/answers but
   no acquisition log, upstream revision, full dataset citation or item-level
   permissions. Do not invent these. Historical Gutenberg text includes its
   own notice; that does not license the contemporary benchmark. The paper does
   not redistribute article bodies, personal author identifiers or annotator
   comments. A release needs appropriate source attribution and rights review.

10. **Paths and environments have drifted.** The root detector expects a root
    model that is now in `models/current`. The release detector correctly names
    its adjacent portable model but requires launching from the release folder.
    Pairing scripts expect root-level raw HC3 JSONLs. Trainers write root-level
    outputs. `docs/research_cycle.py` imports `detector` and uses old root-relative
    paths. These archived scripts must not be presented as a tested turnkey
    pipeline in the reorganized checkout. The current paper is self-contained
    and does not include missing root-level pilot/cycle2 figures or tables.

## Manuscript decisions

The manuscript retains the original research questions, useful major headings,
historical equations, failed pilot, diagnostic cycle and valid bibliography.
It adds corpus composition, model evolution, exact V11 validation metrics,
formatting counts, missing-evidence distinctions and this audit's leakage finding.
It removes the obsolete claim that no modern data have been acquired, but does
not replace it with a claim of representative provenance-controlled collection.
It does not call V11 a quality detector, a fine-tuned language model, a calibrated
probability model, a confirmed generalization improvement or a prevalence estimator.

This record supports an exploratory detector-development case study. A stronger
confirmatory contribution needs repaired connected-component splits, complete
provenance and permissions, restored external samples and score logs, independent
human quality annotations if quality is studied, fresh custodian-held evaluation,
source/topic/generator challenges, and appropriate uncertainty analysis. Those
are empirical and infrastructure requirements, not writing changes.

## Final manuscript verification

`latexmk -cd -pdf -interaction=nonstopmode -halt-on-error docs/main.tex`
successfully builds `docs/main.pdf` (20 pages). The final LaTeX log has no
undefined references, undefined citations, missing inputs/figures, or overfull
boxes. The model-evolution table and evidence-map page were rendered and visually
checked; preview images are outside the project in the temporary preview folder.
All 21 headings from the original main manuscript are retained, all ten
bibliography entries are unchanged and cited, and named references resolve.
The exact six V11 aggregate rate/ranking values match the result JSON. All 171
fingerprinted research inputs still match the pre-revision audit: no historical
model, dataset, result, source script, or README was changed. The inference
dependency limitation remains; a successful paper build is not a model replay.