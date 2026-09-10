# AI Text Detector Research Project

A reproducible research project for experimenting with **human-vs-AI text classification**, with a current focus on short and medium-length English text.

The project has evolved through multiple model and dataset experiments. The current best development model is **V11**, a linear Support Vector Machine using normalized word- and character-level TF-IDF features and trained on a combination of the HC3 corpus and a separate modern human/AI benchmark.

> **Important:** This project is a research prototype, not a universally validated AI detector. A high score on one benchmark does not imply the same accuracy on arbitrary writing, websites, authors, subjects, or future language models.

---

## Current status

### Current model: V11

The current V11 classifier uses:

* word TF-IDF features with 1–3 word n-grams
* character TF-IDF features with 2–6 character n-grams
* HTML/Markdown normalization
* normalized quotation and dash characters
* `LinearSVC`
* `C=3`
* a threshold selected using validation data with a target validation FPR of at most 5%

Training data combines:

1. A paired subset of the **HC3 English corpus**
2. The project's separate modern human/AI benchmark

The current combined training set contains:

```text
Training:   2,248 texts
Validation:   620 texts
Labels:      exactly balanced
```

The HC3 portion was built from matched human/ChatGPT answer pairs. The benchmark portion contains multiple human/AI generation conditions.

---

# Quick start

## Requirements

The current development environment used:

```text
Python       3.13.14
numpy        2.5.2
scipy        1.18.0
scikit-learn 1.9.0
joblib       1.5.3
```

The standalone detector does not require pandas.

For the current environment:

```bash
pip3 install numpy==2.5.2 scipy==1.18.0 scikit-learn==1.9.0 joblib==1.5.3
```

For reproducing the broader training pipeline, pandas is also useful:

```bash
pip3 install pandas==3.0.5
```

The exact dependency record is stored in:

```text
models/current/requirements_v11.txt
```

---

# Running the detector

The simplest interface is:

```bash
python3 detector_v11.py article.txt
```

The detector reports:

```text
File
Words
Score
Threshold
Margin
Prediction
```

Example:

```text
File: article.txt
Words: 225
Score: -0.192923
Threshold: -0.364735
Margin: 0.171812
Prediction: AI
```

### Interpreting the result

The score is a **decision-function score**, not a calibrated probability.

The rule is:

```text
score >= threshold  -> AI
score <  threshold  -> HUMAN
```

The margin is:

```text
score - threshold
```

A positive margin means the document is on the AI side of the learned decision boundary.

A negative margin means it is on the human side.

Do not interpret:

```text
score = 0.80
```

as:

```text
80% probability the text is AI
```

No such probability calibration is currently claimed.

---

# Portable release

The intended inference-only package is:

```text
release/
└── V11_RELEASE/
    ├── detector_v11.py
    ├── hc3_v11_portable.joblib
    ├── hc3_v11_combined_results.json
    └── requirements_v11.txt
```

The portable model contains the trained:

* word TF-IDF vectorizer
* character TF-IDF vectorizer
* LinearSVC model
* decision threshold

The standalone `detector_v11.py` contains its own text-normalization function, so inference does not depend on importing the training script.

Run the release version from inside its directory:

```bash
cd release/V11_RELEASE
python3 detector_v11.py ../../samples/article.txt
```

The `.joblib` model is the learned model. The training datasets are **not** needed for ordinary inference.

---

# Reproducing training

The current V11 training workflow is:

```text
raw HC3 data
       |
       v
paired HC3 corpus
       |
       v
combined training/validation corpus
       |
       v
normalized word + character TF-IDF
       |
       v
LinearSVC
       |
       v
validation threshold selection
       |
       v
V11 model
```

Important training files:

```text
training/build_hc3_paired_v10.py
training/build_combined_v11.py
training/train_combined_v11.py
```

The corresponding processed data are stored under:

```text
dataset/
```

The raw HC3 files are stored under:

```text
data/hc3/
```

The separate modern benchmark is stored under:

```text
data/benchmark/
```

---

# Current V11 training configuration

The current combined model uses:

### Word features

```text
TF-IDF
word analyzer
n-grams: 1–3
max features: 50,000
min_df: 2
sublinear TF: enabled
```

### Character features

```text
TF-IDF
char_wb analyzer
n-grams: 2–6
max features: 80,000
min_df: 2
sublinear TF: enabled
```

### Model

```text
LinearSVC
C = 3.0
```

### Text normalization

Before vectorization:

* HTML tags are removed
* Markdown emphasis markers are removed
* curly quotation marks are normalized
* curly apostrophes are normalized
* different dash characters are normalized
* repeated whitespace is collapsed

The normalization is performed before feature extraction.

---

# V11 training data

V11 combines two different sources.

## HC3

The HC3 English corpus contains human and ChatGPT answers associated with the same questions.

The current paired construction produced:

```text
3,278 paired questions
6,556 texts
3,278 human
3,278 ChatGPT
```

For the V11 combined dataset, the HC3 portion was capped by domain so that finance did not completely dominate the training set.

The retained HC3 domains are:

```text
finance
medicine
open_qa
```

Each pair contributes:

```text
1 human answer
1 AI answer
```

The same question/pair is kept in only one split.

## Modern benchmark

The project also contains a separate modern benchmark with multiple generation conditions.

Its purpose is to measure whether the detector transfers beyond the HC3-style question/answer setting.

The benchmark contains:

```text
300 documents
150 human
150 AI
```

and includes several generation conditions, including:

```text
gpt-4o
paraphrased_gpt-4o
claude
o1-pro
humanized_o1-pro
```

The benchmark remains useful for historical comparison, but it should not be treated as a universal estimate of real-world detector accuracy.

---

# Current measured V11 results

On the constructed V11 validation set:

```text
Validation size: 620

Accuracy:  97.42%
Precision: 95.65%
Recall:    99.35%
F1:        97.47%
FPR:        4.52%
AUROC:     0.9991
```

Confusion counts:

```text
TN: 296
FP:  14
FN:   2
TP: 308
```

These are **validation results for this project dataset**.

They are not a claim that V11 will achieve 97.42% accuracy on arbitrary Internet text.

---

# External sanity checks

The project includes a small set of manually prepared external articles used only as qualitative sanity checks.

Current V11 behavior:

```text
article_human.txt -> HUMAN
article_ai.txt    -> AI
article_mixed.txt -> AI
article.txt       -> AI
```

These examples are useful for development, but four articles are nowhere near enough to establish statistical generalization.

The correct interpretation is:

> V11 successfully classified these four particular external examples.

The incorrect interpretation is:

> V11 is proven to be 100% accurate on real-world text.

---

# Model history

The project deliberately preserves earlier experiments.

Important historical versions include:

```text
V2
V3
V4
V5
V6
V7
V8
V11
```

They explored:

* handcrafted stylometric features
* word TF-IDF
* character TF-IDF
* word + character combinations
* LinearSVC
* logistic regression
* class weighting
* short-text training
* score ensembles
* formatting normalization
* broader training data
* HC3 expansion

Older models are stored under:

```text
models/archive/
```

Their corresponding result files are retained so future experiments can be compared against earlier work.

Do not delete historical model artifacts merely because they are no longer the best model.

---

# Important lessons from the experiments

## 1. Benchmark accuracy can be misleading

The original modern benchmark produced very strong results for several configurations.

However, external examples showed that a model can perform extremely well on the benchmark while failing on writing that comes from a different distribution.

This is one of the central research findings of the project.

A detector can learn:

```text
dataset-specific patterns
```

instead of:

```text
general properties of AI-generated writing
```

Therefore benchmark performance must always be interpreted together with out-of-distribution evaluation.

---

## 2. Short text behaves differently

The original benchmark documents were roughly 700–750 words on average.

The project's external examples were roughly 150–300 words.

A dedicated short-text model improved sensitivity on the project's external AI examples but caused substantially more human false positives on the benchmark.

Therefore:

> There is no evidence yet that one simple threshold is optimal for every document length and domain.

---

## 3. Formatting artifacts can leak into classifiers

Feature inspection showed strong model weights associated with patterns such as:

```text
<br>
<b>
**
quotes
```

Some of these artifacts were unevenly distributed between human and AI benchmark documents.

That is dangerous because such patterns may describe the dataset-generation process rather than authorship.

V8 therefore added normalization before feature extraction.

V8 did not eliminate the external generalization problem, but the experiment demonstrated why artifact auditing is necessary.

---

## 4. More diverse training data helped

The major improvement came from adding paired HC3 data rather than endlessly tuning the original 300-document benchmark.

The combined V11 model successfully classified the project's external examples that V5 missed.

This suggests that:

```text
training-data diversity
```

is currently more important than another small hyperparameter search.

---

# Current model limitations

V11 is not a universal authorship detector.

Known limitations include:

### Language

The current project is primarily English-focused.

### Length

Very short text can be unstable.

The current project does not establish a reliable universal minimum or maximum document length for detection.

### Domain shift

Performance can change substantially across:

* journalism
* academic writing
* blogs
* forums
* technical writing
* fiction
* social media
* marketing
* edited professional text

### Model shift

The current training data do not represent every current or future language model.

### Humanization/paraphrasing

AI text that has been heavily edited, paraphrased, or humanized may behave differently from ordinary machine output.

### False positives

A human-written document can be classified as AI.

A high detector score is not proof of AI authorship.

### False negatives

AI-generated text can be classified as human.

A low score is not proof of human authorship.

### Probability

The current SVM score is not a calibrated probability.

---

# Responsible use

This project should be used as a **research and screening tool**, not as definitive evidence of authorship.

Do not use a detector score by itself to:

* accuse someone of cheating
* accuse someone of plagiarism
* punish a student
* remove an author's work
* reject a publication
* make employment decisions
* establish legal responsibility

A detector output should be treated as one weak piece of evidence among many.

For high-stakes decisions, human review and provenance evidence are more important than a single classifier score.

---

# Development philosophy

The project follows these principles:

## Never tune on the final evaluation set

Once a test set has been inspected repeatedly, it should be treated as a comparison benchmark rather than a truly untouched final test.

The project therefore distinguishes between:

```text
training
validation
comparison benchmark
```

rather than pretending every measured number is an independent estimate.

## Prefer grouped splits

Related documents, prompts, variants, and answers should stay in the same split where possible.

This reduces leakage from near-duplicates and shared prompts.

## Preserve model history

Do not overwrite previous models.

Every major experiment should produce:

```text
model artifact
configuration
results
dataset version
```

## Prefer data improvement over blind hyperparameter searching

When a model fails on external text, the first question should be:

> Is the training distribution representative?

rather than:

> Which value of C should I try next?

---

# Recommended future development

The current highest-value direction is **broader and better-controlled data**, not another arbitrary TF-IDF sweep.

Useful future directions include:

1. More modern human text from independent sources
2. More AI generations from different models
3. Human-edited and partially AI-assisted text
4. Cross-domain evaluation
5. Time-based evaluation
6. More independent external challenge sets
7. Calibration of the final score
8. Length-stratified thresholds
9. Robustness against paraphrasing and humanization
10. Confidence/abstention behavior
11. Transformer-based models after sufficient data expansion
12. Proper uncertainty intervals and repeated group-level evaluation

A future model should ideally be evaluated on:

```text
in-domain data
cross-domain data
cross-model data
short text
long text
human-edited AI text
fully human text
fully AI text
```

before making broad performance claims.

---

# Project structure

The current project is organized as follows:

```text
New Project/
│
├── README.md
├── detector_v11.py
│
├── release/
│   └── V11_RELEASE/
│
├── models/
│   ├── current/
│   └── archive/
│
├── data/
│   ├── hc3/
│   ├── benchmark/
│   └── historical_1900/
│
├── dataset/
│
├── training/
│
├── analysis/
│
├── evaluation/
│
├── tools/
│
├── samples/
│
├── docs/
│
└── legacy/
```

### `models/current/`

Current production/development artifacts.

### `models/archive/`

Older model versions and experiment results.

### `data/`

Original external datasets and historical source material.

### `dataset/`

Processed datasets generated by project scripts.

### `training/`

Corpus-building, optimization, and model-training scripts.

### `analysis/`

Dataset and feature analysis.

### `evaluation/`

Evaluation utilities.

### `tools/`

Packaging and model-management utilities.

### `samples/`

Example input documents.

### `docs/`

Paper/research documentation.

### `legacy/`

The original pilot and obsolete detector implementations.

---

# Useful commands

## Run current detector

```bash
python3 detector_v11.py samples/article.txt
```

## Run release detector

```bash
cd release/V11_RELEASE
python3 detector_v11.py ../../samples/article.txt
```

## Check project structure

```bash
find . -maxdepth 2 -type f | sort
```

## Check project size

```bash
du -sh .
```

## Check installed versions

```bash
python3 -c "import sys, numpy, scipy, sklearn, joblib; print(sys.version); print(numpy.__version__); print(scipy.__version__); print(sklearn.__version__); print(joblib.__version__)"
```

---

# Re-training V11

The core training sequence is:

```bash
python3 training/build_hc3_paired_v10.py
python3 training/build_combined_v11.py
python3 training/train_combined_v11.py
```

Do not run retraining blindly over the current model files if you need the exact current V11 artifact.

Preserve the existing model first:

```bash
cp models/current/hc3_v11_combined_model.joblib \
   models/archive/hc3_v11_combined_model_backup.joblib
```

A new experimental model should normally receive a new version identifier.

For example:

```text
V12
V13
V14
```

rather than overwriting V11.

---

# Exact reproducibility

For inference reproducibility, preserve:

```text
detector_v11.py
hc3_v11_portable.joblib
requirements_v11.txt
```

For training reproducibility, preserve:

```text
dataset/
data/
training/
models/
```

and the exact software versions.

The trained model artifact contains the learned feature vocabulary, model weights, and threshold.

The dataset is still needed to reproduce or extend the training process.

---

# Research record

This project contains several generations of experiments.

The historical pilot used older data and an earlier handcrafted feature model.

Subsequent experiments demonstrated:

* operating-point failures despite high AUROC
* benchmark/generalization mismatch
* sensitivity to document length
* formatting leakage
* the limits of short-text-only training
* the value of adding a broader paired corpus
* the importance of keeping validation and comparison data conceptually separate

These are part of the research record and should not be rewritten merely because a later model performs better.

---

# What V11 is

V11 is:

```text
a supervised binary text classifier
```

using:

```text
TF-IDF + LinearSVC
```

with:

```text
word features
+
character features
+
text normalization
```

It is **not**:

```text
a fine-tuned language model
```

and it does not inspect model internals, hidden probabilities, metadata, or Internet URLs.

It only evaluates the text supplied to it.

---

# Final note

The project's goal is not to produce an impressive detector number.

The goal is to determine:

> How well can a text classifier distinguish human and AI-generated writing under realistic distribution shift, and where does that approach fail?

The most important results are therefore not just the highest benchmark score.

The failure cases, external checks, data leakage analysis, distribution shifts, and preserved experiment history are equally important.

For that reason, every future model should be compared against V11 using the same documented evaluation protocol before being called an improvement.
