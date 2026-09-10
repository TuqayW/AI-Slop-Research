# AI Text Detector Research Project

> **Research project by Tuqay Mehdiyev**  
> Baku, Azerbaijan · Baku European Lyceum  
> **Project date:** 10 September 2026

## About this project

This repository is an independent research and software project exploring a simple but difficult question:

> **Can machine-learning methods distinguish human-written text from AI-generated text reliably when the writing style, domain, length, and generation process change?**

The project started as a small experimental detector and gradually became a broader investigation into **AI-text detection, dataset quality, distribution shift, false positives, false negatives, leakage, formatting artifacts, and reproducible evaluation**.

The goal is not to build a system that simply produces an impressive benchmark number. The goal is to understand **what a detector learns, when it works, when it fails, and how much confidence we should place in its output**.

The project currently culminates in **V11**, a supervised text classifier based on normalized word- and character-level TF-IDF features with a linear Support Vector Machine.

> **Important:** V11 is a research prototype, not a universally validated AI detector. Strong performance on a particular benchmark does not imply equivalent performance on arbitrary writing, websites, authors, subjects, languages, or future language models. fileciteturn0file0L5-L7

---

## Author

### Tuqay Mehdiyev

**Location:** Baku, Azerbaijan  
**School:** Baku European Lyceum  
**Date of this research snapshot:** 10 September 2026

Tuqay Mehdiyev is a student researcher developing this project around interests in **artificial intelligence, machine learning, natural-language processing, software development, and reproducible research**.

The project is designed not only as a detector, but also as a learning and research record: each model version documents a question, an experiment, a result, and a reason for the next change.

---

# Why I started this project

AI-generated writing is increasingly common, but detecting it from the final text alone is much harder than it first appears.

A classifier can perform extremely well on one dataset because it learns patterns that are genuinely associated with AI generation. But it can also perform well for the wrong reason—for example, by learning:

- formatting conventions,
- source-specific writing styles,
- prompt-specific vocabulary,
- document-length differences,
- dataset-construction artifacts.

That creates a central research problem:

> **Is the model learning something about AI-generated language, or is it learning something about the dataset?**

This repository is an attempt to investigate that distinction experimentally.

---

# The research journey

The project has evolved through multiple model generations.

```text
Original pilot
      ↓
V2 / V3
      ↓
Modern benchmark construction
      ↓
Grouped / leakage-aware evaluation
      ↓
V4 / V5 feature + model search
      ↓
V6 short-text experiment
      ↓
V7 ensemble experiment
      ↓
V8 formatting normalization
      ↓
HC3 corpus expansion
      ↓
V11 combined model
```

Each stage was kept because the failures are scientifically useful.

For example:

- **V5** performed strongly on the modern benchmark but failed on one external AI example.
- **V6** became much more sensitive to short AI examples but produced an unacceptable increase in human false positives.
- **V8** normalized obvious HTML/Markdown artifacts, but normalization alone did not solve the external generalization problem.
- **V11** combined broader HC3 data with the modern benchmark and successfully classified the current set of small external sanity checks.

This progression is one of the main research findings of the project.

---

# Current model: V11

V11 is a supervised binary text classifier using:

```text
Word TF-IDF
    +
Character TF-IDF
    +
LinearSVC
```

### Word features

```text
analyzer: word
n-grams: 1–3
max features: 50,000
min_df: 2
sublinear TF: enabled
```

### Character features

```text
analyzer: char_wb
n-grams: 2–6
max features: 80,000
min_df: 2
sublinear TF: enabled
```

### Classifier

```text
LinearSVC
C = 3.0
```

### Text normalization

Before feature extraction, V11:

- removes HTML tags
- removes Markdown emphasis markers
- normalizes curly quotation marks
- normalizes curly apostrophes
- normalizes dash variants
- collapses repeated whitespace

The implementation and current model settings are documented in the repository's training code and experiment records. fileciteturn0file0L232-L276

---

# How the detector works

The model receives the text itself.

It does not inspect:

- model internals
- browser metadata
- hidden API information
- author accounts
- Internet URLs
- external webpages

The text is transformed into sparse TF-IDF representations and passed to the LinearSVC classifier.

The detector produces a **decision-function score**.

The current rule is:

```text
score >= threshold  → AI
score <  threshold  → HUMAN
```

The current V11 threshold is:

```text
-0.364734947384451
```

The margin is:

```text
score - threshold
```

A positive margin means the document is on the AI side of the decision boundary.

A negative margin means it is on the human side.

The score is **not a calibrated probability**. For example, a score of `0.80` must not be interpreted as “80% probability the text is AI.” fileciteturn0file0L108-L141

---

# Run the detector

The main entry point is:

```bash
python3 detector_v11.py samples/article.txt
```

Example output:

```text
File: samples/article.txt
Words: 225
Score: -0.192923
Threshold: -0.364735
Margin: 0.171812
Prediction: AI
```

The model is intended to be a **screening/research tool**, not an authorship proof.

---

# Portable model

The inference-oriented release is stored under:

```text
release/V11_RELEASE/
```

with:

```text
detector_v11.py
hc3_v11_portable.joblib
hc3_v11_combined_results.json
requirements_v11.txt
```

The trained model artifact contains the learned vectorizers, classifier, and decision threshold.

The training datasets are not required for ordinary inference.

> **Release note:** the portable artifact was created as an experimental packaging step during the 10 September 2026 development session. Before calling it an official release, its outputs should be verified against the canonical V11 model artifact. This repository intentionally preserves both artifacts so that the discrepancy can be investigated rather than hidden.

---

# Training data

V11 combines two major sources.

## HC3

The English HC3 corpus provides paired human and ChatGPT answers.

The project's paired construction produced:

```text
3,278 paired questions
6,556 texts
3,278 human
3,278 ChatGPT
```

The V11 training construction used the domains:

```text
finance
medicine
open_qa
```

Each retained question contributes one human answer and one ChatGPT answer.

Train/validation separation is performed at the pair/question level so the same question does not occur in both splits. fileciteturn0file0L280-L314

## Modern benchmark

The project also contains a separate 300-document modern benchmark:

```text
150 human
150 AI
```

with multiple AI-generation conditions including:

```text
gpt-4o
paraphrased_gpt-4o
claude
o1-pro
humanized_o1-pro
```

The benchmark is valuable for comparison, subgroup analysis, and failure analysis, but it is not a universal estimate of real-world detector accuracy. fileciteturn0file0L316-L340

---

# Current V11 validation result

On the constructed V11 validation set:

```text
n = 620

Accuracy:  97.42%
Precision: 95.65%
Recall:    99.35%
F1:        97.47%
FPR:        4.52%
AUROC:     0.9991
```

Confusion matrix:

```text
TN = 296
FP =  14
FN =   2
TP = 308
```

These are **validation results for this project's data**. They should not be presented as proof that the detector will achieve the same accuracy on arbitrary Internet text. fileciteturn0file0L344-L370

---

# External sanity checks

The current small set of manually prepared external examples produced:

```text
article_human.txt → HUMAN
article_ai.txt    → AI
article_mixed.txt → AI
article.txt       → AI
```

These checks are useful because some earlier models failed on `article.txt`.

However, four external examples are not a statistically representative benchmark.

The correct claim is:

> V11 successfully classified these four particular external examples.

The incorrect claim is:

> V11 is proven to be 100% accurate on real-world text. fileciteturn0file0L374-L397

---

# What the experiments taught us

## Dataset performance is not the same as generalization

A classifier can learn dataset-specific signals instead of genuinely general signals of AI authorship.

This project therefore treats distribution shift as a central research question.

## Short text is different

The modern benchmark documents were substantially longer than some external examples.

A short-text specialist model improved sensitivity to several short AI examples but produced substantially more human false positives on the benchmark.

This suggests that a single universal threshold may not be optimal across all document lengths and domains. fileciteturn0file0L469-L481

## Formatting can leak into a classifier

Feature analysis revealed strong signals associated with formatting patterns such as:

```text
<br>
<b>
**
quotes
```

Some of these patterns were unevenly distributed across the benchmark labels.

That matters because a classifier can accidentally learn properties of the dataset-generation pipeline instead of properties of authorship.

V8 added explicit formatting normalization to investigate this problem. fileciteturn0file0L483-L500

## Broader data helped more than blind tuning

The major improvement in the development process came from adding broader paired HC3 data rather than continuously changing one hyperparameter.

This supports a practical lesson:

> **Better and more representative training data can matter more than small improvements in model configuration.** fileciteturn0file0L504-L516

---

# What V11 is not

V11 is:

```text
a supervised binary text classifier
```

It is **not**:

```text
a fine-tuned large language model
```

The model does not establish who wrote a document with certainty.

A positive prediction does not prove AI authorship.

A human prediction does not prove human authorship.

The model score is not a probability.

---

# Limitations

The current project remains limited by:

- English-focused training
- incomplete domain coverage
- limited representation of current/future AI systems
- distribution shift
- short-text instability
- false positives
- false negatives
- paraphrasing and humanization
- benchmark-specific artifacts
- small subgroup sizes in some analyses
- lack of fully independent large-scale external evaluation
- lack of calibrated probabilities
- inability to estimate real-world population prevalence from these experiments

These limitations are not side notes; they are part of the research question.

---

# Responsible use

This system should be treated as a **research and screening signal**, not definitive evidence.

A detector score should not, by itself, be used to:

- accuse someone of cheating
- accuse someone of plagiarism
- punish a student
- reject an academic paper
- remove someone's work
- make employment decisions
- establish legal responsibility

High-stakes decisions require stronger evidence, human review, and provenance information.

---

# Development structure

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
├── analysis/
├── evaluation/
├── tools/
├── samples/
├── docs/
└── legacy/
```

### `models/current/`

Current model artifacts and current result records.

### `models/archive/`

Older model versions and historical experiment results.

### `data/`

Original external and historical source material.

### `dataset/`

Processed datasets generated during development.

### `training/`

Dataset construction, optimization, and model-training scripts.

### `analysis/`

Dataset and feature analysis.

### `evaluation/`

Evaluation utilities.

### `tools/`

Packaging and model-management utilities.

### `samples/`

Example documents used during development.

### `docs/`

Research paper and research documentation.

### `legacy/`

Earlier detector implementations and experimental records.

---

# Reproducing the current model

The main V11 training path is:

```bash
python3 training/build_hc3_paired_v10.py
python3 training/build_combined_v11.py
python3 training/train_combined_v11.py
```

Important files:

```text
training/build_hc3_paired_v10.py
training/build_combined_v11.py
training/train_combined_v11.py
```

Processed training data are under:

```text
dataset/
```

Raw HC3 data are under:

```text
data/hc3/
```

The modern benchmark is under:

```text
data/benchmark/
```

---

# Development rules

## Do not overwrite V11

Future experiments should normally use:

```text
V12
V13
V14
...
```

rather than modifying V11 in place.

## Preserve previous models

Every serious experiment should keep:

```text
model artifact
configuration
dataset version
results
```

## Keep test data conceptually separate

Because development repeatedly inspected the existing comparison benchmark, it should no longer be described as a pristine untouched final test.

Future work should create a genuinely held-out challenge set before making new confirmatory claims.

## Prefer grouped evaluation

Related prompts, variants, and documents should remain together where possible.

## Prefer better data over endless hyperparameter searching

When an external example fails, investigate:

```text
distribution
domain
length
generation process
formatting
data leakage
```

before simply trying another value of `C`.

---

# Future research roadmap

The most valuable next steps are:

1. collect more independent modern human text
2. add AI generations from more systems and settings
3. add human-edited and partially AI-assisted writing
4. create time-based evaluation
5. create cross-domain and cross-model challenge sets
6. calibrate the classifier score
7. investigate length-aware decision rules
8. evaluate robustness to paraphrasing and humanization
9. quantify uncertainty
10. investigate transformer-based approaches only after improving data diversity

A serious future evaluation should include:

```text
in-domain text
cross-domain text
cross-model text
short text
long text
fully human text
fully AI-generated text
human-edited AI text
paraphrased AI text
```

Only after such evaluation should broader claims about detector performance be considered.

---

# Research archive and reproducibility

The repository intentionally preserves the history of failed and successful experiments.

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

The history records why models changed and what each experiment revealed.

The project therefore treats **failed experiments as evidence**, not as clutter.

That is especially important for detector research, where a model that scores well on one dataset can still fail when the data distribution changes.

---

# Code and paper

The research manuscript is maintained in:

```text
docs/main.tex
```

Supporting research/audit files include:

```text
docs/PROJECT_AUDIT.md
docs/audit_repository.py
docs/audit_results.json
docs/research_cycle.py
```

The paper should be updated whenever a substantial model, dataset, or evaluation change alters the scientific conclusions.

The repository's code and the manuscript should therefore be treated as two parts of the same research record:

```text
code + data + experiment history + paper
```

---

# Citation and archival plans

The GitHub repository is the living development record.

For a formal, citable research snapshot, a future release can also be archived through a service such as **Zenodo**, which can assign a DOI to published research outputs and can automatically archive GitHub releases once the repository is connected. citeturn180531search0turn180531search1turn180531search7

---

# Where this research could be published

Because this is a student-led project, I would prioritize venues that explicitly support student research.

## 1. Journal of Emerging Investigators (JEI)

This is probably the most natural first journal target for the current project.

JEI explicitly accepts middle- and high-school student authors aged 13 or older, and it connects student researchers with scientific mentors. However, JEI requires a **senior mentor as a co-author**, and the manuscript must be submitted by an adult. citeturn648701search4turn648701search11

That means the strongest path would be:

```text
Tuqay Mehdiyev
+
a qualified teacher / professor / research mentor
```

with the paper revised into a formal student research manuscript.

## 2. arXiv

arXiv can be useful as a public preprint once the manuscript is mature.

However, arXiv requires endorsement for a first submission or for a new category, and its current policy explains that a first-time submitter can seek endorsement from an established arXiv author. citeturn648701search0

For this project, I would treat arXiv as a **preprint/dissemination step**, not a replacement for peer review.

## 3. Journal of Open Source Software (JOSS)

JOSS is attractive because this repository is not just a paper—it is also an open research-software project.

JOSS is a peer-reviewed open-access journal for research software, but it expects open-source, feature-complete software with clear research significance and substantial scholarly contribution. citeturn648701search5turn648701search8

I would **not submit V11 to JOSS yet**. The software and evaluation protocol need to mature first, particularly around reproducible packaging, tests, licensing, and evidence of broader usefulness. JOSS's current guidance emphasizes research impact, scholarly contribution, maintainability, and credible reuse. citeturn648701search3turn648701search10

## 4. Zenodo + GitHub

This is not peer review, but it is an excellent archival step.

A published Zenodo record receives a DOI, and Zenodo supports GitHub integration for archiving repository releases. citeturn180531search0turn180531search7

A sensible sequence is:

```text
GitHub
   ↓
stable V12/V13 release
   ↓
Zenodo archive + DOI
   ↓
journal / preprint submission
```

---

# Recommended publication path

For this particular project, my recommended order is:

```text
1. Finish the research methodology
        ↓
2. Build a genuinely independent challenge set
        ↓
3. Get a teacher/research mentor
        ↓
4. Revise the manuscript with formal statistical evaluation
        ↓
5. Submit to JEI
        ↓
6. Archive the stable research release on Zenodo
        ↓
7. Consider arXiv
        ↓
8. Consider JOSS after the software becomes
   mature enough for its research-software criteria
```

That path is more credible than trying to publish immediately based on the current benchmark numbers alone.

---

# Project philosophy

The project is built around one principle:

> **A detector should be judged by how honestly it behaves under distribution shift, not only by how impressive its best benchmark score looks.**

The research therefore records:

- successes
- failures
- false positives
- false negatives
- dataset limitations
- leakage concerns
- formatting artifacts
- distribution shifts
- model versions
- reproducibility information

The objective is not to prove that AI text can always be detected.

The objective is to measure **how far we can get, why the approach works when it works, and where it breaks**.

---

## Current snapshot

**Author:** Tuqay Mehdiyev  
**Location:** Baku, Azerbaijan  
**School:** Baku European Lyceum  
**Date:** 10 September 2026  
**Current model:** V11  
**Primary task:** Human-vs-AI text classification  
**Current approach:** Normalized word + character TF-IDF with LinearSVC  
**Current validation set:** 620 texts  
**Current validation AUROC:** 0.9991  
**Current validation accuracy:** 97.42%  
**Current validation AI recall:** 99.35%  
**Current validation FPR:** 4.52%

> These figures describe the current project evaluation setup; they are not a universal estimate of detector accuracy.

