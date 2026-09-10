# Resource and access audit — September 10, 2026

This audit identifies useful resources, not an acquired representative corpus.
No reviewed resource was imported as a fully qualified contemporary, independently
quality-labeled corpus for this study. The findings below describe the inspected
documentation; they do not establish that no suitable dataset exists anywhere.

| Resource | Useful evidence | Gap for this study | Decision |
| --- | --- | --- | --- |
| RAID | Documented model, decoding, attack, domain and source identifiers; broad authorship stress tests | The inspected card does not provide this rubric's independent quality annotations or establish a 2024–2026 provenance-controlled human sampling frame | Candidate external authorship challenge only, after item-level permission/provenance review |
| M4 | Multi-generator, multi-domain, multilingual and time-domain evaluation | Does not by itself establish the required contemporary human provenance and independently adjudicated quality labels | Candidate cross-domain/generator comparison, not accepted as final corpus |
| CoAuthor | Recorded human–AI writing interactions support mixed-authorship analysis | Collaboration is not verified human-only origin; older writing sessions and no demonstrated labels under this rubric | Candidate mixed-authorship methodological reference, not primary binary data |
| SummEval | Published model summaries with crowd and expert quality judgments; annotations span coherence, consistency, fluency and relevance | Summarization-specific, pre-2024 material, different rubric, and not a balanced human-only/generated web-text corpus | Candidate separate annotation-method study; no conversion into this project's ground truth |

Primary source locations inspected:

```text
https://aclanthology.org/2024.acl-long.674/
https://huggingface.co/datasets/liamdugan/raid
https://github.com/mbzuai-nlp/M4
https://coauthor.stanford.edu/
https://github.com/Yale-LILY/SummEval
https://aclanthology.org/2021.tacl-1.24/
```

The RAID card lists an MIT license; that metadata was not treated as an item-level
clearance of every upstream human text. SummEval's maintainers state that model
outputs were shared with the original authors' consent and that source articles
are distributed separately. Neither statement replaces this project's document-
level permission and provenance audit. No legal clearance for redistribution
of upstream text is claimed here.

Two direct bulk-access probes were attempted from the terminal:

```text
https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl
https://huggingface.co/datasets/liamdugan/raid/raw/main/README.md
```

Both failed with `URLError: Tunnel connection failed: 403 Forbidden`. Browser
access to documentation succeeded. No bulk dataset was downloaded, no outside
annotation file was analyzed, and no web summary was used as a human-origin
sample. No independent annotators or consented participants are connected to
this workspace. Those limitations are recorded rather than replaced with
assistant-created labels or statements of consent.

The threshold design was checked against the primary order-statistic treatment
in Tong, Feng and Li (2018), *Neyman-Pearson classification algorithms and NP
receiver operating characteristics*, DOI `10.1126/sciadv.aao1659`:

```text
https://pmc.ncbi.nlm.nih.gov/articles/PMC5804623/
```

Its relevant principle is independent negative calibration for controlling the
probability of exceeding a specified type-I error. The present code uses the
simple maximum-negative special case, not a claim to reproduce the paper's
entire algorithm. The guarantee's independence/transport assumptions remain
requirements, not facts established by the pilot.