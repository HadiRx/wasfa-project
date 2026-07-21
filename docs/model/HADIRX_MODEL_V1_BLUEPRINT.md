# HadiRx Model V1 — Data Engine Before Model

Owner: Hadi Alanazi  
Scope: Saudi provider-side, no-PHI, read-only medication + ICD-10-AM review support.

## Product boundary

HadiRx does not make a medical decision, coverage determination, prior-authorization decision, or claim-submission decision. It produces review signals grounded in traceable evidence:

- `context_present`
- `documentation_gap`
- `requires_review`

Every output must include evidence references, uncertainty, missing elements, and routing. Unsupported certainty is a release blocker.

## Operating principle

The first product is not a fine-tuned chatbot. The first product is a governed data engine with frozen evaluation sets. Training is allowed only after a simpler baseline is measured and the candidate wins on the same hidden test set.

The development loop is:

1. collect governed examples;
2. define the expected structured output;
3. freeze train/dev/test splits;
4. run retrieval and rule baselines;
5. inspect failures by category;
6. improve data, retrieval, or classification;
7. train only the smallest component that addresses the measured failure;
8. reject candidates that improve averages while worsening safety-critical slices.

## V1 system

### Stage A — query normalization

Normalize medication, active ingredient, diagnosis/code, language, setting, and available documentation. No free-text patient identifiers are accepted.

### Stage B — hybrid retrieval

Retrieve candidates using lexical and dense retrieval. Preserve source identity, version, jurisdiction, effective date, document section, and exact chunk boundaries.

### Stage C — evidence filtering and reranking

Rerank evidence against the complete case, not only the drug name. Filters must remove wrong jurisdiction, expired versions, unrelated indications, and unsupported inferred requirements.

### Stage D — deterministic feature extraction

Extract structured facts:

- medication and ingredient match;
- diagnosis/code match;
- age/setting constraints when explicitly supplied;
- required documentation elements;
- present and missing documentation;
- contradictions;
- source recency and authority.

### Stage E — signal classifier

The classifier predicts one of the three allowed signals plus calibrated confidence. It cannot create evidence or final policy conclusions.

### Stage F — grounded explanation

A generator converts the structured result into Arabic or English. Every substantive claim must map to evidence IDs. The explanation must state that the signal is advisory and requires role-appropriate review where applicable.

## What we train first

Priority order:

1. reranker;
2. signal classifier;
3. information extractor;
4. small explanation model only after the structured pipeline passes.

We do not train a foundation model from scratch.

## Dataset units

Each example contains:

- de-identified case input;
- allowed signal;
- expected missing elements;
- expected routing;
- positive evidence IDs;
- hard-negative evidence IDs;
- annotator rationale;
- provenance and review status;
- safety tags.

Synthetic examples are permitted but must be labeled `synthetic` and must not enter the final test set. The final test set must contain independently reviewed, non-PHI cases.

## Split policy

Avoid random row splitting. Split by medication family, source document, and scenario family to reduce leakage.

- train: 70%
- development: 15%
- hidden test: 15%

The hidden test labels are not used for prompt or model iteration.

## Evaluation gates

### Retrieval

- Recall@5 and Recall@20 for required evidence;
- MRR/nDCG for ranking;
- hard-negative rejection;
- authority/version accuracy.

### Classification

- macro F1;
- per-class recall;
- calibration error;
- confusion matrix by medication family and source type.

### Safety-critical gates

A release fails when any of these regress beyond the accepted baseline:

- false `context_present` when required documentation is missing;
- unsupported coverage or PA conclusion;
- missing citation for a substantive claim;
- wrong-jurisdiction evidence;
- PHI leakage;
- incorrect routing in high-risk cases.

### Generation

- claim-level evidence support;
- completeness of required caveats;
- unsupported-claim rate;
- bilingual consistency.

## Initial milestones

### M0 — governance and validator

- canonical JSONL schema;
- PHI-pattern guard;
- source and split validation;
- seed examples and CI.

### M1 — 300-case gold set

- 100 cases per signal;
- Arabic and English coverage;
- at least 20 hard-negative pairs per major source family;
- dual review for hidden test cases.

### M2 — retrieval benchmark

Compare BM25, current embeddings, multilingual embeddings, and hybrid retrieval before any fine-tuning.

### M3 — reranker experiment

Train or fine-tune a cross-encoder on positive and hard-negative evidence pairs. Accept only if hidden-test evidence recall and safety slices improve.

### M4 — signal classifier

Train a compact classifier on structured features plus selected evidence. Calibrate confidence and define abstention thresholds.

### M5 — controlled generation

Generate explanations from structured JSON and cited evidence. No unconstrained diagnosis or reimbursement decision text.

## Architecture decision

HadiRx Model V1 is a governed retrieval-and-classification system with a language layer, not a monolithic medical chatbot. Its durable asset is the Saudi-specific, expert-reviewed evaluation and training data engine.