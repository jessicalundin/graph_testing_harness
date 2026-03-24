# From Guidelines to Guarantees: A Graph-Based Evaluation Harness for Domain-Specific LLMs

This repository contains code accompanying the paper: **"From Guidelines to Guarantees: A Graph-Based Evaluation Harness for Domain-Specific LLMs"** on [arXiv](https://arxiv.org/abs/2508.20810).

We present a graph-based evaluation harness that transforms structured clinical guidelines into a queryable knowledge graph and dynamically instantiates evaluation queries via graph traversal. The framework provides three guarantees: **(1)** complete coverage of guideline relationships; **(2)** surface-form contamination resistance through combinatorial variation; and **(3)** validity inherited from expert-authored graph structure.

As a demonstration, we built a graph from the [WHO IMCI guidelines](https://www.who.int/publications/m/item/integrated-management-of-childhood-illness---chart-booklet-(march-2014)) and generated 432 clinically grounded multiple-choice questions spanning symptom recognition, treatment, severity classification, and follow-up care. These questions can be used for evaluation or post-training (finetuning, alignment).

## Overview

Rigorous evaluation of domain-specific language models requires benchmarks that are comprehensive, contamination-resistant, and maintainable. Static, manually curated datasets do not satisfy these properties. This project addresses the evaluation coverage problem by:

- **Transforming** structured clinical guidelines into a queryable knowledge graph
- **Dynamically instantiating** evaluation queries via graph traversal — not static datasets
- **Ensuring complete coverage** of all encoded guideline relationships
- **Reducing contamination risk** through combinatorial variation in templates, ages, and distractors
- **Inheriting clinical validity** from expert-authored graph structure

The generated MCQA dataset serves dual purposes: benchmarking models and providing data for post-training. Within alignment, MCQA provides naturally ranked outputs for methods such as GRPO, where correct answers serve as high-reward samples and incorrect options serve as progressively lower-reward samples.

## File Structure

```
├── databases/
│   └── IMCI_schema.graphml                  # NetworkX graph structure
├── results/
│   └── IMCI_qamc.json                       # Generated multiple-choice questions
├── src/
│   ├── generate_questions.py                # Multiple-choice question generation
│   └── inference.py                         # AI model inference and evaluation
├── LICENSE
└── README.md
```

## Knowledge Graph

### Graph Construction

We transform the WHO IMCI handbook (an 80-page document containing flowcharts and checklists for childhood illness management) into a directed graph. The resulting graph contains **200+ nodes** and **300+ edges** spanning respiratory, gastrointestinal, nutritional, and infectious diseases.

Automated extraction via PDF parsers and LLMs failed to reliably capture the conditional logic embedded in IMCI flowcharts — relationships expressed visually through color-coded triage paths and nested decision branches cannot be faithfully reconstructed as directed edges by current pipelines. The knowledge graph was therefore **manually curated by a co-author** who is a board-certified physician with over 15 years of clinical practice, specialized pediatric training, and extensive experience implementing WHO IMCI guidelines in clinical settings in sub-Saharan Africa.

This authorship model — where domain expertise is embedded at the graph construction stage rather than applied as post-hoc review — provides stronger validity guarantees than question-level annotation alone: every generated question inherits its clinical accuracy from expert-constructed graph relationships.

### Graph Schema

**Node Types**:
| Type | Count | Description |
|------|-------|-------------|
| `Condition` | 31 | Medical conditions with age range attributes (0–2 months for young infants, 2–60 months for children) |
| `Symptom` | 79 | Observable clinical indicators (e.g., "fast breathing", "convulsions") |
| `Treatment` | 84 | Medical interventions (e.g., "give oral Amoxicillin for 5 days") |
| `FollowUp` | 15 | Monitoring schedules (e.g., "3 days", "7 days") |
| `Severity` | 4 | Triage classifications (severe, moderate, mild, none) |

**Edge Types**:
- `INDICATES`: Symptom → Condition
- `TREAT`: Condition → Treatment
- `FOLLOW`: Condition → FollowUp
- `TRIAGE`: Condition → Severity

## Generated Questions (`results/IMCI_qamc.json`)

The system generates **432 questions** across five relationship types:

| Type | Count | Example |
|------|-------|---------|
| Condition → Treatment | 130 | "Which treatment is recommended for a 21 month old child with Very Severe Disease?" |
| Symptom → Condition | 118 | "A 21 month old child presenting with convulsions is most likely to have:" |
| Condition → Symptom | 118 | "A 2 year old child with Very Severe Disease would most likely present with which symptom?" |
| Condition → Severity | 37 | "A 13 month old child with Very Severe Disease should be classified as:" |
| Condition → FollowUp | 29 | "What is the appropriate follow-up schedule for a 3 year old child with Some Dehydration?" |

### Template-Based Question Generation

Each question type uses **4 templates** to create variety and reduce the influence of phrasing artifacts. The same clinical relationship tested under different phrasings produces meaningfully different accuracy — the `cond_followup_t1` template consistently produces the lowest accuracy across models (14–57%), while `cond_symp_t3` produces some of the highest (50–90%). Using multiple templates per question type provides more robust capability estimates than single-template approaches.

### Age-Aware Distractor Generation

The distractor sampling algorithm ensures clinical validity through age-stratified selection. For each question requiring k=3 distractors, the system identifies all conditions sharing the same age range as the target condition, creating an age-appropriate candidate pool. Distractors are then sampled uniformly without replacement from this pool, ensuring all distractors are clinically plausible within the relevant age group.

### Contamination Resistance

The harness addresses two distinct contamination risks that static benchmarks cannot mitigate:

- **Surface-form contamination**: By generating questions at evaluation time with randomized ages, distractor sampling, and template selection drawn from a large combinatorial space, the probability of repeated surface forms is reduced relative to static benchmarks.
- **Relationship-level contamination**: Because evaluation queries are generated dynamically from a structured representation of the guidelines, the same framework can be applied to updated or modified guidelines that postdate model training — enabling **temporal and versioned evaluation** to probe whether models have genuinely acquired generalizable clinical reasoning or are relying on memorized relationships.

### How to Run Inference

```bash
python src/inference.py --models gpt-4o-mini olmo2:7b --max-questions 50
```

The inference script leverages [AI Suite](https://github.com/andrewyng/aisuite) to support both cloud-based (OpenAI, Anthropic, Google) and local (Ollama) models. Results are written to `results/inference/inference_results.jsonl`.

## Usage

### Generate Questions

```bash
python src/generate_questions.py
```

This script:
1. Loads the existing NetworkX graph from `databases/IMCI_schema.graphml`
2. Generates multiple-choice questions with age-specific scenarios
3. Creates age-appropriate distractors for realistic difficulty
4. Saves questions to `results/IMCI_qamc.json`


