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

## Inference Results

### Model Performance

Baseline evaluation across five models using the generated MCQA dataset:

| Model | Overall | C→S | S→C | C→T | C→Sv | C→F |
|-------|---------|-----|-----|-----|------|-----|
| Claude Sonnet 4.6 | 68.0±13.6 | 73.9±8.4 | 80.3±5.1 | 69.7±8.1 | 56.0±15.7 | 60.0±15.3 |
| GPT-5.2 | 66.3±15.2 | 77.3±12.5 | 79.4±8.1 | 69.6±11.1 | 53.9±10.6 | 51.4±9.9 |
| o4-mini | 67.5±14.0 | 75.1±5.7 | 81.6±4.1 | 65.3±7.5 | 58.0±11.8 | 57.5±19.9 |
| GPT-OSS-20B | 56.9±15.5 | 68.9±5.5 | 71.2±3.6 | 49.7±2.2 | 51.8±14.1 | 42.9±21.0 |
| MedGemma-4B | 49.8±10.4 | 50.0±2.7 | 64.4±1.6 | 45.4±4.9 | 47.6±10.1 | 41.4±11.8 |

*Values shown as accuracy ± standard deviation (%)*

### Key Findings

1. The three frontier closed-source models (Claude Sonnet 4.6, o4-mini, GPT-5.2) achieve similar overall accuracy (~66–68%), outperforming GPT-OSS-20B (~57%) and MedGemma-4B (~50%).
2. Symptom → Condition questions show the highest performance across all models (64–82%), indicating that models better recognize symptoms than prescribe treatments or protocols.
3. Within-model performance varies substantially across question types, underscoring that aggregate accuracy obscures meaningful capability differences.
4. MedGemma-4B underperforms larger models across all question types, suggesting model scale and general reasoning capacity may dominate performance in this setting.

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

## Extensibility

The graph schema — conditions, symptoms, treatments, follow-ups, severities, and their directed relationships — is not specific to IMCI. Any clinical guideline with structured decision logic is a candidate. WHO produces guidelines across malaria, tuberculosis, HIV, and maternal health that share the same flowchart structure as IMCI. Beyond healthcare, structured regulatory guidelines, legal compliance frameworks, and technical standards with explicit relationship structures could support the same approach.

The primary scaling bottleneck is graph construction itself. Future work could reduce this bottleneck through semi-automated graph construction with expert review, particularly for guidelines with consistent structure such as WHO protocols.

## Limitations

- Question quality depends entirely on graph accuracy: any errors in manual annotation propagate to all generated questions.
- The graph was curated by a single clinical expert, which precludes inter-rater reliability assessment. Independent validation by additional clinicians remains important future work.
- Only MCQA format is evaluated, which cannot capture the complexity of real clinical reasoning involving differential diagnosis and incomplete information.
- The text-only approach excludes visual diagnostic elements present in the original IMCI handbook.
- Evaluation on IMCI guidelines may not generalize to other medical domains.

## Ethical Considerations

This evaluation harness is intended for research purposes only and is not suitable for clinical decision-making. Models performing well on MCQA may still fail in actual clinical scenarios requiring differential diagnosis and incomplete information. Our focus on WHO IMCI guidelines reflects the substantial need for AI systems that support scarce healthcare workers in low- and middle-income countries (LMICs), where guidelines are often country-specific and custom evaluation is necessary for accurate measurement of model performance.

## Citation

If you use this work, please cite the accompanying paper (forthcoming ACL proceedings).

## License

This repository is released under an open-source license to enable reproducibility and extension to other clinical guidelines. The generated questions and schemas are based on WHO IMCI guidelines and should be used for educational and research purposes only.
