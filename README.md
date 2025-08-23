# IMCI Graph Testing Harness

This repository contains code accompanying the paper: **"A Graph-Based Test-Harness for LLM Evaluation"** on arXiv (link forthcoming).

This is a prototype for a graph-based LLM testing harness to achieve 100% coverage of a set of health guidelines -- to our knowledge the first of it's kind.  The questions can be dynamically generated for 3+ trillion combinations of the 400+ questions from this particular graph; however the methodology can be extended to any use case.  As a demonstration, we built a graph from the [WHO IMCI guidelines](https://www.who.int/publications/m/item/integrated-management-of-childhood-illness---chart-booklet-(march-2014)), and generated multiple-choice questions.  These questions can be used for evaluation or post-training (finetuning, alignment).

## Overview

This project processes IMCI medical guidelines to create:
- **Multiple-choice questions** for medical knowledge assessment and LLM benchmarking
- **Graph-based knowledge representation** for structured medical data
- **Age-aware distractor generation** for realistic question complexity

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

## Core Data

The system processes **34 medical conditions** from the IMCI guidelines, covering:

- **Age ranges**: 0-2 months (young infants) and 2-60 months (children)
- **Medical domains**: Respiratory, gastrointestinal, nutritional, infectious diseases
- **Severity levels**: Pink (severe), Yellow (moderate), Green (mild)

### Data Structure

Each condition includes:
- **Symptoms**: Clinical indicators (e.g., "fast breathing", "convulsions")
- **Treatments**: Medical interventions (e.g., "Give oral Amoxicillin for 5 days")
- **Follow-up**: Monitoring schedules (e.g., "3 days", "7 days")
- **Severity**: Triage classification (Pink/Yellow/Green)

## Generated Output

### Multiple-Choice Questions (`results/IMCI_qamc.json`)

The system generates **5 types of questions**:

1. **Condition-to-Symptom**: "A 4 year old child with Very Severe Disease would most likely present with which symptom?"
2. **Symptom-to-Condition**: "A 11 month old child presenting with not feeding well is most likely to have:"
3. **Treatment Questions**: "What is the appropriate treatment for a 20 month old child diagnosed with Pneumonia?"
4. **Follow-up Questions**: "When should a 3 year old child with Cough or Cold return for follow-up?"
5. **Severity Questions**: "What is the severity classification of Very Severe Disease in a 2 year old child?"

Each question includes:
- Age-specific scenarios (random ages within appropriate ranges)
- 4 multiple-choice options (A, B, C, D)
- Correct answer identification
- Template variations for question diversity

### Graph Structure (`databases/IMCI_schema.graphml`)

The NetworkX graph contains:

**Node Types**:
- `Condition`: Medical conditions with age_range property
- `Symptom`: Clinical indicators
- `Treatment`: Medical interventions
- `FollowUp`: Monitoring schedules
- `Severity`: Triage classifications (severe/moderate/mild)

**Relationship Types**:
- `INDICATES`: Symptom → Condition
- `TREAT`: Condition → Treatment
- `FOLLOW`: Condition → FollowUp
- `TRIAGE`: Condition → Severity

## Inference with AI Models

The project includes an automated inference script (`src/inference.py`) for benchmarking the generated multiple-choice questions (MCQA) using a variety of AI models. This script leverages [AI Suite](https://github.com/andrewyng/aisuite) to support both cloud-based (OpenAI, Anthropic, Google) and local (Ollama) models.


### How to use
**Run inference:**
   ```bash
   python src/inference.py --models gpt-4o-mini olmo2:7b --max-questions 50
   ```

### Output
- **JSONL:** Full inference results for each model/question (in `results/inference/inference_results.jsonl`)


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

### Output Files

- `results/IMCI_qamc.json`: 400+ multiple-choice questions with age-aware distractors

## Key Features

### Age-Aware Distractor Generation
The system creates realistic multiple-choice questions by generating age-appropriate distractors:

### Template-Based Question Generation
Each question type uses **4 different templates** to create variety and reduce memorization:

- **Condition→Symptom questions** (4 templates):
  - "Which symptom indicates {condition} in a {age} old child?"
  - "What is a key symptom of {condition} in a {age} old child?"
  - "A {age} old child with {condition} would most likely present with which symptom?"
  - "Which of the following symptoms suggests {condition} in a {age} old child?"

- **Symptom→Condition questions** (4 templates):
  - "A {age} old child with {symptom} most likely has which condition?"
  - "Which condition should you suspect in a {age} old child presenting with {symptom}?"
  - "A {age} old child presenting with {symptom} is most likely to have:"
  - "What is the most probable diagnosis for a {age} old child with {symptom}?"

- **Treatment, Follow-up, and Severity questions** each have their own 4 template variations

### Age-Specific Clinical Accuracy
Each question uses a specific random age within the appropriate range

## Contributing

This is a research/testing tool for medical knowledge graph evaluation. The generated questions and schemas are based on WHO IMCI guidelines and should be used for educational and research purposes only. This repository is not regularly maintained.