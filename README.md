# PII NER Assignment Skeleton

This repo is a skeleton for a token-level NER model that tags PII in STT-style transcripts.

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

# Final Evaluation Metrics

## Per-Entity Metrics
| Entity        | Precision | Recall | F1 Score |
|---------------|-----------|--------|----------|
| CITY          | 1.000     | 0.423  | 0.595    |
| CREDIT_CARD   | 0.759     | 0.759  | 0.759    |
| DATE          | 0.821     | 0.742  | 0.780    |
| EMAIL         | 0.455     | 0.455  | 0.455    |
| LOCATION      | 0.000     | 0.000  | 0.000    |
| PERSON_NAME   | 0.418     | 0.605  | 0.495    |
| PHONE         | 0.519     | 0.519  | 0.519    |

**Macro-F1:** **0.514**

---

## Grouped Metrics

### PII-only Metrics
- **Precision:** 0.571  
- **Recall:** 0.626  
- **F1 Score:** 0.597  

### Non-PII Metrics
- **Precision:** 1.000  
- **Recall:** 0.256  
- **F1 Score:** 0.407  

---

## Latency (50 runs, batch_size=1)
- **p50:** 28.65 ms  
- **p95:** 34.58 ms  

