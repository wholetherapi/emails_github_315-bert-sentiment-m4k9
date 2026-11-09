# BERT-Sentiment — Aspect-Based Sentiment Analysis

**Status: Released (v1.2.0)**

Aspect-based sentiment analysis toolkit built on fine-tuned BERT encoders.
The repository contains the released model code and inference utilities used
for the SemEval experiments.

## Contents

- `src/model.py` — aspect-aware BERT classifier
- `src/predict.py` — batch inference utility
- `requirements.txt` — runtime dependencies

## Usage

```bash
pip install -r requirements.txt
python -m src.predict --input reviews.jsonl --output predictions.jsonl
```

## License

MIT
