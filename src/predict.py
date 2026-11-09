import argparse
import json

import torch
from transformers import BertTokenizer

from src.model import AspectSentimentClassifier

LABELS = ["negative", "neutral", "positive"]


def predict(input_path, output_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = AspectSentimentClassifier().eval()
    rows = []
    with open(input_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            tokens = tokenizer(record["text"], return_tensors="pt")
            with torch.no_grad():
                logits = model(
                    tokens["input_ids"], tokens["attention_mask"], torch.ones_like(tokens["input_ids"])
                )
            rows.append({"label": LABELS[int(logits.argmax(-1))]})
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predict(args.input, args.output)
