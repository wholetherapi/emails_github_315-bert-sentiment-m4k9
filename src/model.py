from transformers import BertModel
import torch
from torch import nn


class AspectSentimentClassifier(nn.Module):
    """Aspect-aware sentiment classifier on top of a BERT encoder."""

    def __init__(self, num_labels=3):
        super().__init__()
        self.encoder = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, aspect_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = (outputs.last_hidden_state * aspect_mask.unsqueeze(-1)).sum(1)
        return self.classifier(pooled)
