import torch
from transformers import AutoModel

from src.config import MODEL_NAME, NUM_LABELS


class EssayModel(torch.nn.Module):
    def __init__(self):
        super(EssayModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(
            self.encoder.config.hidden_size,
            NUM_LABELS
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        logits = self.classifier(x)

        return logits