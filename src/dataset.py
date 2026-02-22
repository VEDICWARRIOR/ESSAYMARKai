import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import MODEL_NAME, MAX_LENGTH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class EssayDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx]["essay"]
        label = self.df.iloc[idx]["label"]

        encoding = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }