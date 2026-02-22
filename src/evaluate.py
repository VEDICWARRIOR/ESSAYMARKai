import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from src.config import DEVICE, BATCH_SIZE


def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)

    print("Accuracy:", acc)
    print(classification_report(all_labels, all_preds))