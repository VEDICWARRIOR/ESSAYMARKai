import torch
from transformers import AutoTokenizer

from src.model import EssayModel
from src.config import MODEL_NAME, DEVICE, MODEL_SAVE_PATH

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def predict(text):
    model = EssayModel()
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        prediction = torch.argmax(outputs, dim=1).item()

    return prediction