import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DEVICE, LR, EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH


def train(model, train_dataset):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader)

        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Average Loss:", total_loss / len(train_loader))

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Model saved at:", MODEL_SAVE_PATH)