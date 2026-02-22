import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import EssayDataset
from src.model import EssayModel
from src.train import train
from src.evaluate import evaluate

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/processed/essays.csv")

train_df, val_df = train_test_split(df, test_size=0.2)

train_dataset = EssayDataset(train_df)
val_dataset = EssayDataset(val_df)

model = EssayModel()

train(model, train_dataset)

evaluate(model, val_dataset)

print("Pipeline Complete.")