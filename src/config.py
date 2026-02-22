import torch

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 2
MODEL_SAVE_PATH = "models/model.pt"