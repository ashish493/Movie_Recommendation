import os
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SparseAdam, Adam, Adagrad, SGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
train_data = pd.read_csv("./dataset/ml-100k/u1.base", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
test_data = pd.read_csv("./dataset/ml-100k/u1.test", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
n_users, n_items = 943, 1682

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
    def forward(self, user, item):
        user = user.to(device) - 1
        item = item.to(device) - 1
        u, it = self.user_factors(user), self.item_factors(item)
        x = (u * it).sum(1)
        assert x.shape == user.shape
        return x * 5

model = MatrixFactorization(n_users, n_items).to(device)
opt = Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()
BATCH_SIZE = 32

avg = []
mx = []
states = {}
model.train(True)
for e in range(20):
    for it in range(len(train_data) // BATCH_SIZE):
        # Setup batch data
        df = train_data.sample(frac=BATCH_SIZE / len(train_data))
        users = torch.tensor(df.user_id.values, dtype=torch.long, device=device)
        items = torch.tensor(df.movie_id.values, dtype=torch.long, device=device)
        targets = torch.tensor(df.rating.values, dtype=torch.float32, device=device)
        assert users.shape == (BATCH_SIZE,) == items.shape
        
        # Train model
        opt.zero_grad()
        preds = model(users, items)
        mx.append((preds.max().item(), preds.min().item()))
        loss = criterion(preds, targets)
        assert preds.shape == targets.shape
        loss.backward()
        opt.step()
        avg.append(loss.item())

    print(f"EPOCH {e+1}: {sum(avg) / len(avg)}")
    avg = []
    states[e+1] = model.state_dict()

torch.save(model.state_dict(), "matrix_factorization_model.pth")