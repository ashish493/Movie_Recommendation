import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Matrix Factorization model class
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        
    def forward(self, user, item):
        user = user.to(device) - 1
        item = item.to(device) - 1
        u, it = self.user_factors(user), self.item_factors(item)
        return (u * it).sum(1) * 5

# Model initialization function
def initialize_model(n_users, n_items, n_factors=20):
    model = MatrixFactorization(n_users, n_items, n_factors).to(device)
    return model

# Model training function
def train_model(model, train_data, epochs=20, batch_size=32, learning_rate=1e-3):
    opt = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    model.train(True)
    
    # Training loop
    for epoch in range(epochs):
        avg_loss = []
        
        for _ in range(len(train_data) // batch_size):
            df = train_data.sample(frac=batch_size / len(train_data))
            users = torch.tensor(df.user_id.values, dtype=torch.long, device=device)
            items = torch.tensor(df.movie_id.values, dtype=torch.long, device=device)
            targets = torch.tensor(df.rating.values, dtype=torch.float32, device=device)

            opt.zero_grad()
            preds = model(users, items)
            loss = criterion(preds, targets)
            loss.backward()
            opt.step()
            avg_loss.append(loss.item())
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {sum(avg_loss) / len(avg_loss)}")

# Save model state function
def save_model(model, path="matrix_factorization_model.pth"):
    torch.save(model.state_dict(), path)

# Load model state function
def load_model_state(model, path="matrix_factorization_model.pth"):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

# Execute training only if script is run directly
if __name__ == "__main__":
    COLS = ['user_id', 'movie_id', 'rating', 'timestamp']
    train_data = pd.read_csv("./dataset/ml-100k/u1.base", sep='\t', names=COLS).drop(columns=['timestamp']).astype(int)
    n_users, n_items = 943, 1682

    # Initialize and train model
    model = initialize_model(n_users, n_items)
    train_model(model, train_data)
    save_model(model)
