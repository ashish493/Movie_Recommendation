import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=50):
        super(RecommenderNet, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def initialize_model(num_users, num_items):
    model = RecommenderNet(num_users, num_items)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    return model, optimizer, loss_fn

def train_model(model, optimizer, loss_fn, train_data, epochs=5):
    model.to(device)  # Ensure the model is on the correct device
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for _, row in train_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            rating = row['rating']

            # Check if indices are within valid ranges
            if user_id >= model.user_embedding.num_embeddings or movie_id >= model.item_embedding.num_embeddings:
                print(f"Invalid index: user_id={user_id}, movie_id={movie_id}")
                continue
            
            user = torch.tensor([user_id], dtype=torch.long).to(device)
            item = torch.tensor([movie_id], dtype=torch.long).to(device)
            rating = torch.tensor([rating], dtype=torch.float32).to(device).view(-1, 1)  # Ensure rating is the same size as output

            optimizer.zero_grad()
            output = model(user, item)
            loss = loss_fn(output, rating)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

def predict_rating(model, user_id, movie_id):
    model.eval()
    with torch.no_grad():
        user = torch.tensor([user_id], dtype=torch.long).to(device)
        item = torch.tensor([movie_id], dtype=torch.long).to(device)
        output = model(user, item)
        return output.item()

def load_model(num_users, num_items, model_path='model.pth'):
    model, optimizer, loss_fn = initialize_model(num_users, num_items)
    model.load_state_dict(torch.load(model_path))
    return model