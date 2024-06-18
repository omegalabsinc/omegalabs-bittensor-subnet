import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import json

# Define a simple custom dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return torch.tensor(embedding, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Define the MLP model with more layers and a softmax output
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.softmax(x)
        return x

# Load your dataset
# Assuming embeddings is a numpy array of shape (num_samples, embedding_dim)
# and labels is a numpy array of shape (num_samples,)
# Load the JSON file
with open('desc_embeddings_openai_scored.json', 'r') as f:
    data = json.load(f)

# Extract embeddings and labels
embeddings = []
labels = []

for item in data:
    embedding = item[1]
    label = item[2]
    
    embeddings.append(embedding)
    labels.append(label)

# Convert to numpy arrays
embeddings = np.array(embeddings)
labels = np.array(labels)

dataset = EmbeddingDataset(embeddings, labels)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss function, and optimizer
input_dim = embeddings.shape[1]
model = MLP(input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with validation
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for embeddings_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(embeddings_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for embeddings_batch, labels_batch in val_loader:
            outputs = model(embeddings_batch)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        print(f'Saving best model in epoch {epoch+1} with validation loss: {best_val_loss}')
        torch.save(model.state_dict(), 'best_mlp_openai_model.pth')

# Save the final model
#torch.save(model.state_dict(), 'final_mlp_openai_model.pth')