import torch
import torch.nn as nn

# Define the MLP model with the same architecture
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Load the model
input_dim = 1024 # input dimension for embeds is 1024
model = MLP(input_dim)
model.load_state_dict(torch.load('best_mlp_model.pth'))
model.eval()

def check_desc_embedding_against_MLP(embedding):
    # Convert the embedding to a tensor
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        prediction = model(embedding_tensor).squeeze().item()

    # Interpret the result
    predicted_label = 1 if prediction >= 0.7 else 0
    print(f"Prediction probability: {prediction}")
    print(f"Predicted label: {'valid' if predicted_label == 1 else 'random'}")