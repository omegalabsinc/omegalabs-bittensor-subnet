import torch
import torch.nn as nn

MODEL_PATH = "./omega/models/imagebind_desc_mlp.pth"

# Define the MLP model with more layers and a softmax output
class ImagebindMLP(nn.Module):
    def __init__(self, input_dim):
        super(ImagebindMLP, self).__init__()
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

# Load the model
input_dim = 1024 # input dimension for embeds is 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImagebindMLP(input_dim).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def is_desc_embedding_valid(embedding):
    # Ensure the embedding is on the correct device and has the correct shape
    if not isinstance(embedding, torch.Tensor):
        # Convert the embedding to a tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(device)
        #raise ValueError("Embedding should be a torch.Tensor")
    else:
        embedding_tensor = embedding.clone().detach().unsqueeze(0).to(device)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        prediction = model(embedding_tensor)
    
    # Get the predicted class and its probability
    predicted_probabilities = prediction.squeeze().cpu().numpy()  # Move back to CPU for numpy operations
    predicted_label = predicted_probabilities.argmax()
    #prediction_probability = predicted_probabilities[predicted_label]

    # Interpret the result
    #print(f"Prediction probabilities: {predicted_probabilities}")
    #print(f"Predicted label: {'valid' if predicted_label == 1 else 'random'}")
    #print(f"Prediction probability: {prediction_probability}")
    #print(f"predicted_label: {predicted_label}")

    # if valid, return true
    if predicted_label == 1:
        return True
    else:
        return False
    