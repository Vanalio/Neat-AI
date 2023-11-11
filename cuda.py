import torch
import torch.nn as nn

# Check if NVIDIA GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define a simple dense neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Define layers, assuming a simple feedforward network as an example
        self.fc1 = nn.Linear(in_features=10, out_features=20)  # Replace with your architecture
        self.fc2 = nn.Linear(in_features=20, out_features=5)   # Replace with your architecture

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x

# Instantiate the model
model = NeuralNetwork().to(device)  # This moves your model to the GPU

# Example forward pass with random data
input_tensor = torch.rand(1, 10).to(device)  # This moves your data to the GPU
output_tensor = model(input_tensor)  # This runs the forward pass on the GPU
