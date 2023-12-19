import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.type(torch.double)),  # Cast to double
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing the images
])

# Check for CUDA and set device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(f"Using device: {device}")
input("Press Enter to continue...")

def clipped_relu(x):
    return torch.clamp(x, min=0, max=1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Initialize a placeholder for the first layer
        self.first_layer = None

        # Create a sequence of fully connected layers
        self.layers = nn.ModuleList()
        layer_sizes = [3**i for i in range(net_magnitude, 0, -1)]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
            nn.init.normal_(layer.weight, mean=0, std=1 / layer_sizes[i] ** 0.5)  # LeCun initialization
            nn.init.constant_(layer.bias, 0.1)  # Initialize biases to 0.1
            self.layers.append(layer)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Dynamically create and initialize the first layer based on input size
        if self.first_layer is None:
            num_features = x.size(1)
            self.first_layer = nn.Linear(in_features=num_features, out_features=3**net_magnitude).to(device).double()
            nn.init.normal_(self.first_layer.weight, mean=0, std=1 / num_features ** 0.5)  # LeCun initialization
            nn.init.constant_(self.first_layer.bias, 0.1)  # Initialize biases to 0.1

        # Apply the first layer with clipped ReLU
        x = clipped_relu(self.first_layer(x))

        # Iterate over all layers and apply them with clipped ReLU
        for layer in self.layers:
            x = layer(x)
            x = clipped_relu(x)

        return x

def loss(output):
    ...

def print_parameter_summary(model):
    print("Parameter Summary:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name} - Mean: {param.data.mean().item()}, Std: {param.data.std().item()}")

def print_gradient_norms(model):
    print("Gradient Norms:")
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            print(f"{name} - Grad Norm: {param_norm.item()}")
    total_norm = total_norm ** 0.5
    print(f"Total Norm: {total_norm}")

def visualize_images(images):
    # Number of images
    num_images = len(images)

    # Calculate the number of rows and columns to display the images
    num_cols = int(np.sqrt(num_images))
    num_rows = (num_images + num_cols - 1) // num_cols

    # Set up the matplotlib figure and axes
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axs = axs.flatten()  # Flatten to 1D array for easy indexing

    # Loop through each image
    for i in range(num_images):
        img = images[i].cpu().numpy()  # Convert to numpy array
        img = np.transpose(img, (1, 2, 0))  # Rearrange dimensions from CHW to HWC
        img = img * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))  # Unnormalize
        img = img.clip(0, 1)  # Ensure the image is between 0 and 1

        axs[i].imshow(img)
        axs[i].axis('off')  # Turn off axis

    # Hide any unused subplots
    for i in range(num_images, len(axs)):
        axs[i].axis('off')

    plt.show()

# Set the grid size
batch_size = 64
print(f"Batch size: {batch_size}")

net_magnitude = 9

# Download and load the training data
trainset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=grid_size**3, shuffle=True)

net = Net().to(device).double()  # Cast to double

optimizer = optim.Adam(net.parameters())

print_interval = 1  # How often to print the parameter summary

# Training loop
for epoch in range(2):
    print(f"Starting epoch {epoch}:")
    for images, _ in trainloader:
        images = images.to(device).double()  # Move data to GPU and cast to double

        # Forward pass
        #print("Forward pass")
        output = net(images)
        print(f"Output size: {output.size()}")
        print(f"Output first 3 items: {output[:3]}")
        #print("Forwarded output")

        # Calculate loss
        print("Calculating loss")
        loss = grid_based_loss(output, grid_size)
        print(f"Loss: {loss.item()}")

        # Backward pass and optimization
        #print("Backward pass")
        #print("Zeroing gradients")
        optimizer.zero_grad()
        #print("Zeroed gradients")
        #print("Backwarding loss")
        loss.backward()
        #print("Backwarded loss")
        #print("Stepping optimizer")
        optimizer.step()
        #print("Stepped optimizer")

    # Print loss, parameter summary, and gradient norms at specified intervals
    if epoch % print_interval == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")
        print_parameter_summary(net)
        print_gradient_norms(net)

# Save the entire model
torch.save(net, 'model.pth')

'''
# Load the entire model
model = torch.load('model.pth')

# Load a single batch of data
images, _ = next(iter(trainloader))
images = images.to(device).double()

# Forward pass to get outputs
with torch.no_grad():
    output = model(images)

# Determine subcube/subbox assignments
subcube_assignments = determine_subcube_assignments(output, grid_size)

# Convert subcube assignments to a format suitable for indexing
subcube_assignments = subcube_assignments[:, 0] * (grid_size ** 2) + subcube_assignments[:, 1] * grid_size + subcube_assignments[:, 2]

# Randomly select a subcube/subbox
selected_subcube = random.choice(subcube_assignments.unique())

# Get images corresponding to the selected subcube
selected_images = images[subcube_assignments == selected_subcube]

# Visualization code (using matplotlib or similar)
# You need to implement visualize_images()
visualize_images(selected_images.cpu())
'''