import torch
import torch.nn as nn
import torch.optim as optim

# Check for CUDA and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
input("Press Enter to continue...")

def clipped_relu(x):
    return torch.clamp(x, min=0, max=1)

class ModifiedArbitraryDimNet(nn.Module):
    def __init__(self):
        super(ModifiedArbitraryDimNet, self).__init__()
        # Initialize a placeholder for the first layer
        self.first_layer = None

        # Create a sequence of fully connected layers
        self.layers = nn.ModuleList()
        layer_sizes = [3**i for i in range(5, 0, -1)]
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
            self.first_layer = nn.Linear(in_features=num_features, out_features=3**5).to(device)
            nn.init.normal_(self.first_layer.weight, mean=0, std=1 / num_features ** 0.5)  # LeCun initialization
            nn.init.constant_(self.first_layer.bias, 0.1)  # Initialize biases to 0.1

        # Apply the first layer with clipped ReLU
        x = clipped_relu(self.first_layer(x))

        # Iterate over all layers and apply them with clipped ReLU
        for layer in self.layers:
            x = layer(x)
            x = clipped_relu(x)

        return x

def grid_based_loss(output, grid_size):
    # Scale output coordinates to sub-box indices
    sub_box_indices = (output * (grid_size - 1)).clamp(0, grid_size - 1)

    # Calculate expected points per sub-box for a uniform distribution
    expected_per_box = output.size(0) / (grid_size ** 3)

    # Create a tensor to hold sub-box counts
    sub_box_counts = torch.zeros((grid_size, grid_size, grid_size), device=output.device)

    # Soft assignment to sub-boxes (differentiable)
    for i in range(sub_box_indices.size(0)):
        # Find nearest neighbors in the grid and assign softly
        neighbors = torch.floor(sub_box_indices[i]).long()
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbor = neighbors + torch.tensor([dx, dy, dz], device=output.device)
                    if torch.all(neighbor >= 0) and torch.all(neighbor < grid_size):
                        # Soft assignment based on distance
                        distance = torch.norm(sub_box_indices[i] - neighbor.float())
                        weight = 1 / (distance + 1e-6)
                        sub_box_counts[tuple(neighbor)] += weight

    # Calculate the loss
    excess_counts = torch.where(sub_box_counts > expected_per_box, 
                                sub_box_counts - expected_per_box, 
                                torch.zeros_like(sub_box_counts))
    loss = torch.sum(excess_counts ** 2)

    return loss

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

net = ModifiedArbitraryDimNet().to(device)
optimizer = optim.Adam(net.parameters())

# Set the grid size
grid_size = 7  # approx. cubic root of batch size
print_interval = 5  # How often to print the parameter summary

# Training loop
for epoch in range(3000):
    input_tensor = torch.rand(256, 512, 512, 3).to(device)

    # Forward pass
    output = net(input_tensor)

    # Compute grid-based loss
    loss = grid_based_loss(output, grid_size)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss, parameter summary, and gradient norms at specified intervals
    if epoch % print_interval == 0:
        print(f"Epoch {epoch} - Loss: {loss.item()}")
        print_parameter_summary(net)
        print_gradient_norms(net)
        print(f"Output mean: {output.mean().item()}, std: {output.std().item()}")

