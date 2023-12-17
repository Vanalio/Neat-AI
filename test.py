import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            nn.init.uniform_(layer.weight, -0.0001, 0.0001)  # Uniform initialization for weights between -0.01 and 0.01
            nn.init.uniform_(layer.bias, -0.0001, 0.0001)      # Uniform initialization for biases between -0.1 and 0.1
            self.layers.append(layer)

    def forward(self, x):
        # Flatten the input
        #print("flattening")
        x = x.view(x.size(0), -1)
        #print(f"x size {x.size()}")
        #print(f"x: {x}")

        # Dynamically create and initialize the first layer based on input size
        if self.first_layer is None:
            num_features = x.size(1)
            self.first_layer = nn.Linear(in_features=num_features, out_features=3**5).to(x.device)
            nn.init.uniform_(self.first_layer.weight, -0.0001, 0.0001)  # Uniform initialization for weights between -0.01 and 0.01
            nn.init.uniform_(self.first_layer.bias, -0.0001, 0.0001)      # Uniform initialization for biases between -0.1 and 0.1

        # Apply the first layer with clipped ReLU
        #print("apply clipped_relu to first layer")
        x = clipped_relu(self.first_layer(x))
        #print(f"x size {x.size()}")
        #print(f"x: {x}")

        # Iterate over all layers and apply them with clipped ReLU
        for layer in self.layers:
            #print(f"compute layer {layer}")
            x = layer(x)
            #print(f"x size {x.size()}")
            #print(f"x: {x}")
            #print(f"apply clipped_relu to x")
            x = clipped_relu(x)
            #print(f"x size {x.size()}")
            #print(f"x: {x}")

        return x

def grid_based_loss(output, grid_size):
    # Initialize the count for each sub-box, ensuring it requires gradients
    sub_box_counts = torch.zeros((grid_size, grid_size, grid_size), requires_grad=True)

    # Classify points into sub-boxes
    sub_box_indices = (output * grid_size).floor().int()
    sub_box_indices = torch.clamp(sub_box_indices, 0, grid_size - 1)

    # Count points in each sub-box without using in-place operations
    for i in range(sub_box_indices.size(0)):
        x, y, z = sub_box_indices[i]
        sub_box_counts = sub_box_counts.clone()  # Clone to avoid in-place modification
        sub_box_counts[x, y, z] = sub_box_counts[x, y, z] + 1

    # Calculate expected points per sub-box for a uniform distribution
    expected_per_box = output.size(0) / (grid_size ** 3)

    # Calculate the loss
    excess_counts = torch.where(sub_box_counts > expected_per_box, 
                                sub_box_counts - expected_per_box, 
                                torch.zeros_like(sub_box_counts))
    loss = torch.sum(excess_counts ** 2)

    return loss

net = ModifiedArbitraryDimNet().to(device)
optimizer = optim.Adam(net.parameters())

# Set the grid size
grid_size = 4  # Adjust as needed

for _ in range(3000):
    #for name, param in net.named_parameters():
        #if param.requires_grad:
            #print(name, "gradient", param.grad)
            #print(name, "data", param.data)

    input_tensor = torch.rand(8, 64, 64, 3).to(device)

    # Forward pass
    output = net(input_tensor)

    # Compute grid-based loss
    loss = grid_based_loss(output, grid_size)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Input: {input_tensor}")
    print(f"Output: {output}")
    print(f"Loss: {loss.item()}")
