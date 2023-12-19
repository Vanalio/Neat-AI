import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.first_layer = None
        self.layers = nn.ModuleList()
        layer_sizes = [3**i for i in range(net_magnitude, 0, -1)]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
            nn.init.normal_(layer.weight, mean=0, std=np.sqrt(0.1 / layer_sizes[i]))
            nn.init.constant_(layer.bias, 0.1)  # Initialize biases to 0.1
            self.layers.append(layer)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Dynamically create and initialize the first layer based on input size
        if self.first_layer is None:
            num_features = x.size(1)
            self.first_layer = nn.Linear(in_features=num_features, out_features=3**net_magnitude).to(device).double()
            nn.init.normal_(self.first_layer.weight, mean=0, std=np.sqrt(0.1 / num_features))
            nn.init.constant_(self.first_layer.bias, 0.1)  # Initialize biases to 0.1

        # Apply the first layer with clipped ReLU
        x = clipped_relu(self.first_layer(x))

        # Iterate over all layers and apply them with clipped ReLU
        for layer in self.layers:
            x = clipped_relu(layer(x))

        return x


def clipped_relu(x):
    return torch.clamp(x, min=0, max=1)


def repulsion_loss(outputs):
    # Ensure outputs are in double precision
    outputs = outputs.double() 

    # Calculate pairwise distances (Euclidean)
    pairwise_distances = torch.cdist(outputs, outputs, p=2)

    # Increase penalty for small distances (especially for identical points)
    # Example: using a smaller denominator for the exponential's exponent
    repulsion = torch.exp(-pairwise_distances / 0.5)  # Adjust this value as needed

    # Sum the penalties, avoiding self-comparison (diagonal elements)
    mask = torch.eye(len(outputs), device=outputs.device).bool()
    repulsion = repulsion.masked_fill(mask, 0)
    
    return repulsion.sum() / len(outputs)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Using device: {device}")
input("Press Enter to continue...")

epochs = 50
batch_size = 600
print(f"Batch size: {batch_size}")
net_magnitude = 9

print_interval = 1


#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Lambda(lambda x: x.type(torch.double)),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])
#trainset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
#trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.type(torch.double)),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing for MNIST
])

# Load the MNIST dataset
trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


net = Net().to(device).double()

optimizer = optim.Adam(net.parameters())


plt.ion()  # Turn on interactive mode
fig = plt.figure(figsize=(12, 6))

# Subplot for 3D scatter plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D Points Visualization')
ax1.view_init(elev=20, azim=20)
ax1.invert_xaxis()


# Subplot for loss plot
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Batch')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Over Batches')
line, = ax2.plot([], [], 'r-', label='Repulsion Loss')
ax2.legend()

loss_values = []  # Initialize the list to store loss values


# Training loop
for epoch in range(epochs):
    print(f"Starting epoch {epoch}:")
    for images, _ in trainloader:
        images = images.to(device).double()
        output = net(images)

        print(f"Output size: {output.size()}")
        print(f"Output first 10 items: {output[:10]}")
        print("Calculating loss")
        loss_value = repulsion_loss(output)
        print(f"Loss: {loss_value.item()}")

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # Update loss plot
        loss_value = repulsion_loss(output).item()  # Get loss as a Python float
        loss_values.append(loss_value)
        line.set_xdata(range(len(loss_values)))
        line.set_ydata(loss_values)
        ax2.relim()  # Recalculate limits
        ax2.autoscale_view(True,True,True)

        # Plotting the 3D points
        ax1.clear()
        output_np = output.cpu().detach().numpy()  # Convert output to numpy for plotting
        ax1.scatter(output_np[:, 0], output_np[:, 1], output_np[:, 2])
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Points Visualization')
        ax1.view_init(elev=20, azim=20)
        ax1.invert_xaxis()


        plt.draw()
        plt.pause(0.1)  # Pause to update the plot

    if epoch % print_interval == 0:
        print_parameter_summary(net)
        print_gradient_norms(net)
    
    torch.save(net, 'model.pth')

plt.ioff()  # Interactive mode off
plt.show()  # Keep the plot open
