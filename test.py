import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torchvision




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.first_layer = None
        self.layers = nn.ModuleList()
        layer_sizes = [3**i for i in range(net_magnitude, 0, -1)]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(in_features=layer_sizes[i], out_features=layer_sizes[i + 1])
            nn.init.normal_(layer.weight, mean=0, std=np.sqrt(1 / layer_sizes[i]))
            nn.init.constant_(layer.bias, 0.0)  # Initialize biases to 0
            self.layers.append(layer)

    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Dynamically create and initialize the first layer based on input size
        if self.first_layer is None:
            num_features = x.size(1)
            self.first_layer = nn.Linear(in_features=num_features, out_features=3**net_magnitude).to(device).double()
            nn.init.normal_(self.first_layer.weight, mean=0, std=np.sqrt(1 / num_features))
            nn.init.constant_(self.first_layer.bias, 0.0)  # Initialize biases to 0

        x = torch.nn.functional.tanh(self.first_layer(x))

        for layer in self.layers:
            x = torch.nn.functional.tanh(layer(x))

        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x_normalized = x / norm  # L2 normalization

        return x_normalized


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


def ciftci_loss(net, x, epsilon=0.001):
    x_perturbed = perturb_data(x, epsilon)
    output_original = net(x)
    output_perturbed = net(x_perturbed)
    loss = torch.norm(output_original - output_perturbed, p=2, dim=1).pow(2).mean()
    return loss


def perturb_data(x, epsilon=0.01):
    noise = torch.randn_like(x) * epsilon
    return x + noise


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


# Function to unnormalize and show an image (CIFAR10 specific)
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(f"Using device: {device}")
input("Press Enter to continue...")

epochs = 50
batch_size = 10000
print(f"Batch size: {batch_size}")
net_magnitude = 5

print_interval = 1


# CIFAR10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.type(torch.double)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

"""
# MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.type(torch.double)),
    transforms.Normalize((0.1307,), (0.3081,))  # Normalizing for MNIST
])

# Load the MNIST dataset
trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
"""

net = Net().to(device).double()

optimizer = optim.Adam(net.parameters())

# Interactive Evaluation Mode
user_input = input("Do you want to go into evaluation mode and find nearest neighbors? (yes/no): ").strip().lower()
if user_input == 'yes':
    net.eval()  # Set the network to evaluation mode

    # Select batch_size Images from the Dataset
    dataiter = iter(trainloader)
    images, _ = next(dataiter)
    images = images[:batch_size].to(device)

    # Get the output from the network
    with torch.no_grad():
        outputs = net(images.double())

    # Calculate pairwise distances
    pairwise_distances = torch.cdist(outputs, outputs, p=2)

    # Choose a random image index
    random_idx = np.random.randint(0, batch_size)
    distances = pairwise_distances[random_idx]

    # Find indices of the two nearest neighbors
    nearest_neighbors = distances.topk(3, largest=False)[1]
    nearest_neighbors = nearest_neighbors.cpu().numpy()

    # Exclude the selected image itself
    nearest_neighbors = nearest_neighbors[nearest_neighbors != random_idx][:2]

    # Show the selected image
    print("Selected Image:")
    imshow(torchvision.utils.make_grid(images[random_idx].cpu()))

    # Show the nearest neighbors
    print("Nearest Neighbors:")
    imshow(torchvision.utils.make_grid(images[nearest_neighbors].cpu()))

    input("Press Enter to continue...")

# Interactive Training Mode
user_input = input("Do you want to go into training mode? (yes/no): ").strip().lower()
if user_input == 'yes':
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(12, 6))

    # Subplot for 3D scatter plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Points Visualization')
    ax1.view_init(elev=20, azim=15)
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
            loss_value = ciftci_loss(net, images)
            #loss_value = repulsion_loss(output)
            print(f"Loss: {loss_value.item()}")

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            # Update loss plot
            loss_values.append(loss_value.item())
            line.set_xdata(range(len(loss_values)))
            line.set_ydata(loss_values)
            ax2.relim()  # Recalculate limits
            ax2.autoscale_view(True,True,True)

            # Plotting the 3D points
            ax1.clear()
            output_np = output.cpu().detach().numpy()  # Convert output to numpy for plotting
            ax1.scatter(output_np[:, 0], output_np[:, 1], output_np[:, 2])
            # Projecting onto XY plane (Z = 0)
            ax1.scatter(output_np[:, 0], output_np[:, 1], np.zeros_like(output_np[:, 2]), color='red')
            # Projecting onto XZ plane (Y = 0)
            ax1.scatter(output_np[:, 0], np.zeros_like(output_np[:, 1]), output_np[:, 2], color='black')
            # Projecting onto YZ plane (X = 0)
            ax1.scatter(np.zeros_like(output_np[:, 0]), output_np[:, 1], output_np[:, 2], color='green')
        
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('3D Points Visualization')
            ax1.view_init(elev=20, azim=15)
            ax1.invert_xaxis()


            plt.draw()
            plt.pause(0.1)  # Pause to update the plot

        if epoch % print_interval == 0:
            print_parameter_summary(net)
            print_gradient_norms(net)
        
        torch.save(net, 'data/model.pth')

    plt.ioff()  # Interactive mode off
    plt.show()  # Keep the plot open
