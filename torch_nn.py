import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNetwork(nn.Module):
    def __init__(self, neurons, connections):
        super(CustomNetwork, self).__init__()
        self.neurons = {neuron.neuron_id: neuron for neuron in neurons}

        # Number of neurons
        num_neurons = len(neurons)

        # Initialize neuron states
        self.neuron_states = torch.zeros(num_neurons, dtype=torch.float32)

        # Create weight matrix and bias vector
        self.weight_matrix = torch.zeros(num_neurons, num_neurons, dtype=torch.float32)
        self.biases = torch.zeros(num_neurons, dtype=torch.float32)

        for conn in connections:
            self.weight_matrix[conn.from_neuron_id, conn.to_neuron_id] = conn.weight

        for neuron_id, neuron in self.neurons.items():
            self.biases[neuron_id] = neuron.bias

        # Index neurons by type and store their activation functions
        self.layer_activation_functions = {}
        self.layer_indices = {'input': [], 'hidden': [], 'output': []}

        for i, neuron in enumerate(neurons):
            self.layer_indices[neuron.layer].append(i)
            if neuron.layer not in self.layer_activation_functions:
                self.layer_activation_functions[neuron.layer] = self.activation_functions[neuron.activation]

        # Activation functions mapping
        self.activation_functions = {
            'relu': F.relu,
            'softsign': F.softsign,
            # Add other activation functions as needed
        }

    def forward(self, input_values):
        # Assign input values to input neurons
        self.neuron_states[self.layer_indices['input']] = torch.tensor(input_values, dtype=torch.float32)

        # Propagate through the network
        total_input = torch.matmul(self.neuron_states, self.weight_matrix) + self.biases

        # Apply activation functions by layer
        for layer, indices in self.layer_indices.items():
            if layer != 'input':  # Skip input layer for activations
                activation_func = self.layer_activation_functions[layer]
                self.neuron_states[indices] = activation_func(total_input[indices])

        # Extract and return output states
        return self.neuron_states[self.layer_indices['output']]

    def reset_states(self):
        self.neuron_states = torch.zeros_like(self.neuron_states)

