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

        # Index neurons by type
        self.input_indices = [i for i, neuron in enumerate(neurons) if neuron.layer == 'input']
        self.output_indices = [i for i, neuron in enumerate(neurons) if neuron.layer == 'output']

        # Activation functions mapping
        self.activation_functions = {
            'relu': F.relu,
            'softsign': F.softsign,
            # Add other activation functions as needed
        }

    def forward(self, input_values):
        # Assign input values to input neurons
        self.neuron_states[self.input_indices] = torch.tensor(input_values, dtype=torch.float32)

        # Propagate through the network
        total_input = torch.matmul(self.neuron_states, self.weight_matrix) + self.biases

        # Apply activation functions
        for i, neuron_id in enumerate(self.neurons):
            neuron = self.neurons[neuron_id]
            activation_func = self.activation_functions.get(neuron.activation, lambda x: x)
            self.neuron_states[neuron_id] = activation_func(total_input[neuron_id])

        # Extract and return output states
        return self.neuron_states[self.output_indices]

    def reset_states(self):
        self.neuron_states = torch.zeros_like(self.neuron_states)
