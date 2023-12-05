import torch
import torch.nn as nn

from torch_activation_functions import ActivationFunctions as activation_functions
from config import Config

config = Config("config.ini", "DEFAULT")

class NeuralNetwork(nn.Module):
    def __init__(self, genome, refractory_factor=config.refractory_factor):
        super(NeuralNetwork, self).__init__()

        # Define device based on GPU availability
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        # Extract neurons from genome and create neuron ID to index mapping
        self.neurons = {neuron.id: neuron for neuron in genome.neuron_genes.values()}
        self.neuron_id_to_index = {neuron_id: index for index, neuron_id in enumerate(self.neurons.keys())}

        # Initialize neuron states
        num_neurons = len(self.neurons)
        self.neuron_states = torch.zeros(num_neurons, dtype=torch.float32).to(self.device)

        # Create weight matrix and bias vector
        self.weight_matrix = torch.zeros(num_neurons, num_neurons, dtype=torch.float32).to(self.device)
        self.biases = torch.zeros(num_neurons, dtype=torch.float32).to(self.device)

        for conn in genome.connection_genes.values():
            from_index = self.neuron_id_to_index[conn.from_neuron]
            to_index = self.neuron_id_to_index[conn.to_neuron]
            self.weight_matrix[from_index, to_index] = conn.weight

        for neuron_id, neuron in self.neurons.items():
            neuron_index = self.neuron_id_to_index[neuron_id]
            self.biases[neuron_index] = neuron.bias

        # Index neurons by type and store their activation functions
        self.layer_activation_functions = {}
        self.layer_indices = {"input": [], "hidden": [], "output": []}

        for neuron_id, neuron in self.neurons.items():
            neuron_index = self.neuron_id_to_index[neuron_id]
            self.layer_indices[neuron.layer].append(neuron_index)
            if neuron.layer not in self.layer_activation_functions:
                # Retrieve the activation function using getattr
                activation_func = getattr(activation_functions, neuron.activation, None)
                if activation_func is None:
                    raise ValueError(f"Activation function \"{neuron.activation}\" not found in ActivationFunctions class")
                self.layer_activation_functions[neuron.layer] = activation_func

        # Refractory factor
        self.refractory_factor = refractory_factor

    def forward(self, input_values):
        # Apply refractory factor to hidden states
        hidden_indices = self.layer_indices["hidden"]
        self.neuron_states[hidden_indices] *= self.refractory_factor

        # Assign input values to input neurons
        input_indices = self.layer_indices["input"]
        input_tensor = torch.tensor(input_values, dtype=torch.float32).to(self.device)
        self.neuron_states[input_indices] = input_tensor

        # Propagate through the network
        total_input = torch.matmul(self.neuron_states, self.weight_matrix) + self.biases

        # Apply activation functions per neuron
        for neuron_id, neuron in self.neurons.items():
            neuron_index = self.neuron_id_to_index[neuron_id]
            if neuron.layer != "input":  # Skip input layer for activations
                activation_func = getattr(activation_functions, neuron.activation, None)
                if activation_func is None:
                    raise ValueError(f"Activation function \"{neuron.activation}\" not found in ActivationFunctions class")
                self.neuron_states[neuron_index] = activation_func(total_input[neuron_index])

        # Extract and return output states
        output_indices = self.layer_indices["output"]
        return self.neuron_states[output_indices]

    def forward_batch(self, input_values):
        # Expecting input_values to be a batch of inputs, shape: [batch_size, num_inputs]
        batch_size = input_values.size(0)
        num_neurons = len(self.neurons)

        # Initialize batch neuron states
        batch_neuron_states = torch.zeros(batch_size, num_neurons, dtype=torch.float32).to(self.device)
        
        # Apply refractory factor to hidden states
        hidden_indices = self.layer_indices["hidden"]
        batch_neuron_states[:, hidden_indices] *= self.refractory_factor

        # Assign input values to input neurons
        input_indices = self.layer_indices["input"]
        input_tensor = input_values.to(self.device)  # Shape: [batch_size, num_inputs]
        batch_neuron_states[:, input_indices] = input_tensor

        # Propagate through the network
        total_input = torch.matmul(batch_neuron_states, self.weight_matrix) + self.biases

        # Apply activation functions per neuron in batch
        for neuron_id, neuron in self.neurons.items():
            neuron_index = self.neuron_id_to_index[neuron_id]
            if neuron.layer != "input":  # Skip input layer for activations
                activation_func = getattr(activation_functions, neuron.activation, None)
                if activation_func is None:
                    raise ValueError(f"Activation function \"{neuron.activation}\" not found in ActivationFunctions class")
                batch_neuron_states[:, neuron_index] = activation_func(total_input[:, neuron_index])

        # Extract and return output states for the batch
        output_indices = self.layer_indices["output"]
        return batch_neuron_states[:, output_indices]

    def reset_states(self):
        self.neuron_states = torch.zeros_like(self.neuron_states).to(self.device)
