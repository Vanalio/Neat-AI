import torch
import torch.nn as nn

from torch_activation_functions import ActivationFunctions as activation_functions

from config import Config

config = Config("config.ini", "DEFAULT")

class NeuralNetwork(nn.Module):
    def __init__(self, genome, refractory_factor=config.refractory_factor):
        super(NeuralNetwork, self).__init__()

        self.device = torch.device("cpu")

        # Create neuron ID to index mapping and filter enabled neurons
        self.neurons = {neuron.id: neuron for neuron in genome.neuron_genes.values() if neuron.enabled}
        self.neuron_id_to_index = {neuron_id: index for index, neuron_id in enumerate(self.neurons.keys())}

        self.layer_indices = {'input': [], 'hidden': [], 'output': []}
        self._initialize_layers()

        # Initialize weight matrix and biases
        self._initialize_weights_and_biases(genome.connection_genes.values())

        # Refractory factor
        self.refractory_factor = refractory_factor

        # Neuron states initialization
        self.neuron_states = None

    def _initialize_layers(self):
        for neuron_id, neuron in self.neurons.items():
            neuron_index = self.neuron_id_to_index[neuron_id]
            self.layer_indices[neuron.layer].append(neuron_index)

    def _initialize_weights_and_biases(self, connection_genes):
        num_neurons = len(self.neurons)
        self.weight_matrix = torch.zeros(num_neurons, num_neurons, dtype=torch.float32, device=self.device)
        self.biases = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)

        for conn in connection_genes:
            if not conn.enabled:
                continue

            from_neuron = conn.from_neuron
            to_neuron = conn.to_neuron

            if from_neuron not in self.neurons or to_neuron not in self.neurons:
                # Skipping connections that refer to disabled or non-existing neurons
                continue

            try:
                from_index = self.neuron_id_to_index[from_neuron]
                to_index = self.neuron_id_to_index[to_neuron]
            except KeyError as e:
                print(f"KeyError in connection gene. KeyError: {e.args[0]}")
                print(f"From Neuron: {from_neuron}, To Neuron: {to_neuron}")
                continue  # Skip this iteration if a KeyError occurs

            self.weight_matrix[from_index, to_index] = conn.weight
            self.biases[to_index] = self.neurons[to_neuron].bias

    def forward_batch(self, input_values):
        if self.neuron_states is None:
            self._initialize_neuron_states(input_values.size(0))

        self._apply_refractory_factor()
        self._assign_input_values(input_values)

        total_input = torch.matmul(self.neuron_states, self.weight_matrix) + self.biases
        self._apply_activation_functions(total_input)

        return self.neuron_states[:, self.layer_indices["output"]]

    def _initialize_neuron_states(self, batch_size):
        num_neurons = len(self.neurons)
        self.neuron_states = torch.zeros(batch_size, num_neurons, dtype=torch.float32, device=self.device)

    def _apply_refractory_factor(self):
        self.neuron_states[:, self.layer_indices["hidden"]] *= self.refractory_factor

    def _assign_input_values(self, input_values):
        self.neuron_states[:, self.layer_indices["input"]] = input_values.to(self.device)

    def _apply_activation_functions(self, total_input):
        for neuron_id, neuron in self.neurons.items():
            if neuron.layer == "input":
                continue
            neuron_index = self.neuron_id_to_index[neuron_id]

            # Try to get the activation function from custom ActivationFunctions class
            activation_func = getattr(activation_functions, neuron.activation, None)

            # If not found in custom, try to get it from torch.nn.functional
            if activation_func is None:
                activation_func = getattr(torch.nn.functional, neuron.activation, None)
            
            # Raise error if still not found
            if activation_func is None:
                raise ValueError(f"Activation function \"{neuron.activation}\" not found")
            
            self.neuron_states[:, neuron_index] = activation_func(total_input[:, neuron_index])

    def reset_states(self):
        self.neuron_states = None
