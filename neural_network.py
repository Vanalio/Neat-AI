import torch

from torch_activation_functions import ActivationFunctions as activation_functions

from config import Config
config = Config("config.ini", "DEFAULT")

class NeuralNetwork:
    def __init__(self, genome, input_indices, output_indices):
        self.neurons = {}  # Neuron ID -> NeuronGene
        self.connections = {}  # (From, To) -> ConnectionGene
        self.hidden_indices = {}  # Neuron ID -> Index in hidden_states
        self.input_indices = input_indices  # Neuron ID -> Index in input_tensor
        self.output_indices = output_indices  # Neuron ID -> Index in output_tensor
        self.hidden_states = None
        self.build_network(genome)

    def build_network(self, genome):
        # Add neurons and create mapping for hidden neuron indices
        hidden_neurons = [n for n in genome.neuron_genes.values() if n.layer == 'hidden' and n.enabled]
        for idx, neuron in enumerate(hidden_neurons):
            self.neurons[neuron.id] = neuron
            self.hidden_indices[neuron.id] = idx

        # Initialize hidden states tensor
        self.hidden_states = torch.zeros(len(hidden_neurons))

        # Add connections
        for conn in genome.connection_genes.values():
            if conn.enabled and conn.from_neuron in self.neurons and conn.to_neuron in self.neurons:
                self.connections[(conn.from_neuron, conn.to_neuron)] = conn

    def propagate(self, input_values):
        input_tensor = torch.from_numpy(input_values).float()
        output_tensor = torch.zeros(len(self.output_indices))
        new_hidden_states = self.hidden_states.clone()

        # Process connections from input to hidden and recurrent connections
        for (from_neuron, to_neuron), conn in self.connections.items():
            if from_neuron in self.input_indices:
                # Input to hidden
                activation_func = getattr(activation_functions, self.neurons[to_neuron].activation)
                new_hidden_states[self.hidden_indices[to_neuron]] += self.compute_activation(activation_func(input_tensor[self.input_indices[from_neuron]]), conn)
            elif from_neuron in self.hidden_indices:
                # Recurrent connection (hidden to hidden)
                activation_func = getattr(activation_functions, self.neurons[to_neuron].activation)
                new_hidden_states[self.hidden_indices[to_neuron]] += self.compute_activation(activation_func(self.hidden_states[self.hidden_indices[from_neuron]]), conn)

        # Update hidden states with decay
        self.hidden_states = new_hidden_states * config.refractory_factor

        # Compute output neuron activations
        for neuron_id, neuron in self.neurons.items():
            if neuron.layer == 'output':
                output_activation = getattr(activation_functions, neuron.activation)
                output_value = sum(output_activation(self.compute_activation(self.new_hidden_states[self.hidden_indices[from_neuron]] if from_neuron in self.hidden_indices else input_tensor[self.input_indices[from_neuron]], conn))
                                for (from_neuron, to_neuron), conn in self.connections.items()
                                if to_neuron == neuron_id)
                output_tensor[self.output_indices[neuron_id]] = output_value

        # Convert output to NumPy array
        output_array = output_tensor.detach().numpy()
        return output_array

    def reset_hidden_states(self):
        self.hidden_states = torch.zeros_like(self.hidden_states)

    def compute_activation(self, value, connection):
        # Apply the connection weight
        return value * connection.weight

