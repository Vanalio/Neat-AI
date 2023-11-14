import torch
import torch.nn as nn
import random

config = {

    "input_neurons": 24,
    "hidden_neurons": 1,
    "output_neurons": 4,
    "initial_conn_attempts": 50, # max possible connections = hidden_neurons * (input_neurons + hidden_neurons + output_neurons)
    "attempts_to_max_factor": 5,
    "refractory_factor": 1,

    "generations": 10,
    "population_size": 10,

    "elites_per_species": 2,
    "max_stagnation": 20,
    "target_species": 25,

    "compatibility_threshold": 10,
    "distance_adj_factor": 0.2,
    "disjoint_coefficient": 1,
    "excess_coefficient": 1,
    "weight_diff_coefficient": 1,
    "activation_diff_coefficient": 1,

    "allow_interspecies_mating": True,
    "interspecies_mating_count": 10,

    "keep_best_percentage": 0.5,

    "neuron_add_chance": 0.01,
    "neuron_toggle_chance": 0.0075,
    "bias_mutate_chance": 0.1,
    "bias_mutate_factor": 0.5,
    "bias_init_range": (-2, 2),

    "activation_mutate_chance": 0.1,
    "default_hidden_activation": "tanh",
    "default_output_activation": "tanh",
    "relu_clip_at": 1,

    "gene_add_chance": 0.02,
    "gene_toggle_chance": 0.001,
    "weight_mutate_chance": 0.1,
    "weight_mutate_factor": 0.5,
    "weight_init_range": (-2, 2),

    "parallelize": False,
    "parallelization": 6,

    "global_mutation_enable": False,
    "global_mutation_chance": 0.5,
    "population_save_interval": 10
}

class NeuronGene:
    def __init__(self, layer):
        self.id = IdManager.get_new_id()
        self.layer = layer
        self.activation = config["default_hidden_activation"] if self.layer == "hidden" else (config["default_output_activation"] if self.layer == "output" else "identity")
        self.bias = random.uniform(*config["bias_init_range"]) if self.layer == "output" or self.layer == "hidden" else 0
        self.enabled = True

class ConnectionGene:
    def __init__(self, from_neuron, to_neuron):
        self.id = IdManager.get_new_id()
        self.innovation_number = InnovationManager.get_new_innovation_number()
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = random.uniform(*config["weight_init_range"])
        self.enabled = True

# Example configuration (you may adjust these values as needed)
config = {
    "default_hidden_activation": "tanh",
    "default_output_activation": "identity",
    "bias_init_range": (-1, 1),
    "weight_init_range": (-1, 1)
}

# Example usage with defined sizes
input_size = 5   # Size of input vector
hidden_size = 10 # Number of hidden units
output_size = 3  # Size of output vector
sequence_length = 6 # Length of the input sequence

# Define NeuronGene and ConnectionGene objects (example)
neuron_genes = [
    NeuronGene(layer='input') for _ in range(input_size)
] + [
    NeuronGene(layer='hidden') for _ in range(hidden_size)
] + [
    NeuronGene(layer='output') for _ in range(output_size)
]

connection_genes = [
    ConnectionGene(from_neuron=neuron_genes[i].id, to_neuron=neuron_genes[j].id)
    for i in range(input_size) for j in range(input_size, input_size + hidden_size)
] + [
    ConnectionGene(from_neuron=neuron_genes[i].id, to_neuron=neuron_genes[j].id)
    for i in range(input_size, input_size + hidden_size) for j in range(input_size + hidden_size, len(neuron_genes))
]

# Create random inputs
inputs = [torch.randn(input_size, 1) for _ in range(sequence_length)]

# Initialize RNN
rnn = CustomRNNWithNeuronGenes(neuron_genes, connection_genes)
outputs = rnn(inputs)

# Print outputs
for i, out in enumerate(outputs):
    print(f"Output at time step {i}: {out.detach().numpy().ravel()}")
