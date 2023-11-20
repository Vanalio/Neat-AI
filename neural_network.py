import torch
import torch.nn as nn

from torch_activation_functions import ActivationFunctions as activation_functions


class NeuralNetwork(nn.Module):
    def __init__(self, genome):
        super(NeuralNetwork, self).__init__()
        self.genome = genome
        self.neuron_states = {gene.id: torch.zeros(1) for gene in genome.neuron_genes.values() if gene.enabled}
        self.weights = None
        self.biases = None
        self.input_neuron_mapping = None
        self.input_neuron_mapping = {
            neuron_id: idx 
            for idx, neuron_id in enumerate(
                sorted(
                    neuron_id 
                    for neuron_id, neuron in genome.neuron_genes.items() 
                    if neuron.layer == "input" and neuron.enabled
                )
            )
        }
        #print(f"Input neuron mapping: {self.input_neuron_mapping}")
        #print(f"Creating network for genome {genome.id}...")
        self._create_network()
        #print(f"Network created for genome {genome.id}")
        genome.network = self

    def _create_network(self):
        self.weights = nn.ParameterDict({
            f"{gene.from_neuron}_{gene.to_neuron}": nn.Parameter(torch.tensor(gene.weight, dtype=torch.float32))
            for gene in self.genome.connection_genes.values() if gene.enabled
        })
        self.biases = nn.ParameterDict({
            f"bias_{gene.id}": nn.Parameter(torch.tensor(gene.bias, dtype=torch.float32))
            for gene in self.genome.neuron_genes.values() if gene.enabled
        })

    def print_neuron_info(self):
        print("Neuron Information Snapshot:")
        for neuron_id, neuron_gene in self.genome.neuron_genes.items():
            if neuron_gene.enabled:
                bias_key = f"bias_{neuron_id}"
                bias_value = self.biases[bias_key].item() if bias_key in self.biases else "No bias"
                print(f"Neuron ID: {neuron_id}, Layer: {neuron_gene.layer}, Activation: {neuron_gene.activation}, Bias: {bias_value}")

    def reset_neuron_states(self):
        self.neuron_states = {neuron_id: torch.zeros(1) for neuron_id, neuron in self.genome.neuron_genes.items() if neuron.enabled}

    def forward(self, input):
        if input.shape[0] != len(self.input_neuron_mapping):
            raise ValueError(f"Input size mismatch. Expected {len(self.input_neuron_mapping)}, got {input.shape[0]}")

        # Reset neuron states at the start of each forward pass
        self.reset_neuron_states()

        # Set the states of input neurons
        for neuron_id, idx in self.input_neuron_mapping.items():
            if self.genome.neuron_genes[neuron_id].enabled:
                self.neuron_states[neuron_id] = input[idx]

        # Propagate signals through connections
        for gene in self.genome.connection_genes.values():
            if gene.enabled and self.genome.neuron_genes[gene.from_neuron].enabled and self.genome.neuron_genes[gene.to_neuron].enabled:
                weight = self.weights[f"{gene.from_neuron}_{gene.to_neuron}"]
                from_neuron_id = gene.from_neuron
                to_neuron_id = gene.to_neuron
                self.neuron_states[to_neuron_id] += weight * self.neuron_states[from_neuron_id]

        # Apply activation functions and biases
        for neuron_id, neuron_gene in self.genome.neuron_genes.items():
            if neuron_gene.enabled:
                activation_function = getattr(activation_functions, neuron_gene.activation)
                bias_key = f"bias_{neuron_id}"
                if bias_key in self.biases:
                    self.neuron_states[neuron_id] = self.neuron_states[neuron_id] + self.biases[bias_key]
                self.neuron_states[neuron_id] = activation_function(self.neuron_states[neuron_id])

        # Gather output neuron states
        output_neurons = [neuron_id for neuron_id in self.neuron_states if self.genome.neuron_genes[neuron_id].layer == "output"]
        output = torch.cat([self.neuron_states[neuron_id] for neuron_id in output_neurons if self.genome.neuron_genes[neuron_id].enabled])

        return output
