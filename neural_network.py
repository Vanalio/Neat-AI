import torch

from torch_activation_functions import ActivationFunctions as activation_functions

from config import Config

config = Config("config.ini", "DEFAULT")


class NeuralNetwork:
    def __init__(self, genome, input_ids, output_ids):
        self.neurons = {}
        self.connections = {}
        self.hidden_indices = {}
        self.input_indices = {neuron_id: idx for idx, neuron_id in enumerate(input_ids)}
        self.output_indices = {
            neuron_id: idx for idx, neuron_id in enumerate(output_ids)
        }
        self.hidden_states = None
        self.build_network(genome)

    def build_network(self, genome):

        hidden_neurons = [
            n for n in genome.neuron_genes.values() if n.layer == "hidden" and n.enabled
        ]

        for idx, neuron in enumerate(hidden_neurons):

            self.neurons[neuron.id] = neuron

            self.hidden_indices[neuron.id] = idx

        for neuron_id in self.input_indices:
            if (
                neuron_id in genome.neuron_genes
                and genome.neuron_genes[neuron_id].enabled
            ):
                self.neurons[neuron_id] = genome.neuron_genes[neuron_id]

        for neuron_id in self.output_indices:
            if (
                neuron_id in genome.neuron_genes
                and genome.neuron_genes[neuron_id].enabled
            ):
                self.neurons[neuron_id] = genome.neuron_genes[neuron_id]

        self.hidden_states = torch.zeros(len(hidden_neurons))

        for conn in genome.connection_genes.values():

            if (
                conn.enabled
                and conn.from_neuron in self.neurons
                and conn.to_neuron in self.neurons
            ):

                self.connections[(conn.from_neuron, conn.to_neuron)] = conn

    def propagate(self, input_values):
        input_tensor = torch.from_numpy(input_values).float()

        output_tensor = torch.zeros(len(self.output_indices))

        new_hidden_states = self.hidden_states.clone()
        output_accumulation = {neuron_id: 0.0 for neuron_id in self.output_indices}

        for (from_neuron, to_neuron), conn in self.connections.items():
            activation_input = None

            if from_neuron in self.input_indices:
                input_index = self.input_indices[from_neuron]
                activation_input = input_tensor[input_index] * conn.weight

            elif from_neuron in self.hidden_indices:
                hidden_index = self.hidden_indices[from_neuron]
                activation_input = self.hidden_states[hidden_index] * conn.weight

            if activation_input is not None:
                target_neuron = self.neurons[to_neuron]
                activation_func = getattr(
                    activation_functions, target_neuron.activation
                )
                activation_output = activation_func(
                    activation_input + target_neuron.bias
                )

                if to_neuron in self.hidden_indices:
                    hidden_index = self.hidden_indices[to_neuron]
                    new_hidden_states[hidden_index] += activation_output

                elif to_neuron in self.output_indices:
                    output_accumulation[to_neuron] += activation_output.item()

        self.hidden_states = new_hidden_states * config.refractory_factor

        for neuron_id, accumulated_input in output_accumulation.items():
            output_neuron = self.neurons[neuron_id]
            output_activation_func = getattr(
                activation_functions, output_neuron.activation
            )
            accumulated_input_tensor = torch.tensor(
                [accumulated_input], dtype=torch.float32
            )
            output_tensor[self.output_indices[neuron_id]] = output_activation_func(
                accumulated_input_tensor
            )

        output_array = output_tensor.detach().numpy()
        return output_array

    def reset_hidden_states(self):
        self.hidden_states = torch.zeros_like(self.hidden_states)
