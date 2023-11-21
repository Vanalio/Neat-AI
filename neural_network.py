import torch

from torch_activation_functions import ActivationFunctions as activation_functions

from config import Config
config = Config("config.ini", "DEFAULT")

class NeuralNetwork:
    def __init__(self, genome, input_idx_list, output_idx_list):
        self.neurons = {}  # Neuron ID -> NeuronGene
        self.connections = {}  # (From, To) -> ConnectionGene
        self.hidden_indices = {}  # Neuron ID -> Index in hidden_states
        self.input_indices = {neuron_id: idx for idx, neuron_id in enumerate(input_idx_list)}  # Neuron ID -> Index in input_tensor
        self.output_indices = {neuron_id: idx for idx, neuron_id in enumerate(output_idx_list)}  # Neuron ID -> Index in output_tensor
        self.hidden_states = None
        self.build_network(genome)

    def build_network(self, genome):
        # Add neurons and create mapping for hidden neuron indices
        #print("Building network")  # Debug #print
        hidden_neurons = [n for n in genome.neuron_genes.values() if n.layer == 'hidden' and n.enabled]
        #print("Hidden neurons:", hidden_neurons)  # Debug #print
        for idx, neuron in enumerate(hidden_neurons):
            #print(f"Neuron: {neuron.id}, Layer: {neuron.layer}")
            self.neurons[neuron.id] = neuron
            #print(f"Neuron: {neuron.id}, Index: {idx}")
            self.hidden_indices[neuron.id] = idx
            #print(f"Hidden indices: {self.hidden_indices}")

        # Add input neurons
        for neuron_id in self.input_indices:
            if neuron_id in genome.neuron_genes and genome.neuron_genes[neuron_id].enabled:
                self.neurons[neuron_id] = genome.neuron_genes[neuron_id]

        # Add output neurons
        for neuron_id in self.output_indices:
            if neuron_id in genome.neuron_genes and genome.neuron_genes[neuron_id].enabled:
                self.neurons[neuron_id] = genome.neuron_genes[neuron_id]

        # Initialize hidden states tensor
        self.hidden_states = torch.zeros(len(hidden_neurons))
        #print("Hidden states:", self.hidden_states)

        # Add connections
        for conn in genome.connection_genes.values():
            #print(f"Connection: {conn.id}, From: {conn.from_neuron}, To: {conn.to_neuron}")  # Debug #print
            if conn.enabled and conn.from_neuron in self.neurons and conn.to_neuron in self.neurons:
                #print("Connection enabled")
                self.connections[(conn.from_neuron, conn.to_neuron)] = conn
                #print(f"Connection: {conn.id}, From: {conn.from_neuron}, To: {conn.to_neuron}")  # Debug #print
        
        # wait for user input to continue
        #input("Press Enter to continue...")

    def propagate(self, input_values):
        input_tensor = torch.from_numpy(input_values).float()
        #print("Input tensor:", input_tensor)  # Debug #print
        #print("Size of input tensor:", input_tensor.size())  # Debug the size of input_tensor
        #print("Input indices:", self.input_indices)  # Debug #print for input indices
        output_tensor = torch.zeros(len(self.output_indices))
        #print("Initial output tensor:", output_tensor)  # Debug #print
        new_hidden_states = self.hidden_states.clone()
        #print("Initial hidden states:", new_hidden_states)  # Debug #print

        # Initialize dictionary for output activations
        output_activations = {neuron_id: 0.0 for neuron_id in self.output_indices}

        # Process connections from input to hidden and recurrent connections
        for (from_neuron, to_neuron), conn in self.connections.items():
            #print(f"Connection from {from_neuron} to {to_neuron}:")  # Debug #print
            # Retrieve the target neuron
            target_neuron = self.neurons[to_neuron]
            #print(f"Target neuron: {target_neuron.id}, Layer: {target_neuron.layer}")  # Debug #print

            if from_neuron in self.input_indices:
                # Retrieve the index for this neuron ID
                input_index = self.input_indices[from_neuron]
                #print("Input to hidden connection")  # Debug #print
                #print("Input neuron ID:", from_neuron)  # Debug #print
                #print("Index in input tensor:", self.input_indices[from_neuron])  # Debug #print
                # Input to hidden
                activation_input = self.compute_activation(input_tensor[input_index], conn) + target_neuron.bias
                #print(f"Input neuron: {from_neuron}, Value: {input_tensor[self.input_indices[from_neuron]]}")  # Debug #print
                activation_func = getattr(activation_functions, target_neuron.activation)
                #print(f"Activation function: {target_neuron.activation}")  # Debug #print
                new_hidden_states[self.hidden_indices[to_neuron]] += activation_func(activation_input)
                #print(f"New hidden state: {new_hidden_states[self.hidden_indices[to_neuron]]}")  # Debug #print
            elif from_neuron in self.hidden_indices:
                #print("Recurrent connection")
                # Recurrent connection (hidden to hidden)
                activation_input = self.compute_activation(self.hidden_states[self.hidden_indices[from_neuron]], conn) + target_neuron.bias
                #print(f"Hidden neuron: {from_neuron}, Value: {self.hidden_states[self.hidden_indices[from_neuron]]}")
                activation_func = getattr(activation_functions, target_neuron.activation)
                #print(f"Activation function: {target_neuron.activation}")
                if to_neuron in self.hidden_indices:
                    # Hidden to hidden connection
                    new_hidden_states[self.hidden_indices[to_neuron]] += activation_func(activation_input)
                    #print(f"Hidden to hidden connection. New hidden state for neuron {to_neuron}: {new_hidden_states[self.hidden_indices[to_neuron]]}")
                elif to_neuron in self.output_indices:
                    # Hidden to output connection
                    output_activations[to_neuron] += activation_func(activation_input)
                    #print(f"Hidden to output connection. Accumulated activation for output neuron {to_neuron}: {output_activations[to_neuron]}")
                
            # Verbose Debug #prints for each connection
            #print(f"Connection from {from_neuron} to {to_neuron}:")
            #print(f"  Weight: {conn.weight}, Bias: {target_neuron.bias}")
            #print(f"  Activation input: {activation_input}")
            #print(f"  Activation function: {target_neuron.activation}")
            #print(f"  Activation output: {activation_func(activation_input)}")

        #print("Updated hidden states:", new_hidden_states)  # Debug #print
        # Update hidden states with decay
        self.hidden_states = new_hidden_states * config.refractory_factor

        # Update output tensor with accumulated activations
        for neuron_id, activation in output_activations.items():
            output_tensor[self.output_indices[neuron_id]] = activation
            #print(f"Output neuron: {neuron_id}, Value: {activation}")

        #print("Output tensor:", output_tensor)  # Debug print
        # Convert output to NumPy array
        output_array = output_tensor.detach().numpy()
        return output_array

    def compute_activation(self, value, connection):
        # Apply the connection weight
        return value * connection.weight

    def reset_hidden_states(self):
        self.hidden_states = torch.zeros_like(self.hidden_states)
