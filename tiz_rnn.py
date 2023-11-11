import multiprocessing
import numpy as np
import random
from graphviz import Digraph


# Set the parameters
GEN = 5 # Number of generations
INPUT_NODES = 1
HIDDEN_NODES = 15
OUTPUT_NODES = 2
CONNECTIONS = 30 # n(i+o+n)
CONN_ATTEMPT_RATIO = 6
TIME_STEPS = 1200
BIAS_RANGE = (-0.1, 0.1)
WEIGHT_RANGE = (-0.1, 0.1)

ACTIVATION_FUNCTIONS = {
    "relu": lambda x: np.maximum(0, x),
    "leaky_relu": lambda x: np.where(x > 0, x, x * 0.01),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "linear": lambda x: x,
    "softplus": lambda x: np.log1p(np.exp(x)),
    "abs": np.abs,
    "clipped_relu": lambda x: np.clip(x, 0, 1)  # Clipped ReLU capped at 1
}

# Set the activation functions for hidden and output nodes
HIDDEN_ACTIVATION = "clipped_relu"
OUTPUT_ACTIVATION = "linear"

class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights = {}
        self.biases = {}
        self.connections = []
        self.state = np.zeros(self.hidden_nodes)  # State of hidden nodes for RNN behavior
        self.initialize_biases()  # Initialize biases for all nodes

    def print_attributes(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    def initialize_biases(self):
        # Initialize biases for hidden nodes
        for i in range(self.hidden_nodes):
            self.biases[f"hidden_{i}"] = random.uniform(*BIAS_RANGE)
        # Initialize biases for output nodes
        for i in range(self.output_nodes):
            self.biases[f"output_{i}"] = random.uniform(*BIAS_RANGE)

    def add_connection(self, source, target):
        if (source, target) not in self.connections:
            self.connections.append((source, target))
            self.weights[(source, target)] = random.uniform(*WEIGHT_RANGE)
            return True
        return False  # Return False if the connection was not added

    def propagate(self, input_data):
        # Initialize node values
        values = {"input": input_data, "hidden": self.state.copy(), "output": np.zeros(self.output_nodes)}
        
        # Propagation from input nodes to hidden nodes
        for i in range(INPUT_NODES):
            for j in range(HIDDEN_NODES):
                connection = (f"input_{i}", f"hidden_{j}")
                if connection in self.connections:
                    values["hidden"][j] += values["input"][i] * self.weights[connection]
        
        # Apply activation and bias to hidden nodes
        values["hidden"] = ACTIVATION_FUNCTIONS[HIDDEN_ACTIVATION](values["hidden"] + self.biases.get("hidden", 0))

        # Propagation within hidden nodes (including recurrent connections)
        for source in range(HIDDEN_NODES):
            for target in range(HIDDEN_NODES):
                connection = (f"hidden_{source}", f"hidden_{target}")
                if connection in self.connections:
                    values["hidden"][target] += values["hidden"][source] * self.weights[connection]

        # Propagation from hidden to output nodes
        for j in range(HIDDEN_NODES):
            for k in range(OUTPUT_NODES):
                connection = (f"hidden_{j}", f"output_{k}")
                if connection in self.connections:
                    values["output"][k] += values["hidden"][j] * self.weights[connection]
        
        # Apply activation and bias to output nodes
        values["output"] = ACTIVATION_FUNCTIONS[OUTPUT_ACTIVATION](values["output"] + self.biases.get("output", 0))

        # Update hidden state for the next timestep
        self.state = values["hidden"]
        return values["output"]
    
    def find_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not any(connection[0] == start for connection in self.connections):
            return []
        paths = []
        for connection in self.connections:
            if connection[0] == start:
                if connection[1] not in path:
                    newpaths = self.find_paths(connection[1], end, path)
                    for newpath in newpaths:
                        paths.append(newpath)
        return paths
    
    def plot_network(self):
        # Create a Graphviz directed graph
        dot = Digraph(comment='Neural Network')

        # Initialize node connection count
        connection_count = {f"input_{i}": 0 for i in range(self.input_nodes)}
        connection_count.update({f"hidden_{i}": 0 for i in range(self.hidden_nodes)})
        connection_count.update({f"output_{i}": 0 for i in range(self.output_nodes)})

        # Count the connections for each node
        for source, target in self.connections:
            connection_count[source] += 1
            connection_count[target] += 1

        # Function to determine node size based on connection count
        def get_node_size(count):
            base_size = 0.25  # Base size for nodes with zero connections
            return str(base_size + 0.125 * count)

        # Add nodes with sizes based on the number of connections
        for i in range(self.input_nodes):
            size = get_node_size(connection_count[f"input_{i}"])
            dot.node(f"input_{i}", 'Input\n' + f"input_{i}", color='green', shape='circle', width=size, height=size)
        for i in range(self.hidden_nodes):
            size = get_node_size(connection_count[f"hidden_{i}"])
            dot.node(f"hidden_{i}", 'Hidden\n' + f"hidden_{i}", color='skyblue', shape='circle', width=size, height=size)
        for i in range(self.output_nodes):
            size = get_node_size(connection_count[f"output_{i}"])
            dot.node(f"output_{i}", 'Output\n' + f"output_{i}", color='orange', shape='circle', width=size, height=size)

        # Find all paths from any input to any output
        all_paths = []
        for input_node in range(self.input_nodes):
            for output_node in range(self.output_nodes):
                all_paths.extend(self.find_paths(f"input_{input_node}", f"output_{output_node}"))

        # Flatten the list of paths to get a list of edges that are part of these paths
        edges_in_paths = set()
        for path in all_paths:
            edges_in_paths.update((path[i], path[i + 1]) for i in range(len(path) - 1))

        # Add edges with varying thickness
        for source, target in self.connections:
            penwidth = '4.0' if (source, target) in edges_in_paths else '0.25'
            dot.edge(source, target, color='black', penwidth=penwidth)

        # Render the graph to a file
        dot.render('network.gv', view=True)

def create_and_run_network(input_data, scaling_factor):
    # Create network
    network = Network(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES)

    # Add random connections with the constraints
    connections_added = 0
    total_net_conn_tries = 0
    max_conn_tries = CONNECTIONS * CONN_ATTEMPT_RATIO

    while connections_added < CONNECTIONS and max_conn_tries > 0:
        source_layer = random.choice(["input", "hidden"])
        if source_layer == "input":
            source_node = f"{source_layer}_{random.randint(0, INPUT_NODES - 1)}"
            target_layer = "hidden"
        else:
            source_node = f"{source_layer}_{random.randint(0, HIDDEN_NODES - 1)}"
            target_layer = random.choice(["hidden", "output"])
        if target_layer == "hidden":
            target_node = f"{target_layer}_{random.randint(0, HIDDEN_NODES - 1)}"
        else:
            target_node = f"{target_layer}_{random.randint(0, OUTPUT_NODES - 1)}"
        
        connection_try = network.add_connection(source_node, target_node)
        
        if connection_try:
            connections_added += 1
        
        max_conn_tries -= 1
        total_net_conn_tries += 1
        
    # Propagate the input through the network and gather the output at the last timestep
    for t in range(TIME_STEPS):
        output_at_t = network.propagate(input_data[t])
    final_output = output_at_t * scaling_factor
    scaled_last_timestep = input_data[-1] * scaling_factor
    final_state = network.print_attributes()

    return network, scaled_last_timestep, final_state, final_output, connections_added, total_net_conn_tries
    
def normalize_sample_input(sample_input):
    # Find the maximum value across the entire sample (across all timestep vectors)
    max_value = np.max([np.max(timestep_vector) for timestep_vector in sample_input])
    # Normalize the entire sample by the maximum value
    normalized_sample = sample_input / max_value
    # Return the normalized sample and the scaling factor (max_value)
    return normalized_sample, max_value

def main():
    sample_input = [1000000 * np.random.rand(TIME_STEPS, INPUT_NODES)]
    normalized_samples, scaling_factor = zip(*[normalize_sample_input(sample) for sample in sample_input])
    
    for generation in range(GEN):
        network, scaled_last_timestep, final_state, final_output, connections_added, total_net_conn_tries = create_and_run_network(normalized_samples[0], scaling_factor[0])
        print(f"GEN {generation}: ###########################################################################")
        print(final_state)
        print(f"LAST TIME STEP: {scaled_last_timestep} - OUTPUT: {final_output} - {connections_added} connections in {total_net_conn_tries} tries\n")
        network.plot_network()
if __name__ == "__main__":
    main()
