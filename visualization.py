import random
import matplotlib.pyplot as plt
import networkx as nx

class NeatVisualizer:
    def __init__(self):
        self.fitness_data = []
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))

    def visualize_genome(self, genome):
        self.ax[0].cla()
        G = nx.DiGraph()

        # Define x-coordinates for each layer
        left_x = 0.1  # X-coordinate for input neurons
        right_x = 0.9  # X-coordinate for output neurons
        center_x_range = (0.3, 0.7)  # X-coordinate range for hidden neurons

        # Function to calculate y-positions for input/output neurons
        def calculate_y_positions(neurons, start_y, end_y, x_pos):
            spacing = (end_y - start_y) / (len(neurons) + 1)
            return {neuron_id: (x_pos, start_y + (i + 1) * spacing) for i, neuron_id in enumerate(neurons)}

        # Function to calculate positions for hidden neurons
        def calculate_hidden_positions(neurons, start_y, end_y):
            spacing = (end_y - start_y) / (len(neurons) + 1)
            return {neuron_id: (random.uniform(*center_x_range), start_y + (i + 1) * spacing) for i, neuron_id in enumerate(neurons)}

        # Group neurons by layer
        input_neurons = [neuron_id for neuron_id, neuron_gene in genome.neuron_genes.items() if neuron_gene.layer == "input" and neuron_gene.enabled]
        output_neurons = [neuron_id for neuron_id, neuron_gene in genome.neuron_genes.items() if neuron_gene.layer == "output" and neuron_gene.enabled]
        hidden_neurons = [neuron_id for neuron_id, neuron_gene in genome.neuron_genes.items() if neuron_gene.layer == "hidden" and neuron_gene.enabled]

        # Calculate positions for each layer
        pos = {}
        pos.update(calculate_y_positions(input_neurons, 0.1, 0.9, left_x))
        pos.update(calculate_y_positions(output_neurons, 0.1, 0.9, right_x))
        pos.update(calculate_hidden_positions(hidden_neurons, 0.1, 0.9))

        # Add nodes and edges to the graph with modifications
        for neuron_id, neuron_gene in genome.neuron_genes.items():
            if neuron_gene.enabled:
                color = "skyblue" if neuron_gene.layer == "input" else "lightgreen" if neuron_gene.layer == "output" else "lightgrey"
                G.add_node(neuron_id, color=color, style="filled", shape="circle", activation=neuron_gene.activation)

        edge_weights = []
        for _, conn_gene in genome.connection_genes.items():
            # check if connection gene is enabled and from/to neurons exist
            if not conn_gene.enabled or conn_gene.from_neuron not in G.nodes or conn_gene.to_neuron not in G.nodes:
                continue
            else:
                G.add_edge(conn_gene.from_neuron, conn_gene.to_neuron)
                edge_weights.append(abs(conn_gene.weight))  # use absolute value for visualization

        # Normalize edge weights for visualization
        max_weight = max(edge_weights, default=1)  # Avoid division by zero
        edge_weights = [w / max_weight * 5 for w in edge_weights]  # Adjust scaling factor as needed

        # Draw the graph
        nx.draw(G, pos, with_labels=False, node_color=[G.nodes[node]["color"] for node in G.nodes],
                edge_color="black", width=edge_weights, linewidths=1, node_size=500, alpha=0.9, ax=self.ax[0])

        # Add labels for activation functions of hidden neurons
        for node, data in G.nodes(data=True):
            if data.get('activation') and data['activation'] != 'identity':
                self.ax[0].text(pos[node][0], pos[node][1]+0.05, data['activation'], horizontalalignment='center')

        self.ax[0].set_title("Genome Structure")

    def plot_fitness(self, generation, fitness):
        self.fitness_data.append((generation, fitness))
        generations, fitness_values = zip(*self.fitness_data)

        self.ax[1].cla()  # Clear the current plot
        self.ax[1].plot(generations, fitness_values, "-o")
        self.ax[1].set_title("Fitness over Generations")
        self.ax[1].set_xlabel("Generation")
        self.ax[1].set_ylabel("Fitness")

        plt.draw()
        plt.pause(0.01)
