import random
import matplotlib.pyplot as plt
import networkx as nx

class NeatVisualizer:
    def __init__(self):
        self.reward_data = []
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))

    def visualize_genome(self, genome):
        self.ax[0].cla()
        G = nx.DiGraph()

        # Define x-coordinates for each layer
        left_x = 0.1  # X-coordinate for input neurons
        right_x = 0.9  # X-coordinate for output neurons
        center_x_range = (0.4, 0.6)  # X-coordinate range for hidden neurons

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

        # Add nodes and edges to the graph
        for neuron_id, neuron_gene in genome.neuron_genes.items():
            if neuron_gene.enabled:
                color = "skyblue" if neuron_gene.layer == "input" else "lightgreen" if neuron_gene.layer == "output" else "lightgrey"
                G.add_node(neuron_id, color=color, style="filled", shape="circle")

        for _, conn_gene in genome.connection_genes.items():
            if conn_gene.enabled:
                G.add_edge(conn_gene.from_neuron, conn_gene.to_neuron, weight=conn_gene.weight)

        # Draw the graph
        nx.draw(G, pos, with_labels=False, node_color=[G.nodes[node]["color"] for node in G.nodes],
                edge_color="black", width=1, linewidths=1, node_size=500, alpha=0.9, ax=self.ax[0])
        self.ax[0].set_title("Genome Structure")

    def plot_rewards(self, generation, total_reward):
        self.reward_data.append((generation, total_reward))
        generations, rewards = zip(*self.reward_data)

        self.ax[1].cla()  # Clear the current plot
        self.ax[1].plot(generations, rewards, '-o')
        self.ax[1].set_title("Total Rewards over Generations")
        self.ax[1].set_xlabel("Generation")
        self.ax[1].set_ylabel("Total Reward")

        plt.draw()
        plt.pause(0.01)
