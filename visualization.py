import matplotlib.pyplot as plt
import networkx as nx

class NeatVisualizer:
    def __init__(self):
        self.reward_data = []
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))

    def visualize_genome(self, genome):
        G = nx.DiGraph()

        for neuron_id, neuron_gene in genome.neuron_genes.items():
            if neuron_gene.enabled:
                style = "filled"

                if neuron_gene.layer == "input":
                    color = "skyblue"
                elif neuron_gene.layer == "output":
                    color = "lightgreen"
                else:
                    color = "lightgrey"

                G.add_node(neuron_id, color=color, style=style, shape="circle")

        for _, conn_gene in genome.connection_genes.items():
            if conn_gene.enabled:
                G.add_edge(conn_gene.from_neuron, conn_gene.to_neuron, weight=conn_gene.weight)

        pos = nx.spring_layout(G)
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
