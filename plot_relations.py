
# This script creates a graph of the relationships between some objects in the NEAT algorithm.

import matplotlib.pyplot as plt
import networkx as nx

# Creating a directed graph
G = nx.DiGraph()

# Adding nodes
G.add_node("NeuronGene")
G.add_node("ConnectionGene")
G.add_node("Genome")
G.add_node("Species")
G.add_node("Population")
G.add_node("Network")
G.add_node("Elites")
G.add_node("Representative")
G.add_node("BestGenome")

# Adding edges based on the given relationships
G.add_edge("NeuronGene", "ConnectionGene")
G.add_edge("NeuronGene", "Genome")
G.add_edge("ConnectionGene", "Genome")
G.add_edge("Network", "Genome")
G.add_edge("Genome", "Network")
G.add_edge("Genome", "Species")
G.add_edge("Genome", "Population")
G.add_edge("Elites", "Species")
G.add_edge("Representative", "Species")
G.add_edge("BestGenome", "Population")


#pos = nx.kamada_kawai_layout(G)

plt.figure(figsize=(16, 12))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1000, font_size=15, arrows=True, arrowstyle='->', arrowsize=20, edge_color='black', linewidths=1, pos=nx.fruchterman_reingold_layout(G))
plt.title("Graph of Relationships")
plt.show()