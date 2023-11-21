import networkx as nx


def visualize_genome(genome, ax=None):
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
            G.add_edge(
                conn_gene.from_neuron, conn_gene.to_neuron, weight=conn_gene.weight
            )

    pos = nx.spring_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color=[G.nodes[node]["color"] for node in G.nodes],
        edge_color="black",
        width=1,
        linewidths=1,
        node_size=500,
        alpha=0.9,
        ax=ax,
    )
