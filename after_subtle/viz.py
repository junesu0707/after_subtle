import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from math import ceil

def plot_bigram_graph_groups(
    groups,
    bigram_mats,
    layout="circular",
    node_scaling=2000,
    save_dir=None
):
    """Plot the bigram graph for each group.

    Parameters:
    groups : list
        List of groups to plot.
    bigram_mats : list
        List of bigram matrices for each group.
    layout : str, optional
        Layout of the graph (e.g., 'circular', 'spring'), by default 'circular'.
    node_scaling : int, optional
        Scaling factor for the node size, by default 2000.
    save_dir : str, optional
        Directory to save the plots, by default None.
    """
    n_row = ceil(len(groups) / 2)
    fig, all_axes = plt.subplots(n_row, 2, figsize=(20, 10 * n_row))
    ax = all_axes.flat

    for i in range(len(groups)):
        G = nx.from_numpy_array(bigram_mats[i] * 100)
        widths = nx.get_edge_attributes(G, "weight")
        
        if layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        nx.draw_networkx_nodes(
            G, pos, node_size=node_scaling, node_color="white", edgecolors="red", ax=ax[i]
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=widths.keys(), width=list(widths.values()), edge_color="black", ax=ax[i], alpha=0.6
        )
        nx.draw_networkx_labels(
            G, pos, font_color="black", ax=ax[i]
        )
        ax[i].set_title(groups[i])

    # Axis spines off
    for sub_ax in ax:
        sub_ax.axis("off")

    # Save figures
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "bigram_graphs.pdf"))
        fig.savefig(os.path.join(save_dir, "bigram_graphs.png"))

    plt.show()


