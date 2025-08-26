import networkx as nx
import matplotlib.pyplot as plt

def show_graph(filename,ns=150):
    """
    Load a graph from a file with the following file format:
    - first row: number of nodes
    - subsequent rows: <x> <y> <adj1> <adj2> ...
    """
    with open(filename, 'r') as f:
        lines = f.readlines();

    num_nodes = int(lines[0].strip());
    G = nx.Graph();
    pos = {};

    # Construct the graph with coordinated from the data in the file
    # --------------------------------------------------------------
    for i in range(1, num_nodes + 1):
        parts = lines[i].strip().split();
        x, y = float(parts[0]), float(parts[1]);
        adjlist = list(map(int, parts[2:]));
        
        G.add_node(i-1);    # node i-1 (0-based)
        pos[i-1] = (x,y);   # save the node position

        for v in adjlist:
            G.add_edge(i-1,v); # add the edge

    # Show the graph in the plot window
    # ---------------------------------
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=ns, font_size=10);
    plt.title("graph layout");
    plt.axis("off");
    plt.show();

    return G;
