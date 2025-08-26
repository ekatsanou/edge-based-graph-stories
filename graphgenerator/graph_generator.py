import networkx as nx;
import matplotlib.pyplot as plt;
import sys;
import random;

def generate_graph_ER (n, m, filename, seed, nonplanar=False):
    """
    1. Generate an undirected graph with n vertices and m edges, using the Erdos-Renyi model.
        (uniform probability distribution)
    2. Compute a drawing of the graph using Fruchterman-Reingold's force-directed algorithm.
    3. Saves the graph in a textual file.
    4. If nonplanar=True, it attempts to generate a nonplanar graph (the number of attempts is bounded by a constant)
    5. It returns the last used seed
    """

    # 0. set the seed
    # rng = random.Random(seed);

    # 1. Generate the random graph
    print ("Generating graph " + filename + " ...");

    G = nx.gnm_random_graph(n,m,seed);
    is_planar, embedding = nx.check_planarity(G); # planarity testing

    if (nonplanar):
        count = 0;
        MAX_ATTEMPTS = 1000;
        while (is_planar and count<MAX_ATTEMPTS):
            seed += 1;
            count+=1;
            G = nx.gnm_random_graph(n,m,seed); #try again    
            is_planar, embedding = nx.check_planarity(G); # planarity testing
            
    if (is_planar):
        print ("-----> The generated graph is planar, ....");
    
    # 2. Draw the graph with Fruchterman-Reingold's force-directed algorithm
    pos = nx.spring_layout(G);
    #nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10);
    #plt.title(f"Random graph with {n} nodes and {m} edges");
    #plt.show();

    # 3. Save the graph
    with open(filename, 'w') as f:
        row = str(G.number_of_nodes())+"\n";
        f.write(row);
        for v in G.nodes():
            x,y = pos[v];
            adjlist = list(G.adj[v]);
            row = f"{x:.5f} {y:.5f} " + " ".join(map(str, adjlist)) + "\n";
            f.write(row);
    f.close();

    print(f"Graph saved in '{filename}' as adjacency list.");
    print(f"Last used seed '{seed}'.");
    return seed;



def generate_graph_seq (n_min, n_max, n_step, d_min, d_max, d_step, num_graphs, model, seed, nonplanar=False):
    """
    Generates a sequence of geometric graphs in the specified model (possible values: "er", ...).
    - The number of nodes varies in the range [n_min,n_max] with step n_step.
    - The density (num_edges/num_nodes) varies in the range [d_min,d_max] with step d_step;
      it is expected that densities are float numbers of the form x.y (x and y being single digits) 
    - For each fixed number of nodes and density value, the number of graphs generated equals num_graphs.
    - The value seed is an integer number that specifies the seed of the random generation.
    """
    #rng = random.Random(seed);  # set the seed

    # generate graphs
    for n in range (n_min,n_max+1,n_step):
        d = d_min;
        while (d<=d_max):
            m = int(d*n);
            for g in range(1,num_graphs+1):
                gname = "g_" + model + "_" + str(n) + "_" + str(int(d*10)) + "_" + str(g) + ".txt";
                if (model=="er"):
                    #generate_graph_ER (n, m, gname, rng, nonplanar);
                    new_seed = generate_graph_ER (n, m, gname, seed, nonplanar);
                    seed = new_seed + 1;
                else:
                    print("uknown model\n");
                    sys.exit(1);
            d += d_step;
    
