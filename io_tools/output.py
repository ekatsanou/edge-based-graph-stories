from frame import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx


def export_as_gif(pos: dict, frames: [nx.Graph], out_file: str | None = None, fps=1, **kwargs) -> None:
    """
    Export the edge story as a gif.

    :param pos: A dictionary that maps the vertices of the graph to a tuple of 2-D coordinates.
    :param frames: A list of frames (ie graphs) in the desired order.
    :param out_file: The path to the file to which the gif is saved. If out_file=None, then the gif is shown via matplotlib.
    :param fps: The number of frames per second.
    :param kwargs: Keyword arguments that are passed to the graph drawing function of networkx.
    """

    def update(idx):
        ax.clear()
        ax.set_title(f'Frame {idx+1} with {len(frame_graphs[idx].edges)} edges (obj: {min_number_of_edges})')
        ax.set_xlim(min_x-1, max_x+1)
        ax.set_ylim(min_y-1, max_y+1)
        nx.draw(frame_graphs[idx], pos, ax=ax, **kwargs)

    min_x, max_x = min(x for x, _ in pos.values()), max(x for x, _ in pos.values())
    min_y, max_y = min(y for _, y in pos.values()), max(y for _, y in pos.values())

    fig, ax = plt.subplots()

    frame_graphs = []
    for frame in frames:
        current_graph = nx.Graph()
        current_graph.add_nodes_from(pos.keys())
        current_graph.add_edges_from(frame.edges)
        frame_graphs.append(current_graph)

    min_number_of_edges = min(len(g.edges) for g in frame_graphs)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_graphs), interval=500, repeat=False)

    if out_file:
        ani.save(out_file, writer="pillow", fps=fps)
    else:
        plt.show()


def export_as_vertex_gif(pos: dict, frames: [nx.Graph], original_graph: nx.Graph, out_file: str | None = None, fps=1, **kwargs) -> None:
    """
    Export the crossing story as a gif.

    :param pos: A dictionary that maps the vertices of the graph to a tuple of 2-D coordinates.
    :param frames: A list of frames (ie graphs) in the desired order.
    :param out_file: The path to the file to which the gif is saved. If out_file=None, then the gif is shown via matplotlib.
    :param fps: The number of frames per second.
    :param kwargs: Keyword arguments that are passed to the graph drawing function of networkx.
    """

    def update(idx):
        ax.clear()
        ax.set_title(f'Frame {idx+1} with {len(frame_graphs[idx].nodes)} nodes (min: {min_number_of_nodes})')
        ax.set_xlim(min_x - 1, max_x + 1)
        ax.set_ylim(min_y - 1, max_y + 1)

        # Draw original graph in blue
        nx.draw(original_graph, pos, ax=ax, node_color='blue', **kwargs)

        # Draw frame graph in red
        nx.draw(frame_graphs[idx], pos, ax=ax, node_color='red', **kwargs)


    min_x, max_x = min(x for x, _ in pos.values()), max(x for x, _ in pos.values())
    min_y, max_y = min(y for _, y in pos.values()), max(y for _, y in pos.values())

    fig, ax = plt.subplots()

    frame_graphs = []
    for frame in frames:
        current_graph = nx.Graph()
        current_graph.add_nodes_from(frame.nodes)
        frame_graphs.append(current_graph)

    min_number_of_nodes = min(len(g.nodes) for g in frame_graphs)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_graphs), interval=500, repeat=False)

    if out_file:
        ani.save(out_file, writer="pillow", fps=fps)
    else:
        plt.show()

