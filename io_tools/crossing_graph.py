import networkx as nx
from itertools import combinations
from shapely.geometry import LineString


def get_crossing_graph(graph: nx.Graph, vertex_position: dict) -> nx.Graph:
    crossing_graph = nx.Graph()

    crossing_graph.add_nodes_from(graph.edges)
    crossing_graph.add_edges_from([(e, f) for e, f in combinations(graph.edges, 2)
                                   if _cross(vertex_position[e[0]], vertex_position[e[1]],
                                            vertex_position[f[0]], vertex_position[f[1]])])
    return crossing_graph


def _cross(p: tuple, q: tuple, r: tuple, s: tuple):
    if p in (r, s) or q in (r, s):
        return False

    line1 = LineString([p, q])
    line2 = LineString([r, s])
    return line1.intersects(line2)