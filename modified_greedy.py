from frame import FrameEvent, FrameEventType
import random

from frame_calculations import maximum_pair_optimum_tree, maximum_pair_optimum_decomposition
from io_tools.crossing_graph import get_crossing_graph
from io_tools.output import export_as_gif, export_as_vertex_gif
from io_tools.read_graph import read_hog

import networkx as nx
import os

def generate_hog_story(hog_id: int):
	if not os.path.exists("hog_stories"):
		os.mkdir("hog_stories")

	graph, vertex_pos = read_hog(os.path.join("house_of_graphs", f"{hog_id}.txt"))
	crossing_graph = get_crossing_graph(graph, vertex_pos)

	isolated_vertices = list(nx.isolates(crossing_graph))
	crossing_graph.remove_nodes_from(isolated_vertices)

	if nx.is_tree(crossing_graph):
		initial_frame, last_frame = maximum_pair_optimum_tree(crossing_graph)
	else:
		treewidth, decomposition = nx.approximation.treewidth_min_fill_in(crossing_graph)
		initial_frame, last_frame = maximum_pair_optimum_decomposition(crossing_graph, decomposition)

	frame_events = compute_frames_greedy(crossing_graph, initial_frame, last_frame, 'a')
	frame_events = [FrameEvent(e, 0, FrameEventType.IN) for e in isolated_vertices] + frame_events

	frames = [graph] + FrameEvent.to_frames(frame_events, vertex_pos)

	export_as_gif(vertex_pos, frames, out_file=os.path.join("hog_stories", f"{hog_id}.gif"), node_size=16)


def generate_crossing_story(hog_id: int):
	if not os.path.exists("hog_stories"):
		os.mkdir("hog_stories")

	crossing_graph, vertex_pos = read_hog(os.path.join("house_of_graphs\\crossing_graphs", f"{hog_id}.txt"))

	isolated_vertices = list(nx.isolates(crossing_graph))
	crossing_graph.remove_nodes_from(isolated_vertices)

	if nx.is_tree(crossing_graph):
		initial_frame, last_frame = maximum_pair_optimum_tree(crossing_graph)
	else:
		treewidth, decomposition = nx.approximation.treewidth_min_fill_in(crossing_graph)
		initial_frame, last_frame = maximum_pair_optimum_decomposition(crossing_graph, decomposition)

	frame_events = compute_frames_greedy(crossing_graph, initial_frame, last_frame, 'a')
	frame_events = [FrameEvent(n, 0, FrameEventType.IN) for n in isolated_vertices] + frame_events
	frames = [] + FrameEvent.to_crossing_frames(frame_events)

	export_as_vertex_gif(vertex_pos, frames, crossing_graph, out_file=os.path.join("hog_stories", f"{hog_id}.gif"), node_size=16)


def compute_frames_greedy(crossing_graph: nx.Graph, initial_frame: list[tuple[int, int]], last_frame: list[tuple[int, int]], variation) -> [FrameEvent]:

	frame_events = [FrameEvent(e, 0, FrameEventType.IN) for e in initial_frame]

	current_graph = nx.Graph()
	current_graph.add_nodes_from(initial_frame)
	future_graph = crossing_graph.copy()
	future_graph.remove_nodes_from(initial_frame)

	counter = 1
	while len(future_graph.nodes) > 0:
		neighbors_in_current_graph = dict()
		current_neighbors = dict()
		neighbors_in_future_graph = dict()
		future_neighbors = dict()

		# STEP 1: Pick a future edge with the minimum current degree
		for vertex in future_graph.nodes():
			if not ((vertex in last_frame) and (vertex not in list(nx.isolates(future_graph)))):
				current_neighbors[vertex] = set(crossing_graph.neighbors(vertex)) & set(current_graph.nodes())
				neighbors_in_current_graph[vertex] = len(current_neighbors[vertex])

		filtered_nodes = [vertex for vertex, count in neighbors_in_current_graph.items() if count == min(neighbors_in_current_graph.values())]

		if variation == 'b':
			# (b): Pick a future edge such that the size of the set of all future neighbors of the current neighbors is maximum

			if len(filtered_nodes) > 1:
				# STEP 2
				for vertex in filtered_nodes:
					future_neighbors[vertex] = set()
					for current_neighbor in current_neighbors[vertex]:
						future_neighbors[vertex] &= set(crossing_graph.neighbors(current_neighbor)) & set(future_graph.nodes())
					neighbors_in_future_graph[vertex] = len(future_neighbors[vertex])

				filtered_nodes = [vertex for vertex, count in neighbors_in_future_graph.items() if count == max(neighbors_in_future_graph.values())]

			# STEP 3: Get the one with the maximum future degree
			if len(filtered_nodes) > 1:
				future_neighbors = dict()
				for vertex in filtered_nodes:
					future_neighbors[vertex] = set(future_graph.neighbors(vertex))

				filtered_nodes = [vertex for vertex, count in future_neighbors.items() if count == max(future_neighbors.values())]

		inserted_vertex = random.choice(list(filtered_nodes))
		frame_events.append(
			FrameEvent(inserted_vertex, counter, FrameEventType.IN)
		)
		frame_events = frame_events + [FrameEvent(e, counter, FrameEventType.OUT) for e in current_neighbors[inserted_vertex]]

		current_graph.remove_nodes_from(current_neighbors[inserted_vertex])
		current_graph.add_node(inserted_vertex)
		future_graph.remove_node(inserted_vertex)
		counter += 1

	frame_events = sorted(frame_events)

	return frame_events


if __name__ == '__main__':
	generate_crossing_story(2)