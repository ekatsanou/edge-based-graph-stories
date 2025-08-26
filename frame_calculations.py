from itertools import product
import networkx as nx
# from functools import cache


def maximum_pair_b(crossing_graph: nx.Graph) -> (list, list):
	initial_frame = []
	last_frame = []
	V = set(crossing_graph.nodes)

	while len(initial_frame) < len(V) / 2:
		I1_set = set(initial_frame)
		N_I1 = set()
		for v in initial_frame:
			N_I1.update(crossing_graph.neighbors(v))

		candidates = V - I1_set - N_I1
		if not candidates:
			break

		min_deg_node = min(candidates, key=lambda u: crossing_graph.degree(u))
		initial_frame.append(min_deg_node)

	while True:
		I1_set = set(initial_frame)
		I2_set = set(last_frame)
		N_I2 = set()
		for v in last_frame:
			N_I2.update(crossing_graph.neighbors(v))

		candidates = V - I1_set - I2_set - N_I2
		if not candidates:
			break

		min_deg_node = min(candidates, key=lambda v: crossing_graph.degree(v))
		last_frame.append(min_deg_node)

	return initial_frame, last_frame


def maximum_pair_a(crossing_graph: nx.Graph) -> (list, list):
	j1 = []
	j2 = []
	v = set(crossing_graph.nodes)

	i = 1
	while True:
		ji = j1 if i == 1 else j2
		n_ji = set()

		for u in ji:
			n_ji.update(crossing_graph.neighbors(u))

		candidates = v - set(j1) - set(j2) - n_ji
		if not candidates:
			i = 3 - i
			ji = j1 if i == 1 else j2
			n_ji = set()
			for u in ji:
				n_ji.update(crossing_graph.neighbors(u))
			candidates = v - set(j1) - set(j2) - n_ji
			if not candidates:
				break
			continue

		min_deg_node = min(candidates, key=lambda u: crossing_graph.degree(u))
		ji.append(min_deg_node)
		i = 3 - i

	if len(j1) <= len(j2):
		return j1, j2
	else:
		return j2, j1


# @cache
def maximum_pair_optimum_tree(crossing_graph: nx.Graph) -> (list, list):
	root = nx.center(crossing_graph)[0]
	parent = dict()
	marked = set()
	l_function = dict()
	stack = [root]
	parent[root] = None
	while stack:
		node = stack[-1]
		if node not in marked:
			l_function[node] = {}
			marked.add(node)
			l_function[node][0] = [[0, 0, [set(), set()]]]
			l_function[node][1] = [[1, 0, [{node}, set()]]]
			l_function[node][2] = [[0, 1, [set(), {node}]]]
			for neighbor in crossing_graph.neighbors(node):
				if neighbor != parent[node]:
					parent[neighbor] = node
					stack.append(neighbor)
		else:
			node = stack.pop()
			if node != root:
				l_function[parent[node]][0] = pareto_sum(l_function[parent[node]][0], l_function[node][0] + l_function[node][1] + l_function[node][2])
				l_function[parent[node]][1] = pareto_sum(l_function[parent[node]][1], l_function[node][0] + l_function[node][2])
				l_function[parent[node]][2] = pareto_sum(l_function[parent[node]][2], l_function[node][0] + l_function[node][1])
			else:
				all_triplets = l_function[root][0] + l_function[root][1] + l_function[root][2]
				best = max(all_triplets, key=lambda x: min(x[0], x[1]))

	if best[2][0] < best[2][1]:
		return best[2][0], best[2][1]
	else:
		return best[2][1], best[2][0]


def pareto_optimal_pair(crossing_graph: nx.Graph) -> (list, list):
	_, decomposition = nx.approximation.treewidth_min_fill_in(crossing_graph)
	return maximum_pair_optimum_decomposition(crossing_graph, decomposition)


def maximum_pair_optimum_decomposition(crossing_graph: nx.Graph, tree_decomposition) -> (list, list):
	root = nx.center(tree_decomposition)[0]
	parent = dict()
	marked = set()
	l_function = dict()
	stack = [root]
	parent[root] = None
	while stack:
		bag_node = stack[-1]
		if bag_node not in marked:
			marked.add(bag_node)
			l_function[bag_node] = {}

			# Generate all valid 3-colorings for the vertices in the bag
			for coloring in product(range(3), repeat=len(bag_node)):
				coloring_dict = dict(zip(bag_node, coloring))
				valid = True
				for u in bag_node:
					for v in bag_node:
						if u != v and crossing_graph.has_edge(u, v):
							if ((coloring_dict[u] == 1 and coloring_dict[v] == 1) or
									(coloring_dict[u] == 2 and coloring_dict[v] == 2)):
								valid = False
								break
					if not valid:
						break
				if valid:
					list_1 = {v for v in bag_node if coloring_dict[v] == 1}
					list_2 = {v for v in bag_node if coloring_dict[v] == 2}
					l_function[bag_node][coloring] = [[len(list_1), len(list_2), [list_1, list_2]]]

			for neighbor in tree_decomposition.neighbors(bag_node):
				if neighbor != parent[bag_node]:
					parent[neighbor] = bag_node
					stack.append(neighbor)
		else:
			bag_node = stack.pop()
			if bag_node != root:

				shared_vertices = set(bag_node) & set(parent[bag_node])

				for parent_coloring, parent_value in l_function[parent[bag_node]].items():
					l_comp = []
					for parent_spec_coloring in parent_value:

						for child_coloring, child_value in l_function[bag_node].items():
							for child_spec_coloring in child_value:
								compatible = True
								for v in shared_vertices:
									color1 = 1 if v in child_spec_coloring[2][0] else 2 if v in child_spec_coloring[2][1] else 0
									color2 = 1 if v in parent_spec_coloring[2][0] else 2 if v in parent_spec_coloring[2][1] else 0
									if color1 != color2:
										compatible = False
										break
								if compatible:
									a_1 = child_spec_coloring[0]-len(child_spec_coloring[2][0] & parent_spec_coloring[2][0])
									b_1 = child_spec_coloring[1]-len(child_spec_coloring[2][1] & parent_spec_coloring[2][1])
									pointers = child_spec_coloring[2]
									l_comp.append([a_1, b_1, pointers])
					if l_comp:
						l_function[parent[bag_node]][parent_coloring] = pareto_sum(l_function[parent[bag_node]][parent_coloring], l_comp)

			else:
				best_triplet = None
				max_min_value = float('-inf')

				for parent_coloring, parent_value in l_function[root].items():
					for triplet in parent_value:
						min_val = min(triplet[0], triplet[1])
						if (min_val > max_min_value or
								(min_val == max_min_value and max(triplet[0], triplet[1]) > max(best_triplet[0], best_triplet[1]))):
							max_min_value = min_val
							best_triplet = triplet

				if best_triplet[2][0] < best_triplet[2][1]:
					return best_triplet[2][0], best_triplet[2][1]
				else:
					return best_triplet[2][1], best_triplet[2][0]


def pareto_sum(l1: list[list[int, int, list]], l2: list[list[int, int, list]]) -> list[list]:
	l = []

	for p1 in l1:
		a1, b1, c1 = p1
		for p2 in l2:
			a2, b2, c2 = p2
			l.append([a1 + a2, b1 + b2, [c1[0] | c2[0], c1[1] | c2[1]]])

	l.sort(reverse=True)

	pareto = [l[0]]

	for i in range(1, len(l)):
		p = l[i]
		prev = pareto[-1]

		if not (p[0] == prev[0] or p[1] <= prev[1]):
			pareto.append(p)

	return pareto
