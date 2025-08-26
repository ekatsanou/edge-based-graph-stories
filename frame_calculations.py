from itertools import product
import networkx as nx
# from functools import cache


def maximum_pair_b(crossing_graph: nx.Graph) -> (list, list):
	initial_frame = []
	last_frame = []
	V = set(crossing_graph.nodes)

	while len(initial_frame) <= len(V) / 2 - 1:
		I1_set = set(initial_frame)
		N_I1 = set()
		for v in initial_frame:
			N_I1.update(crossing_graph.neighbors(v))

		candidates = V - I1_set - N_I1
		if not candidates:
			break

		min_deg_node = min(candidates, key=lambda u: len(set(crossing_graph.neighbors(u)) & candidates))
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

		min_deg_node = min(candidates, key=lambda v: len(set(crossing_graph.neighbors(v)) & candidates))
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

		min_deg_node = min(candidates, key=lambda u: len(set(crossing_graph.neighbors(u)) & candidates))
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
				best = max(all_triplets, key=lambda x: (min(x[0], x[1]), max(x[0], x[1])))

	if len(best[2][0]) < len(best[2][1]):
		return best[2][0], best[2][1]
	else:
		return best[2][1], best[2][0]


def pareto_optimal_pair(crossing_graph: nx.Graph) -> (list, list):
	_, decomposition = nx.approximation.treewidth_min_fill_in(crossing_graph)
	return maximum_pair_optimum_decomposition(crossing_graph, decomposition)


def maximum_pair_optimum_decomposition(crossing_graph: nx.Graph, tree_decomposition) -> (list, list):
	"""
	# Pareto optimal maximum pair of independent sets with the dynamic program of Theorem 5 of the paper.

	# For each bag nu and each coloring c of its vertices with 0 (white), 1 (red) and 2 (blue), L(nu, c) is the list of
	# Pareto optimal pairs (alpha, beta). The lists of a bag are stored in an array A(nu): the coloring that gives color
	# gamma_i to the i-th vertex of the bag is at position sum_i gamma_i * 3^i, where the vertices shared with the parent
	# bag come last. Hence, the colorings c' of a child that are compatible with a coloring c of its parent (c' = c on the
	# shared vertices) form a contiguous block of the array, whose position is given by the colors of the shared vertices.
	# The array is stored sparsely (a dictionary from positions to lists), since only the valid colorings have a list.
	# Each entry also records how it was obtained, so that the pair of independent sets is traced back at the end.
	"""
	root = nx.center(tree_decomposition)[0]
	parent = {root: None}
	order = dict()      # bag -> its vertices in the order of the array positions (shared vertices with the parent last)
	shared = dict()     # bag -> number of vertices shared with the parent
	array = dict()      # bag -> A(bag): position of a valid coloring c -> list L(bag, c)
	marked = set()
	stack = [root]

	while stack:
		bag = stack[-1]
		if bag not in marked:  # top-down traversal: initialize A(bag)
			marked.add(bag)
			common = set(bag) & set(parent[bag]) if parent[bag] is not None else set()
			order[bag] = tuple(sorted(set(bag) - common, key=repr) + sorted(common, key=repr))
			shared[bag] = len(common)
			array[bag] = _initial_lists(crossing_graph, bag, order[bag])
			for neighbor in tree_decomposition.neighbors(bag):
				if neighbor != parent[bag]:
					parent[neighbor] = bag
					stack.append(neighbor)
		else:  # bottom-up traversal: add the subtree of bag to the lists of its parent
			stack.pop()
			if bag != root:
				_join_child(bag, parent[bag], order, shared, array)
				del array[bag]  # no longer needed: the entries of the parent refer to the entries they were built from

	best = None
	for lists in array[root].values():
		for entry in lists:
			if best is None or (min(entry[0], entry[1]), max(entry[0], entry[1])) > (min(best[0], best[1]), max(best[0], best[1])):
				best = entry
	red, blue = _trace_back(best)
	if len(red) < len(blue):
		return list(red), list(blue)
	else:
		return list(blue), list(red)


def _initial_lists(crossing_graph: nx.Graph, bag, vertices: tuple) -> dict:
	lists = dict()
	for index in range(3 ** len(vertices)):
		colors = _digits(index, len(vertices))
		valid = not any(colors[i] != 0 and colors[i] == colors[j] and crossing_graph.has_edge(vertices[i], vertices[j])
						for i in range(len(vertices)) for j in range(i + 1, len(vertices)))
		if valid:
			lists[index] = [(colors.count(1), colors.count(2), ("init", vertices, index))]
	return lists


def _join_child(child, parent_bag, order, shared, array) -> None:
	"""Combines each list L(parent, c) with the lists L(child, c') of the colorings c' compatible with c."""
	child_vertices, parent_vertices = order[child], order[parent_bag]
	free = len(child_vertices) - shared[child]   # vertices of the child that are not in the parent
	block_size = 3 ** free
	# The colorings of the child with the same colors on the shared vertices form a block of block_size positions;
	# block b holds the colorings whose colors of the shared vertices are b.
	blocks = dict()
	for index, lists in array[child].items():
		blocks.setdefault(index // block_size, []).extend(lists)
	# Position of each shared vertex in the parent's order
	positions = [parent_vertices.index(v) for v in child_vertices[free:]]

	parent_lists = array[parent_bag]
	for index, lists in parent_lists.items():
		colors = [(index // 3 ** p) % 3 for p in positions]       # colors of the shared vertices under c
		entries = blocks.get(sum(color * 3 ** i for i, color in enumerate(colors)))
		if not entries:
			continue
		alpha_shared, beta_shared = colors.count(1), colors.count(2)  # counted in both the parent's and the child's entry
		combined = [(e1[0] + e2[0] - alpha_shared, e1[1] + e2[1] - beta_shared, ("join", e1, e2))
					for e1 in lists for e2 in entries]
		parent_lists[index] = _pareto_optimal(combined)


def _pareto_optimal(entries: list) -> list:
	"""Pareto optimal entries (alpha, beta, origin): sorted by (alpha, beta) non-increasingly, an entry is kept if it
	has a smaller alpha and a larger beta than the last kept entry."""
	entries.sort(key=lambda e: (e[0], e[1]), reverse=True)
	pareto = [entries[0]]
	for entry in entries[1:]:
		if entry[0] < pareto[-1][0] and entry[1] > pareto[-1][1]:
			pareto.append(entry)
	return pareto


def _trace_back(entry) -> (set, set):
	"""The red and the blue vertices of all colorings from which the entry was built."""
	red, blue = set(), set()
	stack = [entry]
	while stack:
		origin = stack.pop()[2]
		if origin[0] == "init":
			_, vertices, index = origin
			colors = _digits(index, len(vertices))
			red.update(v for v, color in zip(vertices, colors) if color == 1)
			blue.update(v for v, color in zip(vertices, colors) if color == 2)
		else:
			stack.append(origin[1])
			stack.append(origin[2])
	return red, blue


def _digits(index: int, length: int) -> list:
	return [(index // 3 ** i) % 3 for i in range(length)]


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
