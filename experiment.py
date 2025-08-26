"""
Runs the experiments of the paper: the six heuristics AG-1a2a, AG-1a2b, AG-1b2a, AG-1b2b, AG-1c2a, AG-1c2b and the
exact algorithm (ILP) on the graphs of the benchmark.

    python experiment.py --dataset trees                # one data set
    python experiment.py --dataset trees caterpillar    # several data sets
    python experiment.py --all                          # all data sets, one after the other
    python experiment.py --dataset random --no-ilp      # only the heuristics (no Gurobi needed)
    python experiment.py --all --check                  # also check every story

Data sets (graph files in graphgenerator/, format: first line the number of vertices, then one line per vertex with
its coordinates and its neighbors):
    trees, caterpillar, sp_graphs, planar   the files are the crossing graphs themselves (Crossing Graphs)
    random, real_graphs                     the files are straight-line drawings; the crossing graph is computed
                                            from the coordinates (Geometric Graphs)
In the crossing graph, every vertex is an edge of the drawing and two vertices are adjacent if the edges cross.
Isolated vertices (edges without crossings) are removed before running the algorithms; a graph without crossings is
skipped.

Phase 1: 1a = Pareto-optimal maximum pair of independent sets via a tree decomposition (minimum fill-in), computed
in a separate process with a time limit (default 10 minutes); 1b = sequential greedy; 1c = alternating greedy.
Phase 2: 2a = ties broken at random; 2b = tie-breaking rules of the paper, then at random. The random choices are
seeded with "<seed>-<graph name>-<heuristic>", so the stories can be reproduced (with the default seed 0 they are
the stories of the experiments in the paper).
The ILP gets the best story of the heuristics as initial solution and runs with a time limit (default 15 minutes).

Output: <out-dir>/results_<data set>.json (default out-dir: results), one entry per graph with the number of
crossings ("edges_of_the_crossing") and of edges of the drawing ("vertices_of_the_crossing") and, for every heuristic
and the ILP, the runtimes, the objective value (size of the smallest frame) and the story as a list of frame events
(at a time step, an edge enters (frame_type 1) or leaves (frame_type 0) the drawing). For Variant 1a,
"time_to_compute_initial_last_frame_seconds" is measured inside the worker process and "pair_wall_time_seconds"
includes starting it; the time limit applies to the latter. Progress is saved after every graph in
<out-dir>/results_<data set>.jsonl, so an interrupted run continues where it stopped when started again with the
same command. With --check, every story of the heuristics is checked to be a valid planar story (and every pair of
Variant 1a to consist of two disjoint independent sets); problems are reported.
"""
import argparse
import dataclasses
import json
import os
import multiprocessing
import queue
import random
from time import perf_counter

import networkx as nx

from frame import FrameEvent
from frame_calculations import maximum_pair_a, maximum_pair_b, pareto_optimal_pair
from io_tools.crossing_graph import get_crossing_graph
from io_tools.read_graph import read_hog
from modified_greedy import compute_frames_greedy

# data set -> (graph folder, the files are drawings whose crossing graph has to be computed)
DATASETS = {
	"trees": (os.path.join("graphgenerator", "trees"), False),
	"caterpillar": (os.path.join("graphgenerator", "caterpillar"), False),
	"sp_graphs": (os.path.join("graphgenerator", "series-parallel", "series-parallel-drawings"), False),
	"planar": (os.path.join("graphgenerator", "planar_graphs", "planar_drawings"), False),
	"random": (os.path.join("graphgenerator", "er_random_graphs_v2"), True),
	"real_graphs": (os.path.join("graphgenerator", "real_graphs"), True),
}
HEURISTICS = ["1a2a", "1a2b", "1b2a", "1b2b", "1c2a", "1c2b"]


def _pareto_worker(crossing_graph, result_queue):
	try:
		t1 = perf_counter()
		pair = pareto_optimal_pair(crossing_graph)
		result_queue.put(("ok", pair, perf_counter() - t1))
	except Exception as e:
		result_queue.put(("error", repr(e), None))


def pareto_pair_with_time_limit(crossing_graph: nx.Graph, time_limit: float):
	"""Phase 1 of Variant 1a in a separate process. Returns (pair or None, time measured in the worker or None,
	wall time including the process start, error or None). The pair is None if the time limit was reached or the
	computation failed (then error says why)."""
	result_queue = multiprocessing.Queue()
	t1 = perf_counter()
	process = multiprocessing.Process(target=_pareto_worker, args=(crossing_graph, result_queue))
	process.start()
	message = None
	while message is None and perf_counter() - t1 < time_limit:
		try:
			message = result_queue.get(timeout=min(1.0, max(0.01, time_limit - (perf_counter() - t1))))
		except queue.Empty:
			if not process.is_alive():  # the worker ended without sending a result (e.g. out of memory)
				try:
					message = result_queue.get(timeout=1.0)
				except queue.Empty:
					message = ("error", f"worker process ended with exit code {process.exitcode}", None)
	wall_time = perf_counter() - t1
	if process.is_alive():
		process.terminate()
	process.join()
	if message is None:
		return None, None, wall_time, None
	if message[0] == "error":
		return None, None, wall_time, message[1]
	return message[1], message[2], wall_time, None


def story_problem(crossing_graph: nx.Graph, events: list, obj_value, pair=None):
	"""Returns None if the frame events form a valid planar story with this objective value, otherwise a description
	of the problem. Checked: in every step after the first exactly one edge enters, it has not appeared before and it
	crosses no current edge, and only current edges leave; every edge appears; obj_value is the size of the smallest
	frame; if a pair is given, the initial frame is its first set and the final frame contains its second set."""
	steps = {}
	for e in events:
		steps.setdefault(e["time"], ([], []))[0 if e["frame_type"] == 1 else 1].append(_hashable(e["edge"]))
	current, seen, sizes = set(), set(), []
	for t in range(max(steps) + 1):
		inserted, removed = steps.get(t, ([], []))
		if t > 0 and len(inserted) != 1:
			return f"step {t}: {len(inserted)} edges enter"
		if any(v in seen for v in inserted):
			return f"step {t}: an edge enters a second time"
		if any(v not in current for v in removed):
			return f"step {t}: an edge leaves that is not current"
		current -= set(removed)
		for v in inserted:
			if any(u in current for u in crossing_graph.neighbors(v)):
				return f"step {t}: frame is not planar"
			current.add(v)
		seen.update(inserted)
		sizes.append(len(current))
	if seen != set(crossing_graph.nodes):
		return "not every edge appears"
	if min(sizes) != obj_value:
		return f"objective {obj_value}, but the smallest frame has {min(sizes)} edges"
	if pair is not None:
		if set(steps.get(0, ([], []))[0]) != set(pair[0]):
			return "initial frame is not the first set of the pair"
		if not set(pair[1]) <= current:
			return "final frame does not contain the second set of the pair"
	return None


def pair_problem(crossing_graph: nx.Graph, pair):
	"""Returns None if the two sets are disjoint independent sets with |I_1| <= |I_tau| (Variant 1a)."""
	A, B = set(pair[0]), set(pair[1])
	if len(A) != len(pair[0]) or len(B) != len(pair[1]) or A & B:
		return "the two sets are not disjoint"
	if any(u in S for S in (A, B) for v in S for u in crossing_graph.neighbors(v)):
		return "a set is not independent"
	if len(A) > len(B):
		return "|I_1| > |I_tau|"
	return None


def _hashable(x):
	return tuple(_hashable(y) for y in x) if isinstance(x, list) else x


def read_crossing_graph(dataset: str, file: str) -> nx.Graph:
	folder, is_drawing = DATASETS[dataset]
	graph, coordinates = read_hog(os.path.join(folder, file))
	return get_crossing_graph(graph, coordinates) if is_drawing else graph


def run_heuristics(crossing_graph: nx.Graph, name: str, time_limit_1a: float, seed: str):
	"""Returns the results of the six heuristics, the best story found as (objective value, frame events, frames), and
	the pairs of independent sets computed in Phase 1."""
	results, best, pairs = {}, None, {}
	for p1 in ["1a", "1b", "1c"]:
		extra = {}
		if p1 == "1a":
			pair, pair_time, wall_time, error = pareto_pair_with_time_limit(crossing_graph, time_limit_1a)
			extra["pair_wall_time_seconds"] = wall_time
			if pair is None:  # time limit reached (or the computation failed, then "error" says why)
				for p2 in ["2a", "2b"]:
					results[p1 + p2] = {"time_to_compute_initial_last_frame_seconds": None, "pair_wall_time_seconds": wall_time,
										"time_limit_pareto_optimal_reached": error is None, "computation_time_seconds": None,
										"obj_value": None, "frame_events": None}
					if error is not None:
						results[p1 + p2]["error"] = error
				continue
			pair = (list(pair[0]), list(pair[1]))
		else:
			t1 = perf_counter()
			pair = maximum_pair_b(crossing_graph) if p1 == "1b" else maximum_pair_a(crossing_graph)
			pair_time = perf_counter() - t1
		pairs[p1] = pair

		for p2 in ["2a", "2b"]:
			key = p1 + p2
			random.seed(f"{seed}-{name}-{key}")
			t1 = perf_counter()
			events = compute_frames_greedy(crossing_graph, initial_frame=pair[0], last_frame=pair[1], variation=p2[1])
			t2 = perf_counter()
			frames = FrameEvent.to_crossing_frames(events)
			obj_value = min(len(g.nodes) for g in frames)
			results[key] = {"time_to_compute_initial_last_frame_seconds": pair_time, **extra,
							"computation_time_seconds": t2 - t1, "obj_value": obj_value,
							"frame_events": [dataclasses.asdict(e) for e in events]}
			if best is None or obj_value > best[0]:
				best = (obj_value, events, frames)
	return results, best, pairs


def check(crossing_graph: nx.Graph, entry: dict, pairs: dict) -> list:
	problems = []
	if "1a" in pairs and pair_problem(crossing_graph, pairs["1a"]):
		problems.append(f"1a pair: {pair_problem(crossing_graph, pairs['1a'])}")
	for key in HEURISTICS:
		if entry[key]["obj_value"] is not None:
			problem = story_problem(crossing_graph, entry[key]["frame_events"], entry[key]["obj_value"], pairs[key[:2]])
			if problem:
				problems.append(f"{key}: {problem}")
	return problems


def run_ilp(crossing_graph: nx.Graph, best, time_limit: float) -> dict:
	from ilp import compute_frames_max_min  # imported here, so that --no-ilp works without gurobipy
	t1 = perf_counter()
	result = compute_frames_max_min(crossing_graph, frame_events=best[1], frames=best[2], verbose=False,
									max_time_seconds=time_limit)
	t2 = perf_counter()
	return {"computation_time_seconds": t2 - t1, "obj_value": result.objective_value, "best_bound": result.best_bound,
			"gap": result.gap, "frame_events": [dataclasses.asdict(e) for e in result]}


def run_dataset(dataset: str, args):
	folder, _ = DATASETS[dataset]
	files = sorted(f for f in os.listdir(folder) if f.endswith(".txt"))
	os.makedirs(args.out_dir, exist_ok=True)
	progress_file = os.path.join(args.out_dir, f"results_{dataset}.jsonl")
	out_file = os.path.join(args.out_dir, f"results_{dataset}.json")

	done = {}
	if os.path.exists(progress_file):
		with open(progress_file) as f:
			for line in f:
				if line.strip():
					entry = json.loads(line)
					done[entry["name"]] = entry
	print(f"== {dataset}: {len(files)} graphs, {len(done)} already done", flush=True)

	start, invalid = perf_counter(), []
	for i, file in enumerate(files):
		name = file[:-len(".txt")]
		if name in done:
			continue
		crossing_graph = read_crossing_graph(dataset, file)
		entry = {"name": name, "edges_of_the_crossing": crossing_graph.number_of_edges(),
				 "vertices_of_the_crossing": crossing_graph.number_of_nodes()}
		if crossing_graph.number_of_edges() == 0:
			print(f"  [{i + 1}/{len(files)}] {name}: no crossings, skipped", flush=True)
			continue
		crossing_graph.remove_nodes_from(list(nx.isolates(crossing_graph)))

		heuristic_results, best, pairs = run_heuristics(crossing_graph, name, args.time_limit_1a, args.seed)
		entry.update(heuristic_results)
		if not args.no_ilp:
			entry["ILP"] = run_ilp(crossing_graph, best, args.time_limit_ilp)
		problems = check(crossing_graph, entry, pairs) if args.check else []
		invalid += [f"{name} {p}" for p in problems]

		with open(progress_file, "a") as f:
			f.write(json.dumps(entry) + "\n")
		done[name] = entry
		values = ", ".join(f"{k} {entry[k]['obj_value']}" for k in HEURISTICS)
		ilp = "" if args.no_ilp else f", ILP {entry['ILP']['obj_value']:g} (bound {entry['ILP']['best_bound']:g})"
		print(f"  [{i + 1}/{len(files)}] {name}: {entry['edges_of_the_crossing']} crossings; {values}{ilp}"
			  f"{'; INVALID: ' + '; '.join(problems) if problems else ''}"
			  f"  (elapsed {(perf_counter() - start) / 60:.1f} min)", flush=True)

	with open(out_file, "w") as f:
		names = [file[:-len(".txt")] for file in files]
		json.dump([done[name] for name in names if name in done], f, indent=2)
	print(f"wrote {out_file}")
	if args.check:
		print(f"  checked the stories of this session: {len(invalid)} problems")
		for p in invalid:
			print("  INVALID:", p)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--all", action="store_true", help="all data sets, one after the other")
	group.add_argument("--dataset", nargs="+", choices=list(DATASETS), help="one or more data sets")
	parser.add_argument("--no-ilp", action="store_true", help="run only the heuristics")
	parser.add_argument("--time-limit-1a", type=float, default=10 * 60,
						help="time limit in seconds for Phase 1 of Variant 1a (default 600)")
	parser.add_argument("--time-limit-ilp", type=float, default=15 * 60,
						help="time limit in seconds for the ILP (default 900)")
	parser.add_argument("--check", action="store_true", help="check that every story of the heuristics is a valid planar story")
	parser.add_argument("--seed", default="0", help="seed of the random choices (default 0)")
	parser.add_argument("--out-dir", default="results", help="folder of the results files (default results)")
	args = parser.parse_args()
	for dataset in (DATASETS if args.all else args.dataset):
		run_dataset(dataset, args)
