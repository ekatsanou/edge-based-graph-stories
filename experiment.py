import csv
import dataclasses
import itertools
import multiprocessing
import os
import json
import signal
from typing import Tuple, Dict, List, Any

import networkx as nx
from time import perf_counter
from collections import defaultdict
from matplotlib import pyplot as plt

from frame import FrameEvent
from frame_calculations import maximum_pair_b, maximum_pair_optimum_tree, maximum_pair_optimum_decomposition, \
    maximum_pair_a, pareto_optimal_pair
from ilp import compute_frames_max_min
from io_tools.crossing_graph import get_crossing_graph
from io_tools.read_graph import read_hog
from modified_greedy import compute_frames_greedy
from tqdm import tqdm
from itertools import product


class OutputFile:

    def __init__(self, filename: str):
        self.filename = filename

        if not os.path.exists(filename):
            with open(self.filename, "w") as f:
                json.dump([], f)

    def update(self, result):
        with open(self.filename, "r") as f:
            results = json.load(f)

        results.append(result)

        with open(self.filename, "w") as f:
            json.dump(results, f, indent=2)

    def names_in_file(self) -> set[str]:
        with open(self.filename, "r") as f:
            results = json.load(f)
            return set(entry["name"] for entry in results)


class ExperimentManager:
    _cmp_frames = {}
    """
    I.e, '1a' or '2c', ...
    """
    heuristic_variants: list[str]
    outfile: OutputFile
    time_limit_ilp_seconds: None | int
    time_limit_pareto_optimal_pair_seconds: None | int

    def __init__(self, outfile_name: str):
        self.heuristic_variants = []
        self.outfile = OutputFile(outfile_name)
        self._tqdm_progress_bar = None
        self.time_limit_ilp_seconds = None
        self.time_limit_pareto_optimal_pair_seconds = None

        self._cmp_frames = {
        "1": self._pareto_optimal_pair_with_timeout,
        "2": maximum_pair_b,
        "3": maximum_pair_a
    }

    def add_heuristic_variants(self, *args):
        self.heuristic_variants.extend(*args)

    def run_hog_suite(self, directory: str, remove_isolated_vertices: bool = True):
        graphs_in_out_file = self.outfile.names_in_file()
        graphs = [g for g in os.listdir(directory) if g.split(".")[0] not in graphs_in_out_file]
        self._tqdm_progress_bar = tqdm(self._generate_hog_crossing_graphs(directory, graphs),
                                       total=len(graphs), desc="Running Experiments")

        for crossing_graph, graph_name in self._tqdm_progress_bar:
            self._tqdm_progress_bar.set_description(f'Running {graph_name}')
            self._tqdm_progress_bar.set_postfix({"num_nodes": len(crossing_graph.nodes),
                                                 "num_edges": len(crossing_graph.edges)})

            if crossing_graph.number_of_edges() == 0:
                self._tqdm_progress_bar.set_description(f'{graph_name} is planar. Skip.')
                continue

            if remove_isolated_vertices:
                isolated_vertices = list(nx.isolates(crossing_graph))
                crossing_graph.remove_nodes_from(isolated_vertices)

            # results = {"name": graph_name} | self._run_heuristics(crossing_graph) | self._run_ilp(crossing_graph)
            heuristic_results, best_h_result, best_frames = self._run_heuristics(crossing_graph)
            results = {"name": graph_name} | heuristic_results | self._run_ilp(crossing_graph, best_h_result, best_frames)
            self.outfile.update(results)

    def _generate_hog_crossing_graphs(self, directory: str, graphs: list[str]) -> (str, nx.Graph):
        for graph_file in graphs:
            graph, coordinates = read_hog(os.path.join(directory, graph_file))
            # CHANGE THIS LINE IF RUNNING THE CODE ON THE REAL AND THE RANDOM GRAPHS
            yield graph, graph_file.split(".")[0]
            # yield get_crossing_graph(graph, coordinates), graph_file.split(".")[0]

    def _run_heuristics(self, crossing_graph: nx.Graph) -> tuple[
        dict[str, dict[str, float | int | list[dict[str, Any]]]], Any | None, Any | None]:
        result = {}

        best_heuristic_obj = float('-inf')
        best_h_result = None
        best_frames = None

        for frame_variant, selection_variant in self.heuristic_variants:
            t1_init_last_frame = perf_counter()
            init_frame, final_frame = self._cmp_frames[frame_variant](crossing_graph)
            t2_init_last_frame = perf_counter()

            if init_frame is None and final_frame is None:
                result[f'{frame_variant}{selection_variant}'] = {
                    "time_to_compute_initial_last_frame_seconds": t2_init_last_frame - t1_init_last_frame,
                    "time_limit_pareto_optimal_reached": True,
                    "computation_time_seconds": None,
                    "obj_value": None,
                    "frame_events": None
                }
                continue

            t1_heuristic = perf_counter()
            h_result = compute_frames_greedy(crossing_graph, initial_frame=init_frame, last_frame=final_frame,
                                           variation=selection_variant)
            t2_heuristic = perf_counter()

            frames = FrameEvent.to_crossing_frames(h_result)
            heuristic_obj = min(len(g.nodes) for g in frames)

            result[f'{frame_variant}{selection_variant}'] = {
                "time_to_compute_initial_last_frame_seconds": t2_init_last_frame-t1_init_last_frame,
                "computation_time_seconds": t2_heuristic - t1_heuristic,
                "obj_value": heuristic_obj,
                "frame_events": [dataclasses.asdict(event) for event in h_result]
            }

            if heuristic_obj > best_heuristic_obj:
                best_heuristic_obj = heuristic_obj
                best_h_result = h_result
                best_frames = frames

        return result, best_h_result, best_frames


    def _run_ilp(self, crossing_graph: nx.Graph, frame_events:[], frames:[]) -> dict:
        t1 = perf_counter()
        ilp_result = compute_frames_max_min(crossing_graph, frame_events=frame_events, frames=frames, verbose=False, max_time_seconds=self.time_limit_ilp_seconds)
        t2 = perf_counter()

        return {"ILP": {
            "computation_time_seconds": t2 - t1,
            "obj_value": ilp_result.objective_value,
            "best_bound": ilp_result.best_bound,
            "gap": ilp_result.gap,
            "frame_events": [dataclasses.asdict(event) for event in ilp_result]
        }}

    def _pareto_optimal_pair_with_timeout(self, crossing_graph: nx.Graph):
        mp_manager = multiprocessing.Manager()
        result_container = mp_manager.dict()
        process = multiprocessing.Process(
            target=_pareto_optimal_pair_worker,
            args=(crossing_graph, result_container)
        )

        process.start()
        process.join(timeout=self.time_limit_pareto_optimal_pair_seconds)

        if process.is_alive():
            process.terminate()
            process.join()
            return None, None

        return result_container.get('result', (None, None))


def _pareto_optimal_pair_worker(crossing_graph, return_dict):
        try:
            res = pareto_optimal_pair(crossing_graph)
            return_dict['result'] = res
        except Exception as e:
            return_dict['result'] = (None, None)

if __name__ == '__main__':
    manager = ExperimentManager("res_trees.json")
    variants = list(itertools.product(["1", "2", "3"], ["a", "b"]))
    manager.add_heuristic_variants(variants)
    manager.time_limit_ilp_seconds = 15*60
    manager.time_limit_pareto_optimal_pair_seconds = 10*60
    manager.run_hog_suite(os.path.join("graphgenerator", "trees"))
