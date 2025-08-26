from frame import *
import gurobipy as gp
import networkx as nx
from collections import UserList


class ILPResult(UserList):
    """
    A wrapper for the result of the ILP. This wrapper behaves like the normal list[FrameEvent] return type, but
    also offers additional information about the solution, i.e., whether it is optimal and if not, what the best
    bound or optimality gap is.
    """
    objective_value: float
    best_bound: float
    gap: float
    time_limit_seconds: int

    def __init__(self, frame_events: list[FrameEvent], objective_value: float, best_bound: float, gap: float,
                 time_limit_seconds: int):
        super().__init__(frame_events)
        self.objective_value = objective_value
        self.best_bound = best_bound
        self.gap = gap
        self.time_limit_seconds = time_limit_seconds


def compute_frames_max_min(crossing_graph: nx.Graph, num_frames: int = None, frame_events:[] = None, frames:[] = None,
                           verbose: bool = True, max_time_seconds: int | None = None) -> ILPResult:
    """
    Integer linear program for computing an edge story that maximizes the minimum number of edges in a frame.

    :param crossing_graph: The conflict graph of the graph drawing whose edge story is generated.
    :param num_frames: The number of frames to generate. If num_frames = None, then the number of edges in the graph
    drawing, i.e., the number of vertices in the crossing_graph is used.
    :param verbose: Set the verbosity of the ILP solver.

    :returns: A list of frame events representing an optimal solution of the edge story of the graph represented by the
    conflict graph.
    """

    if num_frames is None:
        num_frames = len(crossing_graph.nodes)

    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()

        with gp.Model(env=env) as model:
            if max_time_seconds is not None:
                model.setParam('TimeLimit', max_time_seconds)

            x_vars, z_vars, min_var = add_variables(crossing_graph, model, num_frames)

            add_planarity_constraints(crossing_graph, model, num_frames, x_vars)
            add_edge_existence_constraints(crossing_graph, model, num_frames, x_vars)
            add_min_number_of_edges_in_frames_constraints(crossing_graph, model, num_frames, x_vars, min_var)
            add_continuity_constraints(crossing_graph, model, num_frames, x_vars, z_vars)
            # This is implied, but I think it makes it a bit faster
            model.addConstr(min_var <= len(crossing_graph.nodes)/2)

            # Giving an initial feasible solution to the ILP
            if frame_events and frames:
                for index, graph in enumerate(frames):
                    for node in graph.nodes:
                        x_vars[node, index].Start = 1

                for frame_event in frame_events:
                    if frame_event.frame_type == FrameEventType.IN:
                        z_vars[frame_event.edge, frame_event.time].Start = 1

                model.update()

            # objective function that aims to reduce symmetric solutions.
            # It appears that this doesn't have a performance increase.
            # model.setObjective(
            #     min_var - (1/(num_frames*num_frames+1))*gp.quicksum((t+1)*gp.quicksum(z_vars[e, t] for e in crossing_graph.nodes)
            #                for t in range(num_frames)), gp.GRB.MAXIMIZE)

            model.setObjective(min_var, gp.GRB.MAXIMIZE)

            # Gurobi Parameters
            # model.setParam("MIPFocus", 3)
            # model.setParam("Cuts", 3)
            # model.setParam("Heuristics", 0.01)
            # model.setParam("Symmetry", 2)

            model.optimize()

            if model.SolCount== 0:
                frame_events = []
            else:
                frame_events = [FrameEvent(e, t, FrameEventType.IN) for (e, t) in z_vars.keys() if z_vars[e, t].X == 1]

                for e in crossing_graph.nodes:
                    frame_events.append(
                        FrameEvent(e, max(t for t in range(num_frames) if x_vars[e, t].X == 1)+1, FrameEventType.OUT)
                    )

                frame_events = sorted(frame_events)

                while frame_events[-1].frame_type == FrameEventType.OUT:
                    frame_events.pop()

            return ILPResult(frame_events=frame_events, objective_value=model.Objval, best_bound=model.ObjBound,
                             gap=model.MIPGap, time_limit_seconds=max_time_seconds)


def add_variables(crossing_graph: nx.Graph, model: gp.Model, num_frames: int):
    """
    Returns the necessary variables for the ILP formulation. The naming convention is based on the formulation of the
    paper. A binary variable x_var[e, t] = 1 signifies that edge e is present in frame t. The binary variable
    z_var[e, t] = 1 means that edge e appears in frame t. The continuous variable min_var serves as a linearization of
    the objective function min_var represents the minimum number of edges in any frame.
    """

    x_vars = model.addVars(((e, t) for e in crossing_graph.nodes for t in range(num_frames)),
                           vtype=gp.GRB.BINARY, name="x")
    z_vars = model.addVars(((e, t) for e in crossing_graph.nodes for t in range(num_frames)),
                           vtype=gp.GRB.BINARY, name="z")
    min_var = model.addVar(vtype=gp.GRB.CONTINUOUS, name="min_var", lb=0, ub=num_frames)

    return x_vars, z_vars, min_var


def add_planarity_constraints(crossing_graph: nx.Graph, model: gp.Model, num_frames: int, x_vars):
    model.addConstrs(x_vars[e, t] + x_vars[f, t] <= 1 for e, f in crossing_graph.edges for t in range(num_frames))


def add_edge_existence_constraints(crossing_graph: nx.Graph, model: gp.Model, num_frames: int, x_vars):
    model.addConstrs(gp.quicksum(x_vars[e, t] for t in range(num_frames)) >= 1 for e in crossing_graph.nodes)


def add_min_number_of_edges_in_frames_constraints(crossing_graph: nx.Graph, model: gp.Model, num_frames: int, x_vars,
                                                  min_var):
    model.addConstrs(gp.quicksum(x_vars[e, t] for e in crossing_graph.nodes) >= min_var for t in range(num_frames))


def add_continuity_constraints(crossing_graph: nx.Graph, model: gp.Model, num_frames: int, x_vars, z_vars):
    model.addConstrs(gp.quicksum(z_vars[e, t] for t in range(num_frames)) == 1 for e in crossing_graph.nodes)
    model.addConstrs(z_vars[e, t] + x_vars[e, t-1] >= x_vars[e, t] for e in crossing_graph.nodes
                     for t in range(1, num_frames))
    model.addConstrs(z_vars[e, 0] >= x_vars[e, 0] for e in crossing_graph.nodes)
    model.addConstrs(gp.quicksum(z_vars[e, t] for e in crossing_graph.nodes) <= 1 for t in range(1, num_frames))
