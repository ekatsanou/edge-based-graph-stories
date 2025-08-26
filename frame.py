from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from itertools import groupby
import networkx as nx


class FrameEventType(IntEnum):
    OUT = 0
    IN = 1


@dataclass(frozen=True)
class FrameEvent:
    """
    A FrameEvent is a lightweight way to represent a graph story. Instead of storing an entire graph for each frame,
    a list of frame events only stores which edge appears/disappears at a certain time.

    :param edge: The edge that appears/disappears.
    :param time: The frame at which the edge changes.
    :param frame_type: Signifies if the edge appears (IN) or if the edge disappears (OUT).
    """
    edge: tuple
    time: int
    frame_type: FrameEventType

    def __lt__(self, other):
        return (self.time, self.frame_type) < (other.time, other.frame_type)

    @staticmethod
    def to_frames(frame_events: [FrameEvent], vertices: []) -> [nx.Graph]:
        """
        A function that converts a list of frame events to a list of graphs that represent each frame,
        assuming that the list of frame events is sorted.

        :param frame_events: A list of frame events.
        :param vertices: A list of all vertices of the graph.
        :returns: A list of graphs.
        """
        frames = []

        current_graph = nx.Graph()
        current_graph.add_nodes_from(vertices)

        for time, group in groupby(frame_events, lambda e: e.time):
            for frame_event in group:
                if frame_event.frame_type == FrameEventType.IN:
                    current_graph.add_edge(*frame_event.edge)
                elif frame_event.frame_type == FrameEventType.OUT:
                    current_graph.remove_edge(*frame_event.edge)
            frames.append(current_graph.copy())

        return frames

    @staticmethod
    def to_crossing_frames(frame_events: [FrameEvent]) -> [nx.Graph]:
        """
        A function that converts a list of frame events to a list of graphs that represent each frame,
        assuming that the list of frame events is sorted.

        :param frame_events: A list of frame events.
        :returns: A list of graphs.
        """
        frames = []

        current_graph = nx.Graph()
        # frames.append(current_graph.copy())
        for time, group in groupby(frame_events, lambda e: e.time):
            for frame_event in group:
                if frame_event.frame_type == FrameEventType.IN:
                    current_graph.add_node(frame_event.edge)
                elif frame_event.frame_type == FrameEventType.OUT:
                    current_graph.remove_node(frame_event.edge)
            frames.append(current_graph.copy())

        return frames
