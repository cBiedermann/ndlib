import networkx as nx

from ..DiffusionModel import DiffusionModel
import numpy as np
import future.utils

__author__ = "Giulio Rossetti"
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class IndependentCascadesModel(DiffusionModel):
    """
    Edge Parameters to be specified via ModelConfig

    :param threshold: The edge threshold. As default a value of 0.1 is assumed for all edges.
    """

    def __init__(self, graph, seed=None):
        """
        Model Constructor

        :param graph: A networkx graph object
        """
        super(self.__class__, self).__init__(graph, seed)
        self.available_statuses = {"Susceptible": 0, "Infected": 1, "Removed": 2}

        self.my_graph = graph

        temp = dict(zip(self.my_graph.edges, np.random.random_sample(len(self.my_graph.edges))))

        nx.set_edge_attributes(self.my_graph, temp, 'weight')

        self.parameters = {
            "model": {},
            "nodes": {},
            "edges": {
                "threshold": {
                    "descr": "Edge threshold",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.35,
                }
            },
        }

        self.name = "Independent Cascades"

    def iteration(self, node_status=True):
        """
        Execute a single model iteration

        :return: Iteration_id, Incremental node status (dictionary node->status)
        """
        self.clean_initial_status(self.available_statuses.values())
        actual_status = {
            node: nstatus for node, nstatus in future.utils.iteritems(self.status)
        }

        active_edges = []

        if self.actual_iteration == 0:
            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {
                    "iteration": 0,
                    "status": actual_status.copy(),
                    "node_count": node_count.copy(),
                    "status_delta": status_delta.copy(),
                    "active_edges": active_edges,
                }
            else:
                return {
                    "iteration": 0,
                    "status": {},
                    "node_count": node_count.copy(),
                    "status_delta": status_delta.copy(),
                    "active_edges": active_edges,
                }

        for u in self.graph.nodes:
            if self.status[u] != 1:
                continue

            neighbors = list(
                self.graph.neighbors(u)
            )  # neighbors and successors (in DiGraph) produce the same result

            # Standard threshold
            if len(neighbors) > 0:
                # 1/ len(neighbors)
                threshold = 0.35

                for v in neighbors:
                    if actual_status[v] == 0:
                        key = (u, v)

                        # Individual specified thresholds
                        if "threshold" in self.params["edges"]:
                            if key in self.params["edges"]["threshold"]:
                                threshold = self.params["edges"]["threshold"][key]
                            elif (v, u) in self.params["edges"][
                                "threshold"
                            ] and not self.graph.directed:
                                threshold = self.params["edges"]["threshold"][(v, u)]

                        print(self.my_graph.edges[key]['weight'])
                        if 0.9 <= self.my_graph.edges[key]['weight']:
                            actual_status[v] = 1
                            active_edges.append(key)

            actual_status[u] = 2

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {
                "iteration": self.actual_iteration - 1,
                "status": delta.copy(),
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
                "active_edges": active_edges,
            }
        else:
            return {
                "iteration": self.actual_iteration - 1,
                "status": {},
                "node_count": node_count.copy(),
                "status_delta": status_delta.copy(),
                "active_edges": active_edges,
            }
