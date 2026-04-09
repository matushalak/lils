import copy

import numpy as np

try:
    from ortools.graph import pywrapgraph

    def _make_solver():
        return pywrapgraph.SimpleMinCostFlow()

    def _add_arc(solver, start, end, capacity, cost):
        solver.AddArcWithCapacityAndUnitCost(start, end, capacity, cost)

    def _set_supply(solver, node, supply):
        solver.SetNodeSupply(node, supply)

    def _solve(solver):
        return solver.Solve()

    def _num_arcs(solver):
        return solver.NumArcs()

    def _tail(solver, arc):
        return solver.Tail(arc)

    def _head(solver, arc):
        return solver.Head(arc)

    def _flow(solver, arc):
        return solver.Flow(arc)

except ImportError:
    from ortools.graph.python import min_cost_flow

    def _make_solver():
        return min_cost_flow.SimpleMinCostFlow()

    def _add_arc(solver, start, end, capacity, cost):
        solver.add_arc_with_capacity_and_unit_cost(start, end, capacity, cost)

    def _set_supply(solver, node, supply):
        solver.set_node_supply(node, supply)

    def _solve(solver):
        return solver.solve()

    def _num_arcs(solver):
        return solver.num_arcs()

    def _tail(solver, arc):
        return solver.tail(arc)

    def _head(solver, arc):
        return solver.head(arc)

    def _flow(solver, arc):
        return solver.flow(arc)


class SolveMaxMatching:
    def __init__(self, nworkers, ntasks, k, value=10000, pairwise_lamb=0.1):
        self.nworkers = nworkers
        self.ntasks = ntasks
        self.value = value
        self.k = k

        self.source = 0
        self.sink = self.nworkers + self.ntasks + 1

        self.pairwise_cost = int(pairwise_lamb * value)

        self.supplies = [self.nworkers * self.k] + (self.ntasks + self.nworkers) * [0] + [-self.nworkers * self.k]
        self.start_nodes = list()
        self.end_nodes = list()
        self.capacities = list()
        self.common_costs = list()

        for work_idx in range(self.nworkers):
            self.start_nodes.append(self.source)
            self.end_nodes.append(work_idx + 1)
            self.capacities.append(self.k)
            self.common_costs.append(0)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(self.nworkers + 1 + task_idx)
                self.end_nodes.append(self.sink)
                self.capacities.append(1)
                self.common_costs.append(work_idx * self.pairwise_cost)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(work_idx + 1)
                self.end_nodes.append(self.nworkers + 1 + task_idx)
                self.capacities.append(1)

        self.nnodes = len(self.start_nodes)

    def solve(self, array):
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)

        self.array = self.value * array
        self.array = -self.array
        self.array = self.array.astype(np.int32)

        costs = copy.copy(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs.append(self.array[work_idx][task_idx])

        costs = np.array(costs)
        costs = costs.tolist()

        assert len(costs) == self.nnodes, "Length of costs should be {} but {}".format(self.nnodes, len(costs))

        min_cost = _make_solver()
        for idx in range(self.nnodes):
            _add_arc(min_cost, self.start_nodes[idx], self.end_nodes[idx], self.capacities[idx], costs[idx])
        for idx in range(self.ntasks + self.nworkers + 2):
            _set_supply(min_cost, idx, self.supplies[idx])

        _solve(min_cost)
        results = list()
        for arc in range(_num_arcs(min_cost)):
            if _tail(min_cost, arc) != self.source and _head(min_cost, arc) != self.sink:
                if _flow(min_cost, arc) > 0:
                    results.append([_tail(min_cost, arc) - 1, _head(min_cost, arc) - self.nworkers - 1])

        results_np = np.zeros_like(array)
        for i, j in results:
            results_np[i][j] = 1
        return results, results_np


class SimpleHungarianSolver:
    def __init__(self, nworkers, ntasks, value=10000):
        self.nworkers = nworkers
        self.ntasks = ntasks
        self.value = value

        self.source = 0
        self.sink = self.nworkers + self.ntasks + 1

        self.supplies = [self.nworkers] + (self.ntasks + self.nworkers) * [0] + [-self.nworkers]
        self.start_nodes = list()
        self.end_nodes = list()
        self.capacities = list()
        self.common_costs = list()

        for work_idx in range(self.nworkers):
            self.start_nodes.append(self.source)
            self.end_nodes.append(work_idx + 1)
            self.capacities.append(1)
            self.common_costs.append(0)

        for task_idx in range(self.ntasks):
            self.start_nodes.append(self.nworkers + 1 + task_idx)
            self.end_nodes.append(self.sink)
            self.capacities.append(1)
            self.common_costs.append(0)

        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                self.start_nodes.append(work_idx + 1)
                self.end_nodes.append(self.nworkers + 1 + task_idx)
                self.capacities.append(1)

        self.nnodes = len(self.start_nodes)

    def solve(self, array):
        assert array.shape == (self.nworkers, self.ntasks), "Wrong array shape, it should be ({}, {})".format(self.nworkers, self.ntasks)

        self.array = self.value * array
        self.array = -self.array
        self.array = self.array.astype(np.int32)

        costs = copy.copy(self.common_costs)
        for work_idx in range(self.nworkers):
            for task_idx in range(self.ntasks):
                costs.append(self.array[work_idx][task_idx])

        costs = np.array(costs)
        costs = costs.tolist()

        assert len(costs) == self.nnodes, "Length of costs should be {} but {}".format(self.nnodes, len(costs))

        min_cost = _make_solver()
        for idx in range(self.nnodes):
            _add_arc(min_cost, self.start_nodes[idx], self.end_nodes[idx], self.capacities[idx], costs[idx])
        for idx in range(self.ntasks + self.nworkers + 2):
            _set_supply(min_cost, idx, self.supplies[idx])

        _solve(min_cost)
        results = list()
        for arc in range(_num_arcs(min_cost)):
            if _tail(min_cost, arc) != self.source and _head(min_cost, arc) != self.sink:
                if _flow(min_cost, arc) > 0:
                    results.append([_tail(min_cost, arc) - 1, _head(min_cost, arc) - self.nworkers - 1])

        results_np = np.zeros_like(array)
        for i, j in results:
            results_np[i][j] = 1
        return results, results_np
