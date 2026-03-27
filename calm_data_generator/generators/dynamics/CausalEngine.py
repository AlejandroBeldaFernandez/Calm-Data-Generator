"""
CausalEngine — DAG-based causal cascade propagation.

Allows defining directed acyclic graphs (DAGs) of variable dependencies and
propagating a perturbation from a trigger variable to all its descendants,
using user-defined non-linear functions.

Usage example (IoT sensor chain):

    dag = {
        "temperature": [],
        "pressure": [
            {"parent": "temperature", "func": "linear", "params": {"slope": 1.2}}
        ],
        "sensor_fail_rate": [
            {"parent": "pressure", "func": "exponential", "params": {"scale": 0.001, "rate": 0.3}}
        ],
    }
    engine = CausalEngine(dag)
    df_result = engine.apply_cascade(df, trigger_col="temperature", delta=np.full(len(df), 5.0))
"""

from collections import deque
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from calm_data_generator.generators.utils.propagation import apply_func


class CausalEngine:
    """
    Propagates a perturbation from a trigger variable through a DAG of causal dependencies.

    The DAG is defined as a dict mapping each node name to a list of parent-edge
    specifications. Each edge specifies the parent column, the transfer function,
    and its parameters.

    The propagation uses a **differential** approach:
        delta_child = f(v_parent + delta_parent) - f(v_parent)

    This preserves the current absolute values and only propagates the incremental change.
    """

    def __init__(self, dag_config: Dict[str, List[dict]]):
        """
        Args:
            dag_config: Dict mapping node names to a list of parent-edge dicts.
                Each edge dict must have:
                    - "parent" (str): name of the parent node
                    - "func" (str|callable): transfer function
                    - "params" (dict, optional): function parameters

                Nodes with no parents are roots (empty list value).

        Raises:
            ValueError: If the DAG contains cycles or references unknown nodes.
        """
        self._dag = dag_config
        self._topo_order = self._topological_sort()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply_cascade(
        self,
        df: pd.DataFrame,
        trigger_col: str,
        delta: np.ndarray,
        rows: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        """
        Applies delta to trigger_col and propagates to all descendants in topological order.

        Args:
            df: DataFrame to modify in-place.
            trigger_col: Column that receives the initial perturbation. Must be a node in the DAG.
            delta: Array of perturbation values. If rows is None, length must equal len(df).
                   If rows is provided, length must equal len(rows).
            rows: Index labels of rows to affect. None means all rows.

        Returns:
            df (modified in-place).

        Raises:
            ValueError: If trigger_col is not in the DAG.
            ValueError: If a required column is not present in df.
        """
        if trigger_col not in self._dag:
            raise ValueError(
                f"trigger_col '{trigger_col}' not found in DAG. "
                f"Known nodes: {list(self._dag.keys())}"
            )

        if rows is None:
            rows = df.index

        delta = np.asarray(delta, dtype=float)

        # accumulated deltas keyed by node name
        accumulated = {trigger_col: delta}

        # Apply trigger delta
        if trigger_col in df.columns:
            df.loc[rows, trigger_col] = df.loc[rows, trigger_col].values + delta

        # Walk descendants in topological order
        trigger_idx = self._topo_order.index(trigger_col)
        for node in self._topo_order[trigger_idx + 1:]:
            edges = self._dag.get(node, [])
            delta_node = np.zeros(len(rows), dtype=float)

            for edge in edges:
                parent = edge["parent"]
                if parent not in accumulated:
                    # This parent was not in the affected subtree
                    continue

                func = edge.get("func", "linear")
                params = edge.get("params", {})
                d_parent = accumulated[parent]

                if parent in df.columns:
                    # Use values BEFORE the parent delta was applied (original values)
                    v_parent = df.loc[rows, parent].values - d_parent
                else:
                    v_parent = np.zeros(len(rows))

                # Differential: f(v + d) - f(v)
                delta_node += (
                    apply_func(func, params, v_parent + d_parent)
                    - apply_func(func, params, v_parent)
                )

            if np.any(delta_node != 0):
                accumulated[node] = delta_node
                if node in df.columns:
                    df.loc[rows, node] = df.loc[rows, node].values + delta_node

        return df

    def get_topological_order(self) -> List[str]:
        """Returns all nodes in topological order (roots first)."""
        return list(self._topo_order)

    def validate(self) -> None:
        """
        Validates the DAG structure.

        Raises:
            ValueError: If cycles are detected or edges reference unknown nodes.
        """
        for node, edges in self._dag.items():
            for edge in edges:
                parent = edge.get("parent")
                if parent not in self._dag:
                    raise ValueError(
                        f"Node '{node}' references unknown parent '{parent}'. "
                        f"All nodes must be declared as keys in dag_config."
                    )
        # Re-run topological sort — it raises on cycles
        self._topological_sort()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _topological_sort(self) -> List[str]:
        """
        Kahn's algorithm for topological ordering.

        Returns:
            Nodes in topological order (parents before children).

        Raises:
            ValueError: If a cycle is detected.
        """
        # Build in-degree map and adjacency list (parent → children)
        in_degree: Dict[str, int] = {node: 0 for node in self._dag}
        children: Dict[str, List[str]] = {node: [] for node in self._dag}

        for node, edges in self._dag.items():
            for edge in edges:
                parent = edge.get("parent")
                if parent is None:
                    continue
                if parent not in self._dag:
                    raise ValueError(
                        f"Node '{node}' references unknown parent '{parent}'. "
                        f"All nodes must be declared as keys in dag_config."
                    )
                in_degree[node] += 1
                children[parent].append(node)

        queue = deque(node for node, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self._dag):
            cycle_nodes = [n for n in self._dag if n not in order]
            raise ValueError(
                f"Cycle detected in DAG involving nodes: {cycle_nodes}"
            )

        return order
