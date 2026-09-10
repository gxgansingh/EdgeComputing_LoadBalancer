"""Simulation metric collection."""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..entities.edge_node import EdgeNode
from ..entities.task import Task
from ..algorithms.utility import calculate_outcome_utility
from ..config import SimulationConfig


@dataclass
class MetricCollector:
    """Collect task-level and time-series system metrics."""

    response_times: List[float] = field(
        default_factory=list
    )

    utilities: List[float] = field(
        default_factory=list
    )

    utilization_history: List[float] = field(
        default_factory=list
    )

    load_variance_history: List[float] = field(
        default_factory=list
    )

    fairness_history: List[float] = field(
        default_factory=list
    )

    queue_length_history: List[float] = field(
        default_factory=list
    )

    node_utilization_history: List[dict] = field(
        default_factory=list
    )

    selection_records: List[dict] = field(
        default_factory=list
    )

    decision_records: List[dict] = field(
        default_factory=list
    )

    allocation_contexts: dict[int, dict] = field(
        default_factory=dict
    )

    priority_arrivals: dict[int, int] = field(
        default_factory=lambda: {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }
    )

    priority_completions: dict[int, int] = field(
        default_factory=lambda: {
            1: 0,
            2: 0,
            3: 0,
            4: 0,
        }
    )

    successful_tasks: int = 0
    total_tasks: int = 0
    rejected_tasks: int = 0

    def record_arrival(
        self,
        task: Task,
    ) -> None:
        """Record a task arrival."""
        self.total_tasks += 1

        self.priority_arrivals[
            task.priority
        ] = (
            self.priority_arrivals.get(
                task.priority,
                0,
            )
            + 1
        )

    def record_rejection(
        self,
        task: Task,
    ) -> None:
        """Record a rejected task."""
        task.rejected = True
        self.rejected_tasks += 1

    def record_allocation(
        self,
        task: Task,
        node: EdgeNode,
    ) -> None:
        """Store the action context until the task completes."""
        self.allocation_contexts[
            task.task_id
        ] = {
            "node_cpu_capacity": float(node.cpu_capacity),
            "node_memory_capacity": float(node.memory_capacity),
            "node_bandwidth_capacity": float(node.bandwidth_capacity),
            "node_energy_capacity": float(node.energy_capacity),
            "queue_length_at_assignment": int(node.queue_length),
        }

    def record_selection(
        self,
        task: Task,
        node: EdgeNode,
    ) -> None:
        """Record the node-selection decision before allocation."""
        self.selection_records.append(
            {
                "task_id": int(task.task_id),
                "priority": int(task.priority),
                "node_id": int(node.node_id),
                "cpu_capacity": float(
                    node.cpu_capacity
                ),
                "load_ratio": float(
                    node.load_ratio()
                ),
                "queue_length": int(
                    node.queue_length
                ),
            }
        )

    def record_selection_diagnostics(
        self,
        task: Task,
        policy_name: str,
        candidates: list[EdgeNode],
        selected_node: EdgeNode,
        diagnostics: dict,
    ) -> None:
        """Record the complete candidate-set decision context."""
        record = {
            "task_id": int(task.task_id),
            "priority": int(task.priority),
            "policy_name": policy_name,
            "candidate_count": int(
                diagnostics["candidate_count"]
            ),
            "candidate_node_ids": diagnostics[
                "candidate_node_ids"
            ],
            "candidate_scores": diagnostics[
                "candidate_scores"
            ],
            "selected_node_id": int(
                selected_node.node_id
            ),
            "selected_rank": int(
                diagnostics["selected_rank"]
            ),
            "selected_score": float(
                diagnostics["selected_score"]
            ),
            "best_score": float(
                diagnostics["best_score"]
            ),
            "worst_score": float(
                diagnostics["worst_score"]
            ),
            "score_margin": float(
                diagnostics["score_margin"]
            ),
            "score_tie_count": int(
                diagnostics["score_tie_count"]
            ),
            "state": float(
                diagnostics["state"]
            ),
            "control": float(
                diagnostics["control"]
            ),
            "mean_field_score": float(
                diagnostics["mean_field_score"]
            ),
            "cpu_load": float(diagnostics["cpu_load"]),
            "memory_load": float(diagnostics["memory_load"]),
            "bandwidth_load": float(
                diagnostics["bandwidth_load"]
            ),
            "queue_pressure": float(
                diagnostics["queue_pressure"]
            ),
            "energy_pressure": float(
                diagnostics["energy_pressure"]
            ),
        }

        self.decision_records.append(record)

    def record_completion(
        self,
        task: Task,
        config: SimulationConfig,
    ) -> None:
        """Record a completed task and calculate realized utility."""
        if task.response_time is not None:
            self.response_times.append(
                task.response_time
            )

            context = self.allocation_contexts.pop(
                task.task_id,
                None,
            )

            if context is not None:
                utility = calculate_outcome_utility(
                    task=task,
                    response_time=task.response_time,
                    node_cpu_capacity=context[
                        "node_cpu_capacity"
                    ],
                    node_memory_capacity=context[
                        "node_memory_capacity"
                    ],
                    node_bandwidth_capacity=context[
                        "node_bandwidth_capacity"
                    ],
                    queue_length_at_assignment=context[
                        "queue_length_at_assignment"
                    ],
                    node_energy_capacity=context[
                        "node_energy_capacity"
                    ],
                    config=config,
                )
                self.utilities.append(
                    utility
                )

        self.successful_tasks += 1

        self.priority_completions[
            task.priority
        ] = (
            self.priority_completions.get(
                task.priority,
                0,
            )
            + 1
        )

    def record_state(
        self,
        nodes: list[EdgeNode],
        simulation_step: int | None = None,
    ) -> None:
        """Record a system-state snapshot."""
        if not nodes:
            return

        loads = np.array(
            [
                node.load_ratio()
                for node in nodes
            ],
            dtype=float,
        )

        utilization = float(
            np.mean(loads)
        )

        load_variance = float(
            np.var(loads)
        )

        total_load = float(
            np.sum(loads)
        )

        squared_load_sum = float(
            np.sum(loads ** 2)
        )

        if squared_load_sum <= 1e-12:
            fairness = 1.0
        else:
            fairness = (
                total_load ** 2
            ) / (
                len(loads)
                * squared_load_sum
            )

        average_queue = float(
            np.mean(
                [
                    node.queue_length
                    for node in nodes
                ]
            )
        )

        if simulation_step is not None:
            for node in nodes:
                self.node_utilization_history.append(
                    {
                        "simulation_step": int(simulation_step),
                        "node_id": int(node.node_id),
                        "server_id": int(node.server_id),
                        "cpu_utilization": float(
                            node.load_ratio() * 100.0
                        ),
                    }
                )

        self.utilization_history.append(
            utilization
        )

        self.load_variance_history.append(
            load_variance
        )

        self.fairness_history.append(
            fairness
        )

        self.queue_length_history.append(
            average_queue
        )

    def priority_success_ratios(self) -> dict[int, float]:
        """Calculate completion ratios for each priority class."""
        ratios: dict[int, float] = {}

        for priority, arrivals in (
            self.priority_arrivals.items()
        ):
            completions = (
                self.priority_completions.get(
                    priority,
                    0,
                )
            )

            ratios[priority] = (
                completions / arrivals
                if arrivals > 0
                else 0.0
            )

        return ratios

    def summary(
        self,
        nodes: list[EdgeNode],
    ) -> dict:
        """Return aggregated simulation metrics."""
        self.record_state(nodes)

        priority_ratios = (
            self.priority_success_ratios()
        )

        return {
            "utility_mean": (
                float(
                    np.mean(
                        self.utilities
                    )
                )
                if self.utilities
                else 0.0
            ),
            "response_time_mean": (
                float(
                    np.mean(
                        self.response_times
                    )
                )
                if self.response_times
                else 0.0
            ),
            "throughput": float(
                self.successful_tasks
            ),
            "success_ratio": (
                self.successful_tasks
                / self.total_tasks
                if self.total_tasks
                else 0.0
            ),
            "rejected_tasks": float(
                self.rejected_tasks
            ),
            "resource_utilization": (
                float(
                    np.mean(
                        self.utilization_history
                    )
                )
                if self.utilization_history
                else 0.0
            ),
            "load_variance": (
                float(
                    np.mean(
                        self.load_variance_history
                    )
                )
                if self.load_variance_history
                else 0.0
            ),
            "jains_fairness_index": (
                float(
                    np.mean(
                        self.fairness_history
                    )
                )
                if self.fairness_history
                else 1.0
            ),
            "average_queue_length": (
                float(
                    np.mean(
                        self.queue_length_history
                    )
                )
                if self.queue_length_history
                else 0.0
            ),
            "priority_success_ratio": (
                float(
                    np.mean(
                        list(
                            priority_ratios.values()
                        )
                    )
                )
                if priority_ratios
                else 0.0
            ),
        }