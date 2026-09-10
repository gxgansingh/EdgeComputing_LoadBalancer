"""Experiment execution utilities."""

from dataclasses import dataclass
from typing import Callable

from ..algorithms.candidate_filter import (
    build_feasibility_audit,
)
from ..algorithms.utility import (
    calculate_utility,
)
from ..config import SimulationConfig
from ..environment import SimulationEnvironment
from ..metrics.collector import MetricCollector
from ..entities.task import Task


@dataclass
class ExperimentResult:
    """Store the result of one policy experiment."""

    policy_name: str
    metrics: dict
    selection_records: list[dict]
    decision_records: list[dict]
    filtering_records: list[dict]
    node_utilization_records: list[dict]


def run_policy_experiment(
    config: SimulationConfig,
    policy_name: str,
    policy,
) -> ExperimentResult:
    """Run the dynamic workload simulation using one policy."""
    environment = SimulationEnvironment(
        config=config
    )

    metrics = MetricCollector()

    nodes = environment.all_nodes()

    filtering_records: list[dict] = []

    next_task_id = 0

    for simulation_step in range(
        config.simulation_steps
    ):
        tasks = environment.create_tasks(
            starting_task_id=next_task_id
        )

        next_task_id += len(tasks)

        for task in tasks:
            metrics.record_arrival(
                task=task
            )

            feasibility_audits = build_feasibility_audit(
                task=task,
                nodes=nodes,
                config=config,
            )

            filtering_records.extend(
                [
                    {
                        "task_id": audit.task_id,
                        "priority": audit.priority,
                        "priority_class": audit.priority_class,
                        "node_id": audit.node_id,
                        "server_id": audit.server_id,
                        "cpu_pass": audit.cpu_pass,
                        "memory_pass": audit.memory_pass,
                        "bandwidth_pass": audit.bandwidth_pass,
                        "latency_pass": audit.latency_pass,
                        "energy_pass": audit.energy_pass,
                        "queue_pass": audit.queue_pass,
                        "feasible": audit.feasible,
                        # Legacy field names are retained for API compatibility.
                        "cpu_available": audit.cpu_available,
                        "cpu_required": audit.cpu_required,
                        "memory_available": audit.memory_available,
                        "memory_required": audit.memory_required,
                        "bandwidth_available": audit.bandwidth_available,
                        "bandwidth_required": audit.bandwidth_required,
                        "estimated_latency": audit.estimated_latency,
                        "latency_limit": audit.latency_limit,
                        "estimated_energy": audit.estimated_energy,
                        "energy_budget": audit.energy_budget,
                        "queue_length": audit.queue_length,
                        "queue_limit": audit.queue_limit,
                        # Explicit unit-qualified columns for research artifacts.
                        "cpu_available_gc_s": audit.cpu_available,
                        "cpu_required_gc_s": audit.cpu_required,
                        "memory_available_gb": audit.memory_available,
                        "memory_required_gb": audit.memory_required,
                        "bandwidth_available_mbps": audit.bandwidth_available,
                        "bandwidth_required_mbps": audit.bandwidth_required,
                        "estimated_latency_ms": audit.estimated_latency,
                        "latency_limit_ms": audit.latency_limit,
                        "estimated_energy_j": audit.estimated_energy,
                        "energy_budget_j": audit.energy_budget,
                        "queue_length_tasks": audit.queue_length,
                        "queue_limit_tasks": audit.queue_limit,
                        "rejection_reasons": audit.rejection_reasons,
                    }
                    for audit in feasibility_audits
                ]
            )

            candidates = [
                node
                for node, audit in zip(nodes, feasibility_audits)
                if audit.feasible
            ]

            if not candidates:
                metrics.record_rejection(
                    task=task
                )
                continue

            selected_node = (
                policy.select_node(
                    task=task,
                    candidates=candidates,
                )
            )

            if selected_node is None:
                metrics.record_rejection(
                    task=task
                )
                continue

            metrics.record_selection(
                task=task,
                node=selected_node,
            )

            metrics.record_selection_diagnostics(
                task=task,
                policy_name=policy_name,
                candidates=candidates,
                selected_node=selected_node,
                diagnostics=policy.selection_diagnostics(
                    task=task,
                    candidates=candidates,
                    selected_node=selected_node,
                ),
            )

            metrics.record_allocation(
                task=task,
                node=selected_node,
            )

            selected_node.allocate_task(
                task=task,
                current_time=(
                    environment.current_time
                ),
            )

        metrics.record_state(
            nodes=nodes,
            simulation_step=simulation_step,
        )

        completed_tasks = (
            environment.advance_time()
        )

        for task in completed_tasks:
            metrics.record_completion(
                task=task,
                config=config,
            )

    return ExperimentResult(
        policy_name=policy_name,
        metrics=metrics.summary(
            nodes=nodes
        ),
        selection_records=list(
            metrics.selection_records
        ),
        decision_records=list(
            metrics.decision_records
        ),
        filtering_records=filtering_records,
        node_utilization_records=list(
            metrics.node_utilization_history
        ),
    )