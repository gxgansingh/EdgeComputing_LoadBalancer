"""Resource-aware feasibility filtering and audit generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..config import SimulationConfig
from ..entities.edge_node import EdgeNode
from ..entities.edge_server import EdgeServer
from ..entities.task import Task


@dataclass(frozen=True)
class FeasibilityAudit:
    """Record every resource feasibility check for one task-node pair."""

    task_id: int
    priority: int
    priority_class: str
    node_id: int
    server_id: int
    cpu_pass: bool
    memory_pass: bool
    bandwidth_pass: bool
    latency_pass: bool
    energy_pass: bool
    queue_pass: bool
    feasible: bool
    # CPU fields are measured in GC/s.
    cpu_available: float
    cpu_required: float
    # Memory fields are measured in GB.
    memory_available: float
    memory_required: float
    # Bandwidth fields are measured in Mbps.
    bandwidth_available: float
    bandwidth_required: float
    # Latency fields are measured in ms.
    estimated_latency: float
    latency_limit: float
    # Energy fields are measured in Joules.
    estimated_energy: float
    energy_budget: float
    # Queue fields are measured as number of tasks.
    queue_length: int
    queue_limit: int
    rejection_reasons: str


def estimate_node_latency(
    task: Task,
    node: EdgeNode,
    config: SimulationConfig,
) -> float:
    """Estimate task latency in milliseconds from current resource pressure."""
    load_pressure = (
        node.load_ratio()
        + node.memory_load_ratio()
        + node.bandwidth_load_ratio()
    ) / 3.0

    network_pressure = (
        config.network_load_latency_multiplier
        * getattr(config, "network_load", 0.0)
    )

    return float(
        config.base_network_latency
        + config.latency_load_penalty
        * load_pressure
        * (1.0 + network_pressure)
        + config.latency_queue_penalty * node.queue_length
        + config.latency_workload_penalty * task.workload_size
    )


def estimate_task_energy(
    task: Task,
    node: EdgeNode,
    config: SimulationConfig,
) -> float:
    """Estimate task execution energy in Joules."""
    utilization_factor = 1.0 + node.load_ratio()

    return float(
        config.energy_per_cpu_work_unit
        * task.cpu_demand
        * task.workload_size
        * utilization_factor
    )


def evaluate_node_feasibility(
    task: Task,
    node: EdgeNode,
    config: SimulationConfig,
) -> FeasibilityAudit:
    """Evaluate all resource constraints without selecting a node.

    The feasibility checks use physical units directly: GC/s for CPU, GB for
    memory, Mbps for bandwidth, ms for latency, J for energy, and task count
    for queue length. This function only determines feasibility; it never
    performs the final game-theoretic node selection.
    """
    cpu_pass = node.available_cpu >= task.cpu_demand
    memory_pass = node.available_memory >= task.memory_demand
    bandwidth_pass = node.available_bandwidth >= task.bandwidth_demand

    estimated_latency = estimate_node_latency(
        task=task,
        node=node,
        config=config,
    )
    latency_pass = estimated_latency <= task.latency_requirement

    estimated_energy = estimate_task_energy(
        task=task,
        node=node,
        config=config,
    )
    energy_pass = (
        estimated_energy <= task.energy_budget
        and estimated_energy <= node.energy_capacity
    )

    queue_pass = node.queue_length < config.maximum_queue_length

    reasons: list[str] = []
    if not cpu_pass:
        reasons.append("CPU")
    if not memory_pass:
        reasons.append("MEMORY")
    if not bandwidth_pass:
        reasons.append("BANDWIDTH")
    if not latency_pass:
        reasons.append("LATENCY")
    if not energy_pass:
        reasons.append("ENERGY")
    if not queue_pass:
        reasons.append("QUEUE")

    return FeasibilityAudit(
        task_id=task.task_id,
        priority=task.priority,
        priority_class=task.priority_class or "P4",
        node_id=node.node_id,
        server_id=node.server_id,
        cpu_pass=cpu_pass,
        memory_pass=memory_pass,
        bandwidth_pass=bandwidth_pass,
        latency_pass=latency_pass,
        energy_pass=energy_pass,
        queue_pass=queue_pass,
        feasible=not reasons,
        cpu_available=float(node.available_cpu),
        cpu_required=float(task.cpu_demand),
        memory_available=float(node.available_memory),
        memory_required=float(task.memory_demand),
        bandwidth_available=float(node.available_bandwidth),
        bandwidth_required=float(task.bandwidth_demand),
        estimated_latency=estimated_latency,
        latency_limit=float(task.latency_requirement),
        estimated_energy=estimated_energy,
        energy_budget=float(task.energy_budget),
        queue_length=int(node.queue_length),
        queue_limit=int(config.maximum_queue_length),
        rejection_reasons=",".join(reasons),
    )


def build_feasibility_audit(
    task: Task,
    nodes: Iterable[EdgeNode],
    config: SimulationConfig,
) -> list[FeasibilityAudit]:
    """Audit every candidate node before policy selection."""
    return [
        evaluate_node_feasibility(
            task=task,
            node=node,
            config=config,
        )
        for node in nodes
    ]


def filter_feasible_nodes(
    task: Task,
    nodes: Iterable[EdgeNode],
    config: SimulationConfig | None = None,
) -> list[EdgeNode]:
    """Return feasible nodes after all configured resource checks."""
    effective_config = config or SimulationConfig()
    return [
        node
        for node in nodes
        if evaluate_node_feasibility(
            task=task,
            node=node,
            config=effective_config,
        ).feasible
    ]


def filter_feasible_servers(
    task: Task,
    servers: Iterable[EdgeServer],
    config: SimulationConfig | None = None,
) -> list[EdgeServer]:
    """Return servers containing at least one feasible node."""
    effective_config = config or SimulationConfig()
    return [
        server
        for server in servers
        if any(
            evaluate_node_feasibility(
                task=task,
                node=node,
                config=effective_config,
            ).feasible
            for node in server.nodes
        )
    ]
