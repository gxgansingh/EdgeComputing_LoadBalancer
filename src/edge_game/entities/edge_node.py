"""Heterogeneous edge-node model."""

from dataclasses import dataclass, field

from .task import Task


@dataclass
class EdgeNode:
    """Represents an edge node managed by an edge server."""

    node_id: int
    server_id: int

    # CPU capacity and available CPU are measured in GC/s.
    cpu_capacity: float
    # Memory capacity and available memory are measured in GB.
    memory_capacity: float
    # Network bandwidth capacity and availability are measured in Mbps.
    bandwidth_capacity: float
    # Energy capacity is measured in Joules.
    energy_capacity: float

    # Queue length is the number of active/queued tasks.
    queue_length: int = 0

    available_cpu: float = 0.0
    available_memory: float = 0.0
    available_bandwidth: float = 0.0

    active_tasks: list[Task] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """Initialize available resources."""

        if self.available_cpu == 0.0:
            self.available_cpu = (
                self.cpu_capacity
            )

        if self.available_memory == 0.0:
            self.available_memory = (
                self.memory_capacity
            )

        if self.available_bandwidth == 0.0:
            self.available_bandwidth = (
                self.bandwidth_capacity
            )

    def can_process(
        self,
        task: Task,
    ) -> bool:
        """Return whether the node satisfies CPU, memory, and bandwidth requirements.

        CPU is measured in GC/s, memory in GB, and bandwidth in Mbps.
        """

        return (
            self.available_cpu
            >= task.cpu_demand
            and self.available_memory
            >= task.memory_demand
            and self.available_bandwidth
            >= task.bandwidth_demand
        )

    def allocate_task(
        self,
        task: Task,
        current_time: float,
    ) -> None:
        """Allocate node resources to a task."""

        if not self.can_process(task):
            raise ValueError(
                f"Node {self.node_id} cannot allocate "
                f"task {task.task_id}."
            )

        self.available_cpu -= (
            task.cpu_demand
        )

        self.available_memory -= (
            task.memory_demand
        )

        self.available_bandwidth -= (
            task.bandwidth_demand
        )

        task.assigned_node_id = (
            self.node_id
        )

        task.start_time = current_time

        self.active_tasks.append(
            task
        )

        self.queue_length = len(
            self.active_tasks
        )

    def _priority_service_weight(
        self,
        task: Task,
        critical_priority_weight: float = 1.75,
        high_priority_weight: float = 1.50,
        medium_priority_weight: float = 1.25,
        low_priority_weight: float = 1.0,
    ) -> float:
        """Return service weight for the task priority."""

        weights = {
            1: low_priority_weight,
            2: medium_priority_weight,
            3: high_priority_weight,
            4: critical_priority_weight,
        }

        return float(
            weights.get(
                task.priority,
                low_priority_weight,
            )
        )

    def effective_service_rate(
        self,
        task: Task,
        base_cpu_rate: float,
        reference_cpu_capacity: float,
        low_priority_weight: float = 1.0,
        medium_priority_weight: float = 1.25,
        high_priority_weight: float = 1.50,
        critical_priority_weight: float = 1.75,
    ) -> float:
        """Calculate the effective processing rate in GC/s.

        The returned rate is expressed in the same GC/s unit as CPU capacity.
        """

        if reference_cpu_capacity <= 0.0:
            raise ValueError(
                "reference_cpu_capacity must be positive."
            )

        if not self.active_tasks:
            return 0.0

        capacity_factor = (
            self.cpu_capacity
            / reference_cpu_capacity
        )

        node_service_rate = (
            base_cpu_rate
            * capacity_factor
        )

        task_weight = (
            self._priority_service_weight(
                task=task,
                critical_priority_weight=(
                    critical_priority_weight
                ),
                high_priority_weight=(
                    high_priority_weight
                ),
                medium_priority_weight=(
                    medium_priority_weight
                ),
                low_priority_weight=(
                    low_priority_weight
                ),
            )
        )

        total_weight = sum(
            self._priority_service_weight(
                task=active_task,
                critical_priority_weight=(
                    critical_priority_weight
                ),
                high_priority_weight=(
                    high_priority_weight
                ),
                medium_priority_weight=(
                    medium_priority_weight
                ),
                low_priority_weight=(
                    low_priority_weight
                ),
            )
            for active_task in self.active_tasks
        )

        if total_weight <= 0.0:
            return 0.0

        return (
            node_service_rate
            * task_weight
            / total_weight
        )

    def process_tasks(
        self,
        time_step: float,
        cpu_rate: float,
        reference_cpu_capacity: float = 20.0,
        low_priority_weight: float = 1.0,
        medium_priority_weight: float = 1.25,
        high_priority_weight: float = 1.50,
        critical_priority_weight: float = 1.75,
    ) -> list[Task]:
        """Advance active tasks using weighted service."""

        completed_tasks: list[Task] = []

        for task in list(
            self.active_tasks
        ):
            service_rate = (
                self.effective_service_rate(
                    task=task,
                    base_cpu_rate=cpu_rate,
                    reference_cpu_capacity=(
                        reference_cpu_capacity
                    ),
                    critical_priority_weight=(
                        critical_priority_weight
                    ),
                    high_priority_weight=(
                        high_priority_weight
                    ),
                    medium_priority_weight=(
                        medium_priority_weight
                    ),
                    low_priority_weight=(
                        low_priority_weight
                    ),
                )
            )

            task.remaining_work -= (
                service_rate
                * time_step
            )

            if task.remaining_work <= 0.0:
                completed_tasks.append(
                    task
                )

        return completed_tasks

    def release_task(
        self,
        task: Task,
        current_time: float,
    ) -> None:
        """Release resources held by a completed task."""

        if task not in self.active_tasks:
            return

        self.active_tasks.remove(
            task
        )

        self.available_cpu = min(
            self.cpu_capacity,
            self.available_cpu
            + task.cpu_demand,
        )

        self.available_memory = min(
            self.memory_capacity,
            self.available_memory
            + task.memory_demand,
        )

        self.available_bandwidth = min(
            self.bandwidth_capacity,
            self.available_bandwidth
            + task.bandwidth_demand,
        )

        task.remaining_work = 0.0
        task.completion_time = (
            current_time
        )

        self.queue_length = len(
            self.active_tasks
        )

    def load_ratio(self) -> float:
        """Return current CPU load ratio."""

        if self.cpu_capacity <= 0.0:
            return 1.0

        return 1.0 - (
            self.available_cpu
            / self.cpu_capacity
        )

    def memory_load_ratio(self) -> float:
        """Return current memory load ratio."""

        if self.memory_capacity <= 0.0:
            return 1.0

        return 1.0 - (
            self.available_memory
            / self.memory_capacity
        )

    def bandwidth_load_ratio(self) -> float:
        """Return current bandwidth load ratio."""

        if self.bandwidth_capacity <= 0.0:
            return 1.0

        return 1.0 - (
            self.available_bandwidth
            / self.bandwidth_capacity
        )