"""Task model for the edge computing simulation."""

from dataclasses import dataclass


@dataclass
class Task:
    """Represents one workload task."""

    task_id: int
    priority: int

    # CPU demand in GC/s (Gigacycles per Second).
    cpu_demand: float
    # Memory demand in GB (Gigabytes).
    memory_demand: float
    # Network bandwidth demand in Mbps (Megabits per Second).
    bandwidth_demand: float

    # Maximum acceptable latency in milliseconds.
    latency_requirement: float
    # Maximum task energy budget in Joules.
    energy_budget: float
    workload_size: float

    priority_class: str | None = None
    priority_score: float | None = None

    arrival_time: float = 0.0
    start_time: float | None = None
    completion_time: float | None = None

    assigned_node_id: int | None = None
    remaining_work: float | None = None

    def __post_init__(self) -> None:
        """Initialize derived task attributes."""

        if self.priority_class is None:
            priority_mapping = {
                1: "P4",
                2: "P3",
                3: "P2",
                4: "P1",
            }

            self.priority_class = priority_mapping.get(
                self.priority,
                "P4",
            )

        if self.priority_score is None:
            self.priority_score = float(
                self.priority
            )

        if self.remaining_work is None:
            self.remaining_work = float(
                self.workload_size
            )

    @property
    def response_time(self) -> float | None:
        """Return task response time after completion."""

        if self.completion_time is None:
            return None

        return float(
            self.completion_time
            - self.arrival_time
        )

    @property
    def waiting_time(self) -> float | None:
        """Return waiting time before processing starts."""

        if self.start_time is None:
            return None

        return float(
            self.start_time
            - self.arrival_time
        )

    @property
    def processing_time(self) -> float | None:
        """Return actual processing time."""

        if (
            self.start_time is None
            or self.completion_time is None
        ):
            return None

        return float(
            self.completion_time
            - self.start_time
        )