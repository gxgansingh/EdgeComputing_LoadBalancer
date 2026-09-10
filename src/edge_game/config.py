"""Simulation configuration and canonical resource units."""

from dataclasses import dataclass


# Canonical physical units used by the simulation resource model and audit.
# CPU: 1 GC/s = 10^9 CPU cycles per second.
# Memory: GB means gigabytes.
# Bandwidth: Mbps means megabits per second.
# Latency: milliseconds.
# Energy: joules.
# Queue: number of tasks.
RESOURCE_UNITS = {
    "cpu": "GC/s",
    "memory": "GB",
    "bandwidth": "Mbps",
    "latency": "ms",
    "energy": "J",
    "queue": "tasks",
}

RESOURCE_UNIT_DESCRIPTIONS = {
    "cpu": "Gigacycles per Second",
    "memory": "Gigabytes",
    "bandwidth": "Megabits per Second",
    "latency": "Milliseconds",
    "energy": "Joules",
    "queue": "Number of Tasks",
}


@dataclass(frozen=True)
class SimulationConfig:
    """Global simulation settings."""

    seed: int = 42

    simulation_steps: int = 200
    simulation_time_step: float = 1.0

    tasks_per_step: int = 3

    number_of_servers: int = 3
    nodes_per_server: int = 5

    # CPU capacity in GC/s (Gigacycles per Second).
    minimum_cpu_capacity: float = 10.0
    maximum_cpu_capacity: float = 30.0

    # Memory capacity in GB (Gigabytes).
    minimum_memory_capacity: float = 8.0
    maximum_memory_capacity: float = 32.0

    # Network bandwidth capacity in Mbps (Megabits per Second).
    minimum_bandwidth: float = 10.0
    maximum_bandwidth: float = 100.0

    # Task resource requirements use the same physical units as node capacity.
    minimum_cpu_demand: float = 1.0
    maximum_cpu_demand: float = 8.0

    minimum_memory_demand: float = 1.0
    maximum_memory_demand: float = 8.0

    minimum_bandwidth_demand: float = 1.0
    maximum_bandwidth_demand: float = 15.0

    minimum_workload_size: float = 1.0
    maximum_workload_size: float = 10.0

    # Maximum acceptable communication latency in milliseconds.
    minimum_latency_requirement: float = 5.0
    maximum_latency_requirement: float = 50.0

    # Task energy budget in Joules.
    minimum_energy_budget: float = 1.0
    maximum_energy_budget: float = 20.0

    # Workload priority distribution.
    critical_priority_share: float = 0.10
    high_priority_share: float = 0.20
    medium_priority_share: float = 0.40
    low_priority_share: float = 0.30

    # Priority-aware service configuration.
    priority_service_weight_critical: float = 1.75
    priority_service_weight_high: float = 1.50
    priority_service_weight_medium: float = 1.25
    priority_service_weight_low: float = 1.00

    task_cpu_rate: float = 2.5

    # Dynamic service configuration.
    reference_cpu_capacity: float = 20.0

    # Mean-Field local-state mapping configuration.
    mean_field_cpu_state_weight: float = 0.35
    mean_field_memory_state_weight: float = 0.15
    mean_field_bandwidth_state_weight: float = 0.15
    mean_field_queue_state_weight: float = 0.20
    mean_field_energy_state_weight: float = 0.15

    mean_field_queue_normalization: float = 10.0

    # Research utility configuration.
    utility_priority_reward_weight: float = 1.0
    utility_latency_cost_weight: float = 1.0
    utility_resource_cost_weight: float = 1.0
    utility_queue_cost_weight: float = 1.0
    utility_energy_cost_weight: float = 1.0

    utility_queue_normalization: float = 10.0

    # Mean-Field numerical configuration.
    mean_field_state_points: int = 41
    mean_field_control_points: int = 11

    mean_field_tolerance: float = 1e-4
    mean_field_policy_tolerance: float = 1e-4
    mean_field_raw_policy_tolerance: float = 1e-3

    mean_field_max_iterations: int = 1000

    mean_field_damping: float = 0.05
    mean_field_policy_damping: float = 0.05

    mean_field_diffusion: float = 0.01
    mean_field_discount_factor: float = 0.95

    # FPK inner solver configuration.
    fpk_time_step: float = 0.001
    fpk_tolerance: float = 1e-5
    fpk_max_iterations: int = 5000

    # Dynamic resource-summary configuration.
    resource_summary_update_interval: int = 1

    # Maximum number of queued/active tasks allowed by the feasibility filter.
    maximum_queue_length: int = 20

    # Active external network-load factor used by the feasibility model.
    network_load: float = 0.0

    # Network-load experiment configuration.
    network_load_levels: tuple[float, ...] = (
        0.10,
        0.30,
        0.50,
        0.70,
        0.90,
    )

    # Network load affects the latency pressure seen by feasibility filtering.
    network_load_latency_multiplier: float = 1.0

    # Feasibility-audit latency model.
    base_network_latency: float = 2.0
    latency_load_penalty: float = 8.0
    latency_queue_penalty: float = 0.75
    latency_workload_penalty: float = 0.50

    # Feasibility-audit energy model.
    # Energy consumed per unit of CPU work, expressed so the resulting
    # task energy estimate is measured in Joules.
    energy_per_cpu_work_unit: float = 0.25

    # Repeated experiment configuration.
    experiment_repetitions: int = 10
    experiment_seed_start: int = 42

    # Load-balancer benchmark configuration.
    benchmark_simulation_steps: int = 2500
    benchmark_experiment_repetitions: int = 10
    benchmark_seed_start: int = 42
    benchmark_number_of_servers: int = 1
    benchmark_nodes_per_server: int = 3
    benchmark_mean_field_state_points: int = 11
    benchmark_mean_field_max_iterations: int = 20
    benchmark_fpk_max_iterations: int = 50

    workload_scenarios: tuple[str, ...] = (
        "default",
        "moderate_congestion",
        "high_congestion",
    )

    # Scalability experiment configuration.
    scalability_scenarios: tuple[str, ...] = (
        "small",
        "medium",
        "large",
        "stress",
    )

    output_directory: str = "outputs"
