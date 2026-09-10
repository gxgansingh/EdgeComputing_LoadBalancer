"""Module-2 Mean-Field Game experiment with no-priority load balancing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import MeanFieldPolicy
from ..config import SimulationConfig
from .runner import build_mean_field_policy


def _run_single_experiment(
    config: SimulationConfig,
    seed: int,
) -> tuple[dict, object]:
    """Run one no-priority Mean-Field Game experiment."""
    seed_config = replace(
        config,
        seed=seed,
    )

    policy, equilibrium_diagnostics = (
        build_mean_field_policy(
            config=seed_config,
            ablation_variant="no_priority",
        )
    )

    result = run_policy_experiment(
        config=seed_config,
        policy_name="module2_no_priority_mean_field",
        policy=policy,
    )

    metrics = dict(result.metrics)

    metrics["seed"] = seed
    metrics["policy"] = (
        "no_priority_mean_field"
    )

    metrics["equilibrium_converged"] = (
        equilibrium_diagnostics["converged"]
    )

    metrics["equilibrium_iterations"] = (
        equilibrium_diagnostics["iterations"]
    )

    metrics["equilibrium_residual"] = (
        equilibrium_diagnostics[
            "distribution_residual"
        ]
    )

    metrics["equilibrium_policy_residual"] = (
        equilibrium_diagnostics[
            "policy_residual"
        ]
    )

    return metrics, result


def run_module2_experiment(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Run Module-2 using the no-priority Mean-Field policy."""
    output_path = Path(output_directory)

    raw_path = (
        output_path / "raw"
    )

    aggregated_path = (
        output_path / "aggregated"
    )

    figures_path = (
        output_path / "figures"
    )

    raw_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    filtering_rows = []
    selection_rows = []

    for seed in seeds:
        metrics, result = _run_single_experiment(
            config=config,
            seed=seed,
        )
        rows.append(metrics)

        for record in result.filtering_records:
            filtering_rows.append(
                {
                    "seed": int(seed),
                    "policy": "no_priority_mean_field",
                    **record,
                }
            )

        for record in result.selection_records:
            selection_rows.append(
                {
                    "seed": int(seed),
                    "policy": "no_priority_mean_field",
                    **record,
                }
            )

    raw_results = pd.DataFrame(rows)

    raw_results.to_csv(
        raw_path / "module2_raw.csv",
        index=False,
    )

    numeric_columns = (
        raw_results.select_dtypes(
            include="number"
        ).columns
    )

    summary_rows = []

    for column in numeric_columns:
        if column == "seed":
            continue

        summary_rows.append(
            {
                "metric": column,
                "mean": raw_results[
                    column
                ].mean(),
                "std": raw_results[
                    column
                ].std(
                    ddof=1
                ),
                "min": raw_results[
                    column
                ].min(),
                "max": raw_results[
                    column
                ].max(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        aggregated_path
        / "module2_summary.csv",
        index=False,
    )

    filtering_audit = pd.DataFrame(filtering_rows)
    selection_audit = pd.DataFrame(selection_rows)

    filtering_audit.to_csv(
        raw_path / "resource_filtering_audit.csv",
        index=False,
    )

    filtering_summary = _summarize_filtering_audit(
        filtering_audit=filtering_audit,
    )
    filtering_summary.to_csv(
        aggregated_path / "resource_filtering_summary.csv",
        index=False,
    )

    _generate_figures(
        raw_results=raw_results,
        figures_directory=figures_path,
    )

    _generate_resource_filtering_figure(
        filtering_audit=filtering_audit,
        selection_audit=selection_audit,
        figures_directory=figures_path,
    )

    _generate_report(
        config=config,
        seeds=seeds,
        summary=summary,
        filtering_summary=filtering_summary,
        output_path=(
            output_path
            / "module2_report.md"
        ),
    )

    return raw_results


def _generate_figures(
    raw_results: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Generate Module-2 metric figures."""
    figure_metrics = [
        (
            "response_time_mean",
            "Response Time",
            "Response time",
        ),
        (
            "throughput",
            "Throughput",
            "Completed tasks",
        ),
        (
            "success_ratio",
            "Success Ratio",
            "Success ratio",
        ),
        (
            "resource_utilization",
            "Resource Utilization",
            "Utilization",
        ),
        (
            "load_variance",
            "Load Variance",
            "Load variance",
        ),
        (
            "jains_fairness_index",
            "Jain's Fairness Index",
            "Fairness index",
        ),
        (
            "average_queue_length",
            "Average Queue Length",
            "Queue length",
        ),
        (
            "rejected_tasks",
            "Rejected Tasks",
            "Rejected tasks",
        ),
        (
            "utility_mean",
            "Mean Utility",
            "Utility",
        ),
    ]

    for metric, title, ylabel in figure_metrics:
        if metric not in raw_results.columns:
            continue

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        axis.plot(
            raw_results["seed"],
            raw_results[metric],
            marker="o",
        )

        axis.set_title(
            f"Module-2: {title}"
        )

        axis.set_xlabel(
            "Seed"
        )

        axis.set_ylabel(
            ylabel
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        figure.savefig(
            figures_directory
            / f"module2_{metric}.png",
            dpi=180,
        )

        plt.close(
            figure
        )


def _summarize_filtering_audit(
    filtering_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize resource-filtering outcomes for Module-2."""
    if filtering_audit.empty:
        return pd.DataFrame(
            columns=["metric", "value"]
        )

    total_checks = len(filtering_audit)
    feasible_checks = int(
        filtering_audit["feasible"].sum()
    )
    filtered_checks = total_checks - feasible_checks
    tasks = filtering_audit["task_id"].nunique()
    seeds = filtering_audit["seed"].nunique()

    rows = [
        {"metric": "total_node_checks", "value": float(total_checks)},
        {"metric": "feasible_node_checks", "value": float(feasible_checks)},
        {"metric": "filtered_node_checks", "value": float(filtered_checks)},
        {
            "metric": "filtering_ratio",
            "value": filtered_checks / max(total_checks, 1),
        },
        {
            "metric": "search_space_reduction_ratio",
            "value": filtered_checks / max(total_checks, 1),
        },
        {
            "metric": "average_candidates_per_task",
            "value": total_checks / max(tasks, 1),
        },
        {
            "metric": "average_feasible_nodes_per_task",
            "value": feasible_checks / max(tasks, 1),
        },
        {"metric": "tasks", "value": float(tasks)},
        {"metric": "seeds", "value": float(seeds)},
    ]

    reasons = (
        "CPU",
        "MEMORY",
        "BANDWIDTH",
        "LATENCY",
        "ENERGY",
        "QUEUE",
    )
    counts = {reason: 0 for reason in reasons}

    for value in filtering_audit["rejection_reasons"].fillna(""):
        for reason in str(value).split(","):
            if reason in counts:
                counts[reason] += 1

    rows.extend(
        {
            "metric": f"rejection_reason_{reason.lower()}",
            "value": float(count),
        }
        for reason, count in counts.items()
    )

    return pd.DataFrame(rows)


def _generate_resource_filtering_figure(
    filtering_audit: pd.DataFrame,
    selection_audit: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Generate the professor-facing filtering-to-selection audit figure."""
    if filtering_audit.empty:
        return

    selected_task = None
    selected_seed = None
    selected_node = None

    if not selection_audit.empty:
        first_selection = selection_audit.sort_values(
            ["seed", "task_id"]
        ).iloc[0]
        selected_task = int(first_selection["task_id"])
        selected_seed = int(first_selection["seed"])
        selected_node = int(first_selection["node_id"])

    if selected_task is None:
        first_audit = filtering_audit.sort_values(
            ["seed", "task_id", "node_id"]
        ).iloc[0]
        selected_task = int(first_audit["task_id"])
        selected_seed = int(first_audit["seed"])

    task_audit = filtering_audit.loc[
        (filtering_audit["seed"] == selected_seed)
        & (filtering_audit["task_id"] == selected_task)
    ].copy()

    if task_audit.empty:
        return

    task_audit = task_audit.sort_values("node_id").reset_index(drop=True)
    feasible_nodes = [
        int(row["node_id"])
        for _, row in task_audit.iterrows()
        if bool(row["feasible"])
    ]

    first = task_audit.iloc[0]
    priority_class = str(first["priority_class"])

    def resource_cell(
        available: float,
        required: float,
        passed: bool,
        precision: int = 2,
    ) -> str:
        status = "PASS" if passed else "FAIL"
        return (
            f"{available:.{precision}f} / {required:.{precision}f}\n"
            f"{status}"
        )

    def upper_bound_cell(
        actual: float,
        limit: float,
        passed: bool,
        precision: int = 2,
    ) -> str:
        status = "PASS" if passed else "FAIL"
        return (
            f"{actual:.{precision}f} / {limit:.{precision}f}\n"
            f"{status}"
        )

    def queue_cell(
        current: int,
        limit: int,
        passed: bool,
    ) -> str:
        status = "PASS" if passed else "FAIL"
        return f"{current} / {limit}\n{status}"

    table_rows = []
    for _, row in task_audit.iterrows():
        selected = (
            selected_node is not None
            and int(row["node_id"]) == selected_node
        )
        table_rows.append(
            [
                f"E{int(row['node_id'])}",
                resource_cell(
                    row["cpu_available"],
                    row["cpu_required"],
                    bool(row["cpu_pass"]),
                ),
                resource_cell(
                    row["memory_available"],
                    row["memory_required"],
                    bool(row["memory_pass"]),
                ),
                resource_cell(
                    row["bandwidth_available"],
                    row["bandwidth_required"],
                    bool(row["bandwidth_pass"]),
                ),
                upper_bound_cell(
                    row["estimated_latency"],
                    row["latency_limit"],
                    bool(row["latency_pass"]),
                ),
                upper_bound_cell(
                    row["estimated_energy"],
                    row["energy_budget"],
                    bool(row["energy_pass"]),
                ),
                queue_cell(
                    int(row["queue_length"]),
                    int(row["queue_limit"]),
                    bool(row["queue_pass"]),
                ),
                (
                    "YES\nSELECTED"
                    if selected
                    else "YES"
                    if bool(row["feasible"])
                    else "NO"
                ),
                str(row["rejection_reasons"])
                if str(row["rejection_reasons"])
                else "-",
            ]
        )

    figure, axis = plt.subplots(
        figsize=(17, 9)
    )
    axis.axis("off")

    figure.suptitle(
        "Resource Filtering and Feasible Edge Selection Audit",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    requirements = (
        f"Task T{selected_task} | {priority_class} | Seed {selected_seed}\n"
        f"CPU {first['cpu_required']:.2f} GC/s | "
        f"Memory {first['memory_required']:.2f} GB | "
        f"Bandwidth {first['bandwidth_required']:.2f} Mbps | "
        f"Latency ≤ {first['latency_limit']:.2f} ms | "
        f"Energy ≤ {first['energy_budget']:.2f} J | "
        f"Queue < {int(first['queue_limit'])} tasks"
    )
    axis.text(
        0.5,
        0.90,
        "TASK REQUIREMENTS\n" + requirements,
        ha="center",
        va="center",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.6", "fill": False},
    )

    table = axis.table(
        cellText=table_rows,
        colLabels=[
            "Edge",
            "CPU (GC/s)\navail / req",
            "Memory (GB)\navail / req",
            "Bandwidth (Mbps)\navail / req",
            "Latency (ms)\nest. / limit",
            "Energy (J)\nest. / budget",
            "Queue (Tasks)\ncurrent / limit",
            "Feasible /\nSelection",
            "Rejection\nReason",
        ],
        cellLoc="center",
        colLoc="center",
        colWidths=[0.07, 0.105, 0.105, 0.105, 0.105, 0.105, 0.10, 0.105, 0.13],
        bbox=[0.015, 0.31, 0.97, 0.49],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.7)

    for column_index in range(9):
        table[(0, column_index)].set_text_props(fontweight="bold")

    for row_index, row_data in enumerate(table_rows, start=1):
        is_selected = "SELECTED" in row_data[7]
        is_feasible = row_data[7].startswith("YES")
        for column_index in range(9):
            cell = table[(row_index, column_index)]
            if column_index in (0, 7) or is_selected:
                cell.set_text_props(fontweight="bold")
            if column_index == 7 and is_feasible:
                cell.set_text_props(fontweight="bold")

    feasible_text = (
        "FEASIBLE EDGE SET = "
        + "{" + ", ".join(f"E{node}" for node in feasible_nodes) + "}"
        if feasible_nodes
        else "FEASIBLE EDGE SET = {}"
    )
    selection_text = (
        f"MFG / GAME-THEORETIC SELECTION = E{selected_node}"
        if selected_node is not None
        else "MFG / GAME-THEORETIC SELECTION = NO SELECTION"
    )

    axis.text(
        0.5,
        0.23,
        feasible_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    axis.text(
        0.5,
        0.14,
        selection_text,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.5", "fill": False},
    )
    axis.text(
        0.5,
        0.065,
        "Filtering determines capability. The MFG policy selects only from the feasible edge set.",
        ha="center",
        va="center",
        fontsize=10,
    )

    output_path = figures_directory / "resource_filtering_selection_audit.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

def _generate_report(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    summary: pd.DataFrame,
    filtering_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the Module-2 experiment report."""
    architecture = """
IoT Devices
    |
    v
Edge Gateway
    |
    v
Edge Server
    |
    v
Edge Nodes
    |
    +-- Node 1
    +-- Node 2
    +-- Node 3
    +-- ...
"""

    lines = [
        "# Module-2 Experiment Report",
        "",
        "## Architecture",
        "",
        (
            "Simple IoT-to-Edge architecture "
            "with no-priority load balancing."
        ),
        "",
        "```text",
        architecture.strip(),
        "```",
        "",
        "## Load-Balancing Policy",
        "",
        (
            "The experiment uses the "
            "Mean-Field Game allocation policy "
            "with the no-priority ablation."
        ),
        "",
        (
            "The policy evaluates each feasible "
            "edge node using its Mean-Field state, "
            "control and aggregate mean-field state."
        ),
        "",
        (
            "The node with the minimum Mean-Field "
            "running cost is selected."
        ),
        "",
        "## Configuration",
        "",
        f"- Simulation steps: {config.simulation_steps}",
        f"- Tasks per step: {config.tasks_per_step}",
        f"- Servers: {config.number_of_servers}",
        f"- Nodes per server: {config.nodes_per_server}",
        f"- Seeds: {len(seeds)}",
        "- Ablation variant: no_priority",
        "",
        "## Mean-Field Formulation",
        "",
        (
            "The no-priority formulation removes "
            "priority-dependent reward from the "
            "decision objective."
        ),
        "",
        (
            "Priority-specific service preferences "
            "are replaced by population-weighted "
            "neutral parameters."
        ),
        "",
        "## Resource Filtering Audit",
        "",
        (
            "Every task-node pair is checked for CPU (GC/s), memory (GB), "
            "bandwidth (Mbps), latency (ms), energy (J), and queue length "
            "(tasks) before policy selection."
        ),
        "",
        "The generated audit figure is `figures/resource_filtering_selection_audit.png`.",
        "The detailed audit is stored in `raw/resource_filtering_audit.csv`.",
        "The filtering summary is stored in `aggregated/resource_filtering_summary.csv`.",
        "",
        "The feasible edge set is a capability filter; the Mean-Field policy performs the final selection only from that set.",
        "",
        "## Metrics",
        "",
        "- Mean utility",
        "- Mean response time",
        "- Throughput",
        "- Success ratio",
        "- Rejected tasks",
        "- Resource utilization",
        "- Load variance",
        "- Jain's fairness index",
        "- Average queue length",
        "",
        "## Aggregated Results",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['metric']} | "
            f"{row['mean']:.6f} | "
            f"{row['std']:.6f} | "
            f"{row['min']:.6f} | "
            f"{row['max']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Filtering Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )

    for _, row in filtering_summary.iterrows():
        lines.append(
            "| "
            f"{row['metric']} | "
            f"{row['value']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Module-2 evaluates Mean-Field Game "
                "based load balancing without using "
                "priority as a decision preference."
            ),
            "",
            (
                "The results provide a second "
                "reference point against the "
                "Module-1 least-loaded baseline."
            ),
            "",
            (
                "The primary comparison focuses on "
                "response time, throughput, resource "
                "utilization, load variance and "
                "Jain's fairness index."
            ),
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )