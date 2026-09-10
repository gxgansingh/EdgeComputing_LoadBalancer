"""Load-balancer benchmarking with single-run and repeated-run evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import gc

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import BaselinePolicy
from ..config import SimulationConfig
from .runner import build_mean_field_policy
from .module2 import _generate_resource_filtering_figure, _summarize_filtering_audit


METRICS = (
    "utility_mean",
    "response_time_mean",
    "throughput",
    "success_ratio",
    "resource_utilization",
    "load_variance",
    "jains_fairness_index",
    "average_queue_length",
    "rejected_tasks",
)


def _run_policy(config: SimulationConfig, seed: int, policy_name: str, policy):
    """Run one policy and return its result."""
    seed_config = replace(config, seed=seed)
    return run_policy_experiment(
        config=seed_config,
        policy_name=policy_name,
        policy=policy,
    )


def _run_mfg(config: SimulationConfig, seed: int):
    """Build an equilibrium and run the MFG load balancer."""
    seed_config = replace(config, seed=seed)
    policy, diagnostics = build_mean_field_policy(
        config=seed_config,
        ablation_variant="full",
    )
    result = run_policy_experiment(
        config=seed_config,
        policy_name="mfg_load_balancer",
        policy=policy,
    )
    return result, diagnostics


def _run_baseline(config: SimulationConfig, seed: int):
    """Run the least-loaded feasible-node baseline."""
    seed_config = replace(config, seed=seed)
    policy = BaselinePolicy(config=seed_config)
    result = run_policy_experiment(
        config=seed_config,
        policy_name="least_loaded_baseline",
        policy=policy,
    )
    return result


def _confidence_interval(values: pd.Series) -> float:
    """Return the half-width of a 95 percent Student-t confidence interval."""
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 2:
        return 0.0
    standard_error = np.std(clean, ddof=1) / np.sqrt(n)
    return float(t.ppf(0.975, n - 1) * standard_error)


def _aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated benchmark observations."""
    rows = []
    for (policy, metric), group in raw.groupby(["policy", "metric"], sort=True):
        values = group["value"].astype(float)
        rows.append(
            {
                "policy": policy,
                "metric": metric,
                "runs": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "ci95_half_width": _confidence_interval(values),
            }
        )
    return pd.DataFrame(rows)


def _plot_single_run(records: pd.DataFrame, path: Path) -> None:
    """Plot all edge-node CPU utilization for one MFG run."""
    plt.figure(figsize=(18, 10))
    for node_id, group in records.groupby("node_id", sort=True):
        plt.plot(
            group["simulation_step"],
            group["cpu_utilization"],
            linewidth=1.2,
            marker=".",
            markersize=2,
            label=f"Edge Node {node_id + 1}",
        )
    plt.title("MFG Load Balancer: Single-Run Edge CPU Utilization")
    plt.xlabel("Simulation Tick")
    plt.ylabel("CPU Utilization (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_ten_run_mean(records: pd.DataFrame, path: Path) -> None:
    """Plot mean CPU utilization by edge node across ten runs."""
    grouped = (
        records.groupby(["node_id", "simulation_step"], as_index=False)["cpu_utilization"]
        .mean()
    )
    plt.figure(figsize=(18, 10))
    for node_id, group in grouped.groupby("node_id", sort=True):
        plt.plot(
            group["simulation_step"],
            group["cpu_utilization"],
            linewidth=1.2,
            label=f"Edge Node {node_id + 1}",
        )
    plt.title("MFG Load Balancer: Mean Edge CPU Utilization Across 10 Runs")
    plt.xlabel("Simulation Tick")
    plt.ylabel("Mean CPU Utilization (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_system_metric(aggregate: pd.DataFrame, metric: str, path: Path) -> None:
    """Plot a single benchmark metric for MFG and baseline."""
    data = aggregate[aggregate["metric"] == metric]
    plt.figure(figsize=(10, 6))
    for policy, group in data.groupby("policy", sort=True):
        plt.errorbar(
            [policy.replace("_", " ").title()],
            group["mean"],
            yerr=group["ci95_half_width"],
            fmt="o",
            capsize=5,
            label=policy.replace("_", " ").title(),
        )
    plt.title(f"Benchmark: {metric.replace('_', ' ').title()}")
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _write_report(
    path: Path,
    config: SimulationConfig,
    seeds: tuple[int, ...],
    aggregate: pd.DataFrame,
    filtering_summary: pd.DataFrame,
) -> None:
    """Write the benchmark methodology and artifact index."""
    lines = [
        "# Load Balancer Benchmark Report",
        "",
        "## Benchmark design",
        "",
        f"- Simulation ticks per run: {config.benchmark_simulation_steps}",
        f"- Number of repeated runs: {len(seeds)}",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Edge servers: {config.number_of_servers}",
        f"- Edge nodes per server: {config.nodes_per_server}",
        f"- Total edge nodes: {config.number_of_servers * config.nodes_per_server}",
        f"- Tasks per tick: {config.tasks_per_step}",
        "- Proposed policy: Mean-Field Game load balancer",
        "",
        "## Resource metric units",
        "",
        "| Resource | Unit | Meaning |",
        "|---|---|---|",
        "| CPU | GC/s | Gigacycles per Second |",
        "| Memory | GB | Gigabytes |",
        "| Bandwidth | Mbps | Megabits per Second |",
        "| Latency | ms | Milliseconds |",
        "| Energy | J | Joules |",
        "| Queue | tasks | Number of Tasks |",
        "",
        "The MFG state and utilization metrics are normalized/dimensionless where applicable; physical resource values retain the units above.",
        "",
        "## Evaluation metrics",
        "",
        ", ".join(METRICS),
        "",
        "## Interpretation",
        "",
        "The single-run utilization figure shows the dynamic CPU load of every edge node over the full 2500-tick simulation.",
        "The ten-run figure reports the mean utilization at every simulation tick across the ten independent seeds.",
        "The benchmark tables report repeated-run mean, standard deviation, range, and 95 percent Student-t confidence intervals for the MFG load balancer.",
        "Resource filtering is evaluated before policy selection and is reported separately from the MFG selection decision.",
        "",
        "## Generated artifacts",
        "",
        "- raw/single_run_metrics.csv",
        "- raw/single_run_node_utilization.csv",
        "- raw/ten_run_benchmark_raw.csv",
        "- raw/ten_run_node_utilization_raw.csv",
        "- raw/ten_run_node_utilization_time_series_raw.csv",
        "- raw/resource_filtering_audit.csv",
        "- aggregated/single_run_node_summary.csv",
        "- aggregated/ten_run_benchmark_summary.csv",
        "- aggregated/ten_run_node_utilization_summary.csv",
        "- aggregated/ten_run_node_utilization_time_series_summary.csv",
        "- aggregated/resource_filtering_summary.csv",
        "- figures/single_run_node_utilization.png",
        "- figures/ten_run_node_utilization.png",
        "- figures/ten_run_node_utilization_time_series.png",
        "- figures/resource_filtering_selection_audit.png",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_load_balancer_benchmark(
    config: SimulationConfig,
    seeds: tuple[int, ...] | list[int] | None = None,
    output_directory: str | Path | None = None,
) -> dict[str, Path]:
    """Run the complete single-run and repeated load-balancer benchmark.

    ``seeds`` and ``output_directory`` are optional overrides used by tests and
    programmatic callers. When omitted, the benchmark uses the configured
    seed range and writes under ``config.output_directory``.

    The return value is a mapping of artifact names to their generated paths so
    callers can consume individual benchmark outputs without reconstructing
    the directory layout.
    """
    output = (
        Path(output_directory)
        if output_directory is not None
        else Path(config.output_directory) / "load_balancer_benchmark"
    )
    raw_dir = output / "raw"
    agg_dir = output / "aggregated"
    fig_dir = output / "figures"
    for directory in (raw_dir, agg_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = tuple(
            range(
                config.benchmark_seed_start,
                config.benchmark_seed_start + config.benchmark_experiment_repetitions,
            )
        )
    else:
        seeds = tuple(int(seed) for seed in seeds)

    if not seeds:
        raise ValueError("At least one benchmark seed is required.")

    # The benchmark intentionally uses a 2500-tick configuration without changing
    # the shorter simulation length used by the other research modules.
    benchmark_config = replace(
        config,
        simulation_steps=config.benchmark_simulation_steps,
        number_of_servers=config.benchmark_number_of_servers,
        nodes_per_server=config.benchmark_nodes_per_server,
        mean_field_state_points=config.benchmark_mean_field_state_points,
        mean_field_max_iterations=config.benchmark_mean_field_max_iterations,
        fpk_max_iterations=config.benchmark_fpk_max_iterations,
    )

    metric_rows = []
    node_rows = []
    filtering_rows = []
    single_result = None
    single_diagnostics = None

    # The equilibrium is deterministic for a fixed configuration and does not
    # depend on the random workload seed. Solve it once and reuse the policy
    # across all repeated workload runs.
    equilibrium_policy, equilibrium_diagnostics = build_mean_field_policy(
        config=benchmark_config,
        ablation_variant="full",
    )

    for run_index, seed in enumerate(seeds):
        seed_config = replace(benchmark_config, seed=seed)
        mfg_result = run_policy_experiment(
            config=seed_config,
            policy_name="mfg_load_balancer",
            policy=equilibrium_policy,
        )
        diagnostics = equilibrium_diagnostics

        if run_index == 0:
            single_result = mfg_result
            single_diagnostics = diagnostics

        for metric in METRICS:
            metric_rows.append(
                {
                    "run": run_index + 1,
                    "seed": seed,
                    "policy": "mfg_load_balancer",
                    "metric": metric,
                    "value": float(mfg_result.metrics[metric]),
                }
            )

        for record in mfg_result.node_utilization_records:
            node_rows.append(
                {
                    "run": run_index + 1,
                    "seed": seed,
                    "policy": "mfg_load_balancer",
                    **record,
                }
            )

        # The detailed resource-filtering audit is retained for the single run
        # used in the professor-facing screenshot. Repeating every node check
        # across all ten runs would create an unnecessarily large evidence file.
        if run_index == 0:
            for record in mfg_result.filtering_records:
                filtering_rows.append(
                    {
                        "run": 1,
                        "seed": seed,
                        "policy": "mfg_load_balancer",
                        **record,
                    }
                )

        # Release the large per-task audit structures before starting the next run.
        if run_index > 0:
            mfg_result.filtering_records.clear()
        mfg_result.node_utilization_records.clear()
        gc.collect()

    raw_metrics = pd.DataFrame(metric_rows)
    raw_metrics.to_csv(raw_dir / "ten_run_benchmark_raw.csv", index=False)
    raw_metrics[raw_metrics["run"] == 1].to_csv(
        raw_dir / "single_run_metrics.csv", index=False
    )

    aggregate = _aggregate(raw_metrics)
    aggregate.to_csv(agg_dir / "ten_run_benchmark_summary.csv", index=False)
    aggregate[aggregate["policy"] == "mfg_load_balancer"].to_csv(
        agg_dir / "single_run_node_summary.csv", index=False
    )

    node_data = pd.DataFrame(node_rows)
    node_data.to_csv(raw_dir / "ten_run_node_utilization_raw.csv", index=False)
    single_nodes = node_data[node_data["run"] == 1].copy()
    single_nodes.to_csv(raw_dir / "single_run_node_utilization.csv", index=False)

    node_summary = (
        node_data.groupby("node_id", as_index=False)["cpu_utilization"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"node_id": "node_id"})
    )
    node_summary.to_csv(agg_dir / "ten_run_node_utilization_summary.csv", index=False)

    time_summary = (
        node_data.groupby(["node_id", "simulation_step"], as_index=False)["cpu_utilization"]
        .agg(["mean", "std"])
        .reset_index()
    )
    time_summary.to_csv(
        agg_dir / "ten_run_node_utilization_time_series_summary.csv",
        index=False,
    )
    node_data.to_csv(
        raw_dir / "ten_run_node_utilization_time_series_raw.csv",
        index=False,
    )

    _plot_single_run(single_nodes, fig_dir / "single_run_node_utilization.png")
    _plot_ten_run_mean(node_data, fig_dir / "ten_run_node_utilization.png")
    _plot_ten_run_mean(node_data, fig_dir / "ten_run_node_utilization_time_series.png")

    for metric in METRICS:
        _plot_system_metric(
            aggregate,
            metric,
            fig_dir / f"ten_run_{metric}.png",
        )

    filtering_data = pd.DataFrame(filtering_rows)
    filtering_data.to_csv(raw_dir / "resource_filtering_audit.csv", index=False)
    filtering_summary = _summarize_filtering_audit(filtering_data)
    filtering_summary.to_csv(
        agg_dir / "resource_filtering_summary.csv",
        index=False,
    )

    # Reuse the project's audit renderer, but force a representative task from
    # the full 2500-tick benchmark and the actual MFG selection records.
    selection_rows = []
    for record in single_result.selection_records:
        selection_rows.append(
            {
                "seed": config.benchmark_seed_start,
                "policy": "mfg_load_balancer",
                **record,
            }
        )
    selection_data = pd.DataFrame(selection_rows)
    _generate_resource_filtering_figure(
        filtering_audit=filtering_data[filtering_data["run"] == 1].copy(),
        selection_audit=selection_data,
        figures_directory=fig_dir,
    )

    pd.DataFrame(
        [
            {
                "seed": config.benchmark_seed_start,
                "equilibrium_converged": single_diagnostics["converged"],
                "equilibrium_iterations": single_diagnostics["iterations"],
                "distribution_residual": single_diagnostics["distribution_residual"],
                "policy_residual": single_diagnostics["policy_residual"],
            }
        ]
    ).to_csv(agg_dir / "single_run_equilibrium_diagnostics.csv", index=False)

    _write_report(
        output / "load_balancer_benchmark_report.md",
        benchmark_config,
        seeds,
        aggregate,
        filtering_summary,
    )

    return {
        "output_directory": output,
        "single_run_metrics": raw_dir / "single_run_metrics.csv",
        "single_run_node_utilization": raw_dir / "single_run_node_utilization.csv",
        "ten_run_benchmark_raw": raw_dir / "ten_run_benchmark_raw.csv",
        "ten_run_node_utilization_raw": raw_dir / "ten_run_node_utilization_raw.csv",
        "ten_run_node_utilization_time_series_raw": raw_dir / "ten_run_node_utilization_time_series_raw.csv",
        "resource_filtering_audit": raw_dir / "resource_filtering_audit.csv",
        "single_run_node_summary": agg_dir / "single_run_node_summary.csv",
        "ten_run_benchmark_summary": agg_dir / "ten_run_benchmark_summary.csv",
        "ten_run_node_utilization_summary": agg_dir / "ten_run_node_utilization_summary.csv",
        "ten_run_node_utilization_time_series_summary": agg_dir / "ten_run_node_utilization_time_series_summary.csv",
        "resource_filtering_summary": agg_dir / "resource_filtering_summary.csv",
        "single_run_graph": fig_dir / "single_run_node_utilization.png",
        "ten_run_graph": fig_dir / "ten_run_node_utilization.png",
        "ten_run_time_series_graph": fig_dir / "ten_run_node_utilization_time_series.png",
        "resource_filtering_screenshot": fig_dir / "resource_filtering_selection_audit.png",
        "benchmark_report": output / "load_balancer_benchmark_report.md",
    }
