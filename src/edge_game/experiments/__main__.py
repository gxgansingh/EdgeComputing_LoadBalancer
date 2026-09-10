"""Command-line entry points for experiment modules."""

from __future__ import annotations

import sys
from pathlib import Path

from ..config import SimulationConfig
from .ablation import run_ablation_experiment
from .module1 import run_module1_experiment
from .module2 import run_module2_experiment
from .module3 import run_module3_experiment
from .performance_matrix import run_performance_matrix
from .research_pipeline import run_full_research_pipeline
from .benchmark import run_load_balancer_benchmark
from .module2_comparison import (
    generate_module2_comparison,
)
from .robustness import run_robustness_experiment
from .robustness_analysis import (
    generate_robustness_analysis_outputs,
)
from .robustness_diagnostic import (
    generate_diagnostic_outputs,
)
from .robustness_report import (
    generate_robustness_report,
)
from .visualization import generate_robustness_figures


VALID_COMMANDS = {
    "benchmark",
    "module1",
    "module2",
    "module2-comparison",
    "module3",
    "performance-matrix",
    "full-evaluation",
    "ablation",
    "robustness",
    "robustness-analysis",
    "robustness-diagnostic",
    "robustness-report",
}


def main() -> None:
    """Run the requested experiment command."""
    if (
        len(sys.argv) != 2
        or sys.argv[1] not in VALID_COMMANDS
    ):
        raise SystemExit(
            "Usage: python -m src.edge_game.experiments "
            "{benchmark|module1|module2|module3|module2-comparison|"
            "ablation|performance-matrix|full-evaluation|robustness|"
            "robustness-analysis|robustness-diagnostic|"
            "robustness-report}"
        )

    config = SimulationConfig()

    command = sys.argv[1]

    if command == "benchmark":
        outputs = run_load_balancer_benchmark(config)
        print("Load-balancer benchmark completed.")
        print(f"Simulation ticks per run: {config.benchmark_simulation_steps}")
        print(f"Repeated runs: {config.benchmark_experiment_repetitions}")
        print(f"Results: {outputs['output_directory']}")
        print(f"Single-run graph: {outputs['single_run_graph']}")
        print(f"10-run graph: {outputs['ten_run_time_series_graph']}")
        print(f"Benchmark table: {outputs['ten_run_benchmark_summary']}")
        print(f"Filtering audit: {outputs['resource_filtering_audit']}")
        print(f"Professor screenshot: {outputs['resource_filtering_screenshot']}")
        return

    if command == "full-evaluation":
        result = run_full_research_pipeline(config)
        print("Full research evaluation completed.")
        print(f"Results: {result.output_directory}")
        print(f"Manifest: {result.manifest}")
        print(f"Final report: {result.master_report}")
        return

    seeds = tuple(
        range(
            config.experiment_seed_start,
            (
                config.experiment_seed_start
                + config.experiment_repetitions
            ),
        )
    )

    if command == "module1":
        output_directory = (
            Path(config.output_directory)
            / "module1"
        )

        result = run_module1_experiment(
            config=config,
            seeds=seeds,
            output_directory=output_directory,
        )

        print(
            "Module-1 experiment completed."
        )

        print(
            "Architecture: "
            "IoT -> Edge Gateway -> Edge Server -> Edge Nodes"
        )

        print(
            "Policy: no-priority least-loaded baseline"
        )

        print(
            f"Seeds: {len(seeds)}"
        )

        print(
            f"Results saved to: "
            f"{output_directory}"
        )

        print(
            "\nGenerated outputs:"
        )

        print(
            f"Raw results: "
            f"{output_directory / 'raw' / 'module1_raw.csv'}"
        )

        print(
            f"Summary: "
            f"{output_directory / 'aggregated' / 'module1_summary.csv'}"
        )

        print(
            f"Report: "
            f"{output_directory / 'module1_report.md'}"
        )

        print(
            f"Figures: "
            f"{output_directory / 'figures'}"
        )

        print(
            f"Resource filtering audit: "
            f"{output_directory / 'raw' / 'resource_filtering_audit.csv'}"
        )

        print(
            f"Filtering summary: "
            f"{output_directory / 'aggregated' / 'resource_filtering_summary.csv'}"
        )

        print(
            f"Professor screenshot: "
            f"{output_directory / 'figures' / 'resource_filtering_selection_audit.png'}"
        )

        print(
            "\nMean metric results:"
        )

        numeric_columns = result.select_dtypes(
            include="number"
        ).columns

        for metric in numeric_columns:
            if metric == "seed":
                continue

            print(
                f"{metric}: "
                f"{result[metric].mean():.6f}"
            )

        return

    if command == "module2-comparison":
        output_directory = (
            Path(config.output_directory)
            / "module2"
        )

        comparison = (
            generate_module2_comparison(
                output_directory=output_directory,
            )
        )

        print(
            "\nModule-2 comparison completed."
        )

        print(
            "Comparison: "
            f"{output_directory / 'aggregated' / 'module1_vs_module2.csv'}"
        )

        print(
            "Figure: "
            f"{output_directory / 'figures' / 'module1_vs_module2.png'}"
        )

        return

    if command == "module2":
        output_directory = (
            Path(config.output_directory)
            / "module2"
        )

        result = run_module2_experiment(
            config=config,
            seeds=seeds,
            output_directory=output_directory,
        )

        print(
            "Module-2 experiment completed."
        )

        print(
            "Architecture: "
            "IoT -> Edge Gateway -> "
            "Edge Server -> Edge Nodes"
        )

        print(
            "Policy: no-priority "
            "Mean-Field Game"
        )

        print(
            f"Seeds: {len(seeds)}"
        )

        print(
            f"Results saved to: "
            f"{output_directory}"
        )

        print(
            "\nGenerated outputs:"
        )

        print(
            f"Raw results: "
            f"{output_directory / 'raw' / 'module2_raw.csv'}"
        )

        print(
            f"Summary: "
            f"{output_directory / 'aggregated' / 'module2_summary.csv'}"
        )

        print(
            f"Report: "
            f"{output_directory / 'module2_report.md'}"
        )

        print(
            f"Figures: "
            f"{output_directory / 'figures'}"
        )

        print(
            f"Resource filtering audit: "
            f"{output_directory / 'raw' / 'resource_filtering_audit.csv'}"
        )

        print(
            f"Filtering summary: "
            f"{output_directory / 'aggregated' / 'resource_filtering_summary.csv'}"
        )

        print(
            f"Professor screenshot: "
            f"{output_directory / 'figures' / 'resource_filtering_selection_audit.png'}"
        )

        print(
            "\nMean metric results:"
        )

        numeric_columns = (
            result.select_dtypes(
                include="number"
            ).columns
        )

        for metric in numeric_columns:
            if metric == "seed":
                continue

            print(
                f"{metric}: "
                f"{result[metric].mean():.6f}"
            )

        return

    if command == "ablation":
        output_directory = (
            Path(config.output_directory)
            / "ablation"
        )

        result = run_ablation_experiment(
            config=config,
            seeds=seeds,
            output_directory=output_directory,
        )

        print(
            "Ablation experiment completed."
        )

        print(
            f"Variants: "
            f"{result.raw_results['variant'].nunique()}"
        )

        print(
            f"Scenarios: "
            f"{result.raw_results['scenario'].nunique()}"
        )

        print(
            f"Seeds per variant/scenario: "
            f"{len(seeds)}"
        )

        print(
            f"Results saved to: "
            f"{output_directory}"
        )

        print(
            "\nGenerated outputs:"
        )

        print(
            f"Raw results: "
            f"{output_directory / 'raw' / 'ablation_raw.csv'}"
        )

        print(
            f"Comparison: "
            f"{output_directory / 'aggregated' / 'ablation_comparison.csv'}"
        )

        print(
            f"Summary: "
            f"{output_directory / 'aggregated' / 'ablation_summary.csv'}"
        )

        print(
            f"Figure: "
            f"{output_directory / 'figures' / 'ablation_effect_heatmap.png'}"
        )

        print(
            f"Report: "
            f"{output_directory / 'ablation_report.md'}"
        )

        return

    if command == "module3":
        output_directory = (
            Path(config.output_directory)
            / "module3"
        )

        result = run_module3_experiment(
            config=config,
            seeds=seeds,
            output_directory=output_directory,
        )

        print(
            "Module-3 experiment completed."
        )

        print(
            "Architecture: "
            "IoT -> Edge Gateway -> "
            "Edge Servers -> Edge Nodes"
        )

        print(
            "Policy: hierarchical "
            "least-loaded server + "
            "no-priority Mean-Field node selection"
        )

        print(
            f"Seeds: {len(seeds)}"
        )

        print(
            f"Results saved to: "
            f"{output_directory}"
        )

        print(
            "\nGenerated outputs:"
        )

        print(
            f"Raw results: "
            f"{output_directory / 'raw' / 'module3_raw.csv'}"
        )

        print(
            f"Summary: "
            f"{output_directory / 'aggregated' / 'module3_summary.csv'}"
        )

        print(
            f"Report: "
            f"{output_directory / 'module3_report.md'}"
        )

        print(
            f"Figures: "
            f"{output_directory / 'figures'}"
        )

        print(
            f"Resource filtering audit: "
            f"{output_directory / 'raw' / 'resource_filtering_audit.csv'}"
        )

        print(
            f"Filtering summary: "
            f"{output_directory / 'aggregated' / 'resource_filtering_summary.csv'}"
        )

        print(
            f"Professor screenshot: "
            f"{output_directory / 'figures' / 'resource_filtering_selection_audit.png'}"
        )

        print(
            "\nMean metric results:"
        )

        numeric_columns = result.select_dtypes(
            include="number"
        ).columns

        for metric in numeric_columns:
            if metric == "seed":
                continue

            print(
                f"{metric}: "
                f"{result[metric].mean():.6f}"
            )

        return

    if command == "performance-matrix":
        output_directory = (
            Path(config.output_directory)
            / "performance_matrix"
        )

        result = run_performance_matrix(
            config=config,
            seeds=seeds,
            output_directory=output_directory,
        )

        print("Performance-matrix experiment completed.")
        print(f"Network-load levels: {len(config.network_load_levels)}")
        print(f"Seeds per load level: {len(seeds)}")
        print(f"Results saved to: {output_directory}")
        print(f"Matrix: {output_directory / 'aggregated' / 'performance_matrix.csv'}")
        print(f"Report: {output_directory / 'performance_matrix_report.md'}")
        print(f"Figures: {output_directory / 'figures'}")
        return

    output_directory = (
        Path(config.output_directory)
        / "robustness"
    )

    if command == "robustness-analysis":
        outputs = (
            generate_robustness_analysis_outputs(
                results_directory=output_directory,
            )
        )

        print(
            "Robustness analysis completed."
        )

        for path in outputs:
            print(
                f"Output saved to: {path}"
            )

        return

    if command == "robustness-diagnostic":
        outputs = generate_diagnostic_outputs(
            output_directory
        )

        print(
            "Robustness diagnostic analysis completed."
        )

        for path in outputs:
            print(
                f"Output saved to: {path}"
            )

        return

    if command == "robustness-report":
        report_path = (
            generate_robustness_report(
                output_directory
            )
        )

        print(
            "Robustness report generated."
        )

        print(
            f"Output saved to: {report_path}"
        )

        return

    result = run_robustness_experiment(
        config=config,
        seeds=seeds,
        output_directory=output_directory,
    )

    print(
        "Robustness experiment completed."
    )

    print(
        f"Scenarios: "
        f"{result.raw_results['scenario'].nunique()}"
    )

    print(
        f"Seeds per scenario: {len(seeds)}"
    )

    for scenario in (
        result.paired_comparison[
            "scenario"
        ].unique()
    ):
        print(
            f"\nScenario: {scenario}"
        )

        rows = result.paired_comparison.loc[
            result.paired_comparison[
                "scenario"
            ]
            == scenario
        ]

        for _, row in rows.iterrows():
            print(
                f"{row['metric']} | "
                f"delta="
                f"{row['mean_difference']:.6f} | "
                f"relative="
                f"{row['relative_change_percent']:.2f}% | "
                f"p="
                f"{row['p_value']:.6f}"
            )

    print(
        f"\nResults saved to: "
        f"{output_directory}"
    )

    print(
        "\nFigures generated:"
    )

    for figure in generate_robustness_figures(
        output_directory
    ):
        print(
            f"Figure saved to: {figure}"
        )


if __name__ == "__main__":
    main()