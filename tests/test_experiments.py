"""Tests for repeated policy experiments and statistical analysis."""

import numpy as np
import pandas as pd
import pytest

from src.edge_game.experiments.robustness_diagnostic import (
    build_scenario_summary,
)

from src.edge_game.config import SimulationConfig
from src.edge_game.experiments.statistics import (
    aggregate_policy_metrics,
    paired_policy_comparison,
    paired_selection_analysis,
    policy_selection_summary,
)


def _sample_results() -> pd.DataFrame:
    """Create a small deterministic paired experiment dataset."""
    return pd.DataFrame(
        [
            {
                "seed": 42,
                "policy_name": "least_loaded_baseline",
                "utility_mean": 4.0,
                "response_time_mean": 2.0,
                "throughput": 10.0,
                "success_ratio": 1.0,
                "rejected_tasks": 0.0,
                "resource_utilization": 0.2,
                "load_variance": 0.1,
                "jains_fairness_index": 0.8,
                "average_queue_length": 1.0,
                "priority_success_ratio": 0.9,
            },
            {
                "seed": 42,
                "policy_name": "priority_aware_mean_field",
                "utility_mean": 4.5,
                "response_time_mean": 1.8,
                "throughput": 11.0,
                "success_ratio": 1.0,
                "rejected_tasks": 0.0,
                "resource_utilization": 0.25,
                "load_variance": 0.08,
                "jains_fairness_index": 0.85,
                "average_queue_length": 0.8,
                "priority_success_ratio": 0.95,
            },
            {
                "seed": 43,
                "policy_name": "least_loaded_baseline",
                "utility_mean": 4.2,
                "response_time_mean": 2.2,
                "throughput": 9.0,
                "success_ratio": 0.9,
                "rejected_tasks": 1.0,
                "resource_utilization": 0.22,
                "load_variance": 0.12,
                "jains_fairness_index": 0.75,
                "average_queue_length": 1.2,
                "priority_success_ratio": 0.85,
            },
            {
                "seed": 43,
                "policy_name": "priority_aware_mean_field",
                "utility_mean": 4.6,
                "response_time_mean": 1.9,
                "throughput": 10.0,
                "success_ratio": 0.95,
                "rejected_tasks": 0.0,
                "resource_utilization": 0.24,
                "load_variance": 0.09,
                "jains_fairness_index": 0.82,
                "average_queue_length": 0.9,
                "priority_success_ratio": 0.92,
            },
        ]
    )


def test_repeated_experiment_statistics_are_computed() -> None:
    """Verify repeated policy statistics and paired deltas."""
    frame = _sample_results()

    summary = aggregate_policy_metrics(
        frame
    )

    paired = paired_policy_comparison(
        frame
    )

    assert len(summary) == 20
    assert len(paired) == 10

    utility_row = paired.loc[
        paired["metric"]
        == "utility_mean"
    ].iloc[0]

    assert np.isclose(
        utility_row[
            "mean_difference"
        ],
        0.45,
    )


def test_experiment_configuration_is_valid() -> None:
    """Verify repeated experiment configuration."""
    config = SimulationConfig()

    assert (
        config.experiment_repetitions
        > 0
    )

    assert (
        config.experiment_seed_start
        >= 0
    )

def _sample_selection_records() -> pd.DataFrame:
    """Create deterministic paired node-selection records."""
    return pd.DataFrame(
        [
            {
                "seed": 42,
                "policy_name": "least_loaded_baseline",
                "task_id": 0,
                "priority": 1,
                "node_id": 0,
                "cpu_capacity": 10.0,
                "load_ratio": 0.10,
                "queue_length": 1,
            },
            {
                "seed": 42,
                "policy_name": "priority_aware_mean_field",
                "task_id": 0,
                "priority": 1,
                "node_id": 0,
                "cpu_capacity": 10.0,
                "load_ratio": 0.10,
                "queue_length": 1,
            },
            {
                "seed": 42,
                "policy_name": "least_loaded_baseline",
                "task_id": 1,
                "priority": 3,
                "node_id": 1,
                "cpu_capacity": 15.0,
                "load_ratio": 0.20,
                "queue_length": 2,
            },
            {
                "seed": 42,
                "policy_name": "priority_aware_mean_field",
                "task_id": 1,
                "priority": 3,
                "node_id": 2,
                "cpu_capacity": 25.0,
                "load_ratio": 0.15,
                "queue_length": 1,
            },
        ]
    )


def test_policy_selection_summary_is_computed() -> None:
    """Verify selected-node characteristics are summarized by policy."""
    records = _sample_selection_records()

    summary = policy_selection_summary(records)

    assert len(summary) == 6

    baseline_cpu = summary.loc[
        (summary["policy_name"] == "least_loaded_baseline")
        & (summary["metric"] == "selected_cpu_capacity")
    ].iloc[0]

    assert np.isclose(
        baseline_cpu["mean"],
        12.5,
    )


def test_paired_selection_analysis_detects_divergence() -> None:
    """Verify paired analysis detects different node selections."""
    records = _sample_selection_records()

    comparison, frequency = paired_selection_analysis(
        records
    )

    divergence = comparison.loc[
        comparison["metric"] == "selection_divergence_rate"
    ].iloc[0]

    priority_three = comparison.loc[
        comparison["metric"]
        == "priority_3_selection_divergence_rate"
    ].iloc[0]

    assert np.isclose(
        divergence["comparison_mean"],
        0.5,
    )

    assert np.isclose(
        priority_three["comparison_mean"],
        1.0,
    )

    assert len(frequency) == 4


def test_wilson_interval_is_used_for_selection_divergence() -> None:
    """Verify divergence is reported as a proportion interval."""
    records = _sample_selection_records()

    comparison, _ = paired_selection_analysis(records)

    divergence = comparison.loc[
        comparison["metric"] == "selection_divergence_rate"
    ].iloc[0]

    assert divergence["statistical_test"] == (
        "wilson_proportion_interval"
    )
    assert np.isclose(
        divergence["comparison_mean"],
        0.5,
    )
    assert divergence["ci95_low"] < 0.5
    assert divergence["ci95_high"] > 0.5


def test_workload_scenarios_have_distinct_task_rates() -> None:
    """Verify workload scenarios represent increasing congestion."""
    from src.edge_game.experiments.runner import (
        WORKLOAD_SCENARIO_TASKS,
    )

    assert WORKLOAD_SCENARIO_TASKS["default"] < (
        WORKLOAD_SCENARIO_TASKS["moderate_congestion"]
    )
    assert WORKLOAD_SCENARIO_TASKS["moderate_congestion"] < (
        WORKLOAD_SCENARIO_TASKS["high_congestion"]
    )


def test_invalid_workload_scenario_is_rejected() -> None:
    """Verify unknown workload scenarios fail explicitly."""
    from src.edge_game.experiments.runner import (
        _resolve_workload_scenario,
    )

    config = SimulationConfig()

    try:
        _resolve_workload_scenario(
            config,
            "invalid_scenario",
        )
    except ValueError as exc:
        assert "Unknown workload scenario" in str(exc)
    else:
        raise AssertionError(
            "Invalid workload scenario was accepted."
        )


def _sample_decision_records() -> pd.DataFrame:
    """Create deterministic candidate-set decision records."""
    return pd.DataFrame(
        [
            {
                "seed": 42,
                "task_id": 0,
                "priority": 3,
                "policy_name": "least_loaded_baseline",
                "candidate_count": 3,
                "candidate_node_ids": "0,1,2",
                "candidate_scores": "0:0.1,1:0.2,2:0.3",
                "selected_node_id": 0,
                "selected_rank": 1,
                "selected_score": 0.1,
                "best_score": 0.1,
                "worst_score": 0.3,
                "score_margin": 0.0,
                "score_tie_count": 1,
                "state": 0.1,
                "control": np.nan,
                "mean_field_score": np.nan,
            },
            {
                "seed": 42,
                "task_id": 0,
                "priority": 3,
                "policy_name": "priority_aware_mean_field",
                "candidate_count": 3,
                "candidate_node_ids": "0,1,2",
                "candidate_scores": "0:0.8,1:0.2,2:0.4",
                "selected_node_id": 1,
                "selected_rank": 1,
                "selected_score": 0.2,
                "best_score": 0.2,
                "worst_score": 0.8,
                "score_margin": 0.0,
                "score_tie_count": 1,
                "state": 0.2,
                "control": 0.7,
                "mean_field_score": 0.2,
            },
        ]
    )


def test_decision_audit_detects_same_candidate_set_and_different_choice() -> None:
    """Verify candidate-set identity is separated from node-choice identity."""
    from src.edge_game.experiments.statistics import (
        decision_audit_summary,
        paired_decision_audit,
    )

    records = _sample_decision_records()

    summary = decision_audit_summary(records)
    comparison = paired_decision_audit(records)

    assert not summary.empty

    candidate_identity = comparison.loc[
        comparison["metric"] == "candidate_set_identity_rate"
    ].iloc[0]

    selection_identity = comparison.loc[
        comparison["metric"] == "selected_node_identity_rate"
    ].iloc[0]

    assert np.isclose(
        candidate_identity["comparison_mean"],
        1.0,
    )

    assert np.isclose(
        selection_identity["comparison_mean"],
        0.0,
    )


def test_decision_records_include_candidate_scores() -> None:
    """Verify candidate score traces are retained for diagnosis."""
    records = _sample_decision_records()

    assert "candidate_scores" in records.columns
    assert "control" in records.columns
    assert records.iloc[1]["control"] == 0.7


def test_outcome_utility_is_recorded_on_completion() -> None:
    """Verify completed tasks contribute realized utility to the experiment."""
    from src.edge_game.algorithms.experiment import run_policy_experiment
    from src.edge_game.algorithms.policy import BaselinePolicy

    config = SimulationConfig(
        simulation_steps=5,
        tasks_per_step=1,
        number_of_servers=1,
        nodes_per_server=2,
    )

    result = run_policy_experiment(
        config=config,
        policy_name="least_loaded_baseline",
        policy=BaselinePolicy(config=config),
    )

    assert result.metrics["utility_mean"] != 4.474159
    assert np.isfinite(result.metrics["utility_mean"])


def test_utility_weight_profiles_are_distinct() -> None:
    """Verify utility sensitivity profiles have distinct objectives."""
    from src.edge_game.experiments.utility_sensitivity import (
        UTILITY_WEIGHT_PROFILES,
    )

    names = [profile.name for profile in UTILITY_WEIGHT_PROFILES]
    assert len(names) == len(set(names))
    assert len(UTILITY_WEIGHT_PROFILES) == 4
    assert UTILITY_WEIGHT_PROFILES[0].latency_cost == 1.0
    assert UTILITY_WEIGHT_PROFILES[-1].energy_cost == 1.5


def test_utility_profile_config_changes_only_utility_weights() -> None:
    """Verify profile application does not alter simulation workload settings."""
    from src.edge_game.experiments.utility_sensitivity import (
        UTILITY_WEIGHT_PROFILES,
        _profile_config,
    )

    config = SimulationConfig()
    profile = UTILITY_WEIGHT_PROFILES[-1]
    updated = _profile_config(config, profile)

    assert updated.tasks_per_step == config.tasks_per_step
    assert updated.simulation_steps == config.simulation_steps
    assert updated.number_of_servers == config.number_of_servers
    assert updated.utility_latency_cost_weight == profile.latency_cost
    assert updated.utility_queue_cost_weight == profile.queue_cost
    assert updated.utility_energy_cost_weight == profile.energy_cost


def test_utility_profile_changes_mean_field_equilibrium_policy() -> None:
    """Verify utility weights are coupled to the HJB-FPK optimization."""
    from dataclasses import replace

    from src.edge_game.experiments.runner import build_mean_field_policy

    config = SimulationConfig(
        mean_field_state_points=21,
        mean_field_max_iterations=5,
        fpk_max_iterations=10,
    )

    balanced, _ = build_mean_field_policy(config)
    latency_heavy, _ = build_mean_field_policy(
        replace(
            config,
            utility_latency_cost_weight=2.0,
            utility_queue_cost_weight=1.5,
        )
    )

    differences = [
        np.max(
            np.abs(
                balanced.equilibrium.policies[priority].control
                - latency_heavy.equilibrium.policies[priority].control
            )
        )
        for priority in (1, 2, 3)
    ]

    assert max(differences) > 1e-8


def test_utility_sensitivity_profiles_have_distinct_weights() -> None:
    """Verify utility sensitivity profiles remain explicitly distinct."""
    from src.edge_game.experiments.utility_sensitivity import (
        UTILITY_WEIGHT_PROFILES,
        _profile_config,
    )

    config = SimulationConfig()

    profile_weights = {
        (
            item.utility_priority_reward_weight,
            item.utility_latency_cost_weight,
            item.utility_resource_cost_weight,
            item.utility_queue_cost_weight,
            item.utility_energy_cost_weight,
        )
        for item in (
            _profile_config(config, profile)
            for profile in UTILITY_WEIGHT_PROFILES
        )
    }

    assert len(profile_weights) == 4


def test_utility_sensitivity_preserves_selection_pairing(tmp_path) -> None:
    """Verify utility sensitivity keeps policy and seed keys for selection pairing."""
    from src.edge_game.experiments.utility_sensitivity import (
        UTILITY_WEIGHT_PROFILES,
        run_utility_weight_sensitivity,
    )

    config = SimulationConfig(
        simulation_steps=3,
        tasks_per_step=2,
        number_of_servers=1,
        nodes_per_server=3,
        mean_field_state_points=11,
        mean_field_max_iterations=5,
        fpk_max_iterations=5,
    )

    result = run_utility_weight_sensitivity(
        config=config,
        seeds=[1],
        scenarios=["default"],
        profiles=(UTILITY_WEIGHT_PROFILES[0],),
        output_directory=tmp_path,
    )

    assert not result.selection_comparison.empty
    assert not result.decision_comparison.empty


def test_robustness_scenarios_have_increasing_stress() -> None:
    """Verify robustness scenarios cover distinct stress conditions."""
    from src.edge_game.experiments.robustness import ROBUSTNESS_SCENARIOS
    names = [scenario.name for scenario in ROBUSTNESS_SCENARIOS]
    assert names == ["nominal", "workload_stress", "cpu_scarcity", "memory_bandwidth_stress", "mixed_stress"]
    assert ROBUSTNESS_SCENARIOS[0].tasks_per_step < ROBUSTNESS_SCENARIOS[1].tasks_per_step
    assert ROBUSTNESS_SCENARIOS[-1].maximum_cpu_capacity < ROBUSTNESS_SCENARIOS[0].maximum_cpu_capacity


def test_robustness_scenario_config_applies_perturbations() -> None:
    """Verify scenario configuration changes the intended parameters."""
    from src.edge_game.experiments.robustness import ROBUSTNESS_SCENARIOS, _scenario_config
    config = SimulationConfig()
    stressed = _scenario_config(config, ROBUSTNESS_SCENARIOS[-1])
    assert stressed.tasks_per_step == 15
    assert stressed.maximum_cpu_capacity == 20.0
    assert stressed.maximum_memory_capacity == 20.0
    assert stressed.maximum_bandwidth == 60.0
    assert stressed.maximum_memory_demand == 12.0
    assert stressed.maximum_bandwidth_demand == 20.0

def test_benjamini_hochberg_adjustment():
    from src.edge_game.experiments.statistics import benjamini_hochberg

    p_values = [0.001, 0.01, 0.04, 0.20]

    adjusted, significant = benjamini_hochberg(
        p_values,
        alpha=0.05,
    )

    assert len(adjusted) == len(p_values)
    assert len(significant) == len(p_values)

    assert all(
        0.0 <= value <= 1.0
        for value in adjusted
    )

    assert adjusted[0] <= adjusted[1]
    assert adjusted[1] <= adjusted[2]
    assert adjusted[2] <= adjusted[3]

    assert significant == [True, True, False, False]


def test_benjamini_hochberg_preserves_input_order():
    from src.edge_game.experiments.statistics import benjamini_hochberg

    p_values = [0.20, 0.001, 0.04, 0.01]

    adjusted, significant = benjamini_hochberg(
        p_values,
        alpha=0.05,
    )

    assert len(adjusted) == 4
    assert len(significant) == 4

    assert significant[1] is True
    assert significant[2] is False
    assert significant[3] is True
    assert significant[0] is False


def test_benjamini_hochberg_empty_input():
    from src.edge_game.experiments.statistics import benjamini_hochberg

    adjusted, significant = benjamini_hochberg([])

    assert adjusted == []
    assert significant == []


def test_benjamini_hochberg_rejects_invalid_p_values():
    from src.edge_game.experiments.statistics import benjamini_hochberg

    try:
        benjamini_hochberg([0.1, 1.2])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid p-value.")

def test_robustness_interpretation_uses_fdr_significance() -> None:
    """Verify scenario interpretation uses FDR-corrected significance."""

    from src.edge_game.experiments.robustness_analysis import (
        build_robustness_interpretation,
    )

    analysis = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "metric": "resource_utilization",
                "relative_change_percent": 5.0,
                "significant": True,
                "favorable": True,
                "unfavorable": False,
                "effect": "significant_improvement",
            },
            {
                "scenario": "cpu_scarcity",
                "metric": "load_variance",
                "relative_change_percent": 10.0,
                "significant": False,
                "favorable": False,
                "unfavorable": False,
                "effect": "not_significant",
            },
            {
                "scenario": "cpu_scarcity",
                "metric": "response_time_mean",
                "relative_change_percent": -2.0,
                "significant": False,
                "favorable": False,
                "unfavorable": False,
                "effect": "not_significant",
            },
        ]
    )

    interpretation = build_robustness_interpretation(
        analysis
    )

    assert len(interpretation) == 1

    row = interpretation.iloc[0]

    assert row["scenario"] == "cpu_scarcity"

    assert row["significant_improvements"] == 1

    assert row["significant_regressions"] == 0

    assert row["net_significant_effects"] == 1

    assert (
        row["strongest_improvement_metric"]
        == "resource_utilization"
    )

    assert (
        row["strongest_regression_metric"]
        == ""
    )

    assert (
        row["overall_classification"]
        == "improvement"
    )


def test_robustness_interpretation_classifies_regression_risk() -> None:
    """Verify scenarios with only significant regressions are flagged."""

    from src.edge_game.experiments.robustness_analysis import (
        build_robustness_interpretation,
    )

    analysis = pd.DataFrame(
        [
            {
                "scenario": "memory_bandwidth_stress",
                "metric": "jains_fairness_index",
                "relative_change_percent": -1.87,
                "significant": True,
                "favorable": False,
                "unfavorable": True,
                "effect": "significant_regression",
            },
            {
                "scenario": "memory_bandwidth_stress",
                "metric": "load_variance",
                "relative_change_percent": 16.03,
                "significant": True,
                "favorable": False,
                "unfavorable": True,
                "effect": "significant_regression",
            },
            {
                "scenario": "memory_bandwidth_stress",
                "metric": "resource_utilization",
                "relative_change_percent": 1.45,
                "significant": False,
                "favorable": False,
                "unfavorable": False,
                "effect": "not_significant",
            },
        ]
    )

    interpretation = build_robustness_interpretation(
        analysis
    )

    assert len(interpretation) == 1

    row = interpretation.iloc[0]

    assert row["significant_improvements"] == 0

    assert row["significant_regressions"] == 2

    assert row["net_significant_effects"] == -2

    assert (
        row["strongest_improvement_metric"]
        == ""
    )

    assert (
        row["strongest_regression_metric"]
        == "load_variance"
    )

    assert (
        row["overall_classification"]
        == "regression_risk"
    )


def test_robustness_interpretation_classifies_mixed_effects() -> None:
    """Verify scenarios containing both effect directions are marked mixed."""

    from src.edge_game.experiments.robustness_analysis import (
        build_robustness_interpretation,
    )

    analysis = pd.DataFrame(
        [
            {
                "scenario": "mixed_stress",
                "metric": "resource_utilization",
                "relative_change_percent": 2.0,
                "significant": True,
                "favorable": True,
                "unfavorable": False,
                "effect": "significant_improvement",
            },
            {
                "scenario": "mixed_stress",
                "metric": "response_time_mean",
                "relative_change_percent": 3.0,
                "significant": True,
                "favorable": False,
                "unfavorable": True,
                "effect": "significant_regression",
            },
        ]
    )

    interpretation = build_robustness_interpretation(
        analysis
    )

    row = interpretation.iloc[0]

    assert row["significant_improvements"] == 1

    assert row["significant_regressions"] == 1

    assert row["net_significant_effects"] == 0

    assert (
        row["overall_classification"]
        == "mixed"
    )


def test_robustness_interpretation_classifies_neutral() -> None:
    """Verify scenarios without FDR-significant effects are neutral."""

    from src.edge_game.experiments.robustness_analysis import (
        build_robustness_interpretation,
    )

    analysis = pd.DataFrame(
        [
            {
                "scenario": "nominal",
                "metric": "utility_mean",
                "relative_change_percent": 1.5,
                "significant": False,
                "favorable": False,
                "unfavorable": False,
                "effect": "not_significant",
            },
            {
                "scenario": "nominal",
                "metric": "throughput",
                "relative_change_percent": 0.2,
                "significant": False,
                "favorable": False,
                "unfavorable": False,
                "effect": "not_significant",
            },
        ]
    )

    interpretation = build_robustness_interpretation(
        analysis
    )

    row = interpretation.iloc[0]

    assert row["significant_improvements"] == 0

    assert row["significant_regressions"] == 0

    assert row["net_significant_effects"] == 0

    assert (
        row["strongest_improvement_metric"]
        == ""
    )

    assert (
        row["strongest_regression_metric"]
        == ""
    )

    assert (
        row["overall_classification"]
        == "neutral"
    )

def test_scenario_summary_preserves_significant_tradeoffs():
    interpretation = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "metric": "resource_utilization",
                "direction": "favorable",
                "classification": "significant_improvement",
                "significant": True,
            },
            {
                "scenario": "cpu_scarcity",
                "metric": "throughput",
                "direction": "unfavorable",
                "classification": "directional_regression",
                "significant": False,
            },
        ]
    )

    result = build_scenario_summary(
        interpretation
    )

    row = result.iloc[0]

    assert row["significant_improvements"] == 1
    assert row["significant_regressions"] == 0
    assert row["directional_improvements"] == 0
    assert row["directional_regressions"] == 1

    assert (
        row["overall_classification"]
        == "significant_improvement_with_directional_tradeoffs"
    )


def test_scenario_summary_preserves_significant_regression_tradeoffs():
    interpretation = pd.DataFrame(
        [
            {
                "scenario": "memory_bandwidth_stress",
                "metric": "load_variance",
                "direction": "unfavorable",
                "classification": "significant_regression",
                "significant": True,
            },
            {
                "scenario": "memory_bandwidth_stress",
                "metric": "response_time_mean",
                "direction": "favorable",
                "classification": "directional_improvement",
                "significant": False,
            },
        ]
    )

    result = build_scenario_summary(
        interpretation
    )

    row = result.iloc[0]

    assert row["significant_improvements"] == 0
    assert row["significant_regressions"] == 1
    assert row["directional_improvements"] == 1
    assert row["directional_regressions"] == 0

    assert (
        row["overall_classification"]
        == "significant_regression_with_directional_improvements"
    )


def test_scenario_summary_classifies_directional_net_improvement():
    interpretation = pd.DataFrame(
        [
            {
                "scenario": "nominal",
                "metric": "throughput",
                "direction": "favorable",
                "classification": "directional_improvement",
                "significant": False,
            },
            {
                "scenario": "nominal",
                "metric": "success_ratio",
                "direction": "favorable",
                "classification": "directional_improvement",
                "significant": False,
            },
            {
                "scenario": "nominal",
                "metric": "fairness",
                "direction": "unfavorable",
                "classification": "directional_regression",
                "significant": False,
            },
        ]
    )

    result = build_scenario_summary(
        interpretation
    )

    assert (
        result.iloc[0]["overall_classification"]
        == "directional_net_improvement"
    )

def test_robustness_report_contains_fdr_results(
    tmp_path,
):
    from src.edge_game.experiments.robustness_report import (
        build_robustness_report,
    )

    effect_analysis = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "metric": "resource_utilization",
                "alpha": 0.05,
                "raw_significant": True,
                "significant": True,
                "favorable": True,
                "unfavorable": False,
                "relative_change_percent": 1.46,
                "adjusted_p_value": 0.0014,
            }
        ]
    )

    tradeoff_summary = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "significant_improvements": 1,
                "significant_regressions": 0,
                "net_significant_effects": 1,
            }
        ]
    )

    diagnostic_summary = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "improvements": 2,
                "regressions": 8,
                "neutral": 0,
                "significant_improvements": 1,
                "significant_regressions": 0,
                "directional_improvements": 1,
                "directional_regressions": 8,
                "indeterminate": 0,
                "overall_classification": (
                    "significant_improvement_with_"
                    "directional_tradeoffs"
                ),
            }
        ]
    )

    diagnostic_interpretation = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "comparison_policy": (
                    "priority_aware_mean_field"
                ),
                "metric": "resource_utilization",
                "relative_change_percent": 1.46,
                "direction": "favorable",
                "classification": (
                    "significant_improvement"
                ),
            }
        ]
    )

    raw = pd.DataFrame(
        [
            {
                "scenario": "cpu_scarcity",
                "policy_name": "least_loaded_baseline",
                "seed": 42,
            },
            {
                "scenario": "cpu_scarcity",
                "policy_name": (
                    "priority_aware_mean_field"
                ),
                "seed": 42,
            },
        ]
    )

    effect_analysis.to_csv(
        tmp_path
        / "robustness_effect_analysis.csv",
        index=False,
    )

    tradeoff_summary.to_csv(
        tmp_path
        / "robustness_tradeoff_summary.csv",
        index=False,
    )

    diagnostic_summary.to_csv(
        tmp_path
        / "robustness_diagnostic_scenario_summary.csv",
        index=False,
    )

    diagnostic_interpretation.to_csv(
        tmp_path
        / "robustness_diagnostic_interpretation.csv",
        index=False,
    )

    raw.to_csv(
        tmp_path
        / "robustness_raw.csv",
        index=False,
    )

    report = build_robustness_report(
        tmp_path
    )

    assert (
        "Benjamini-Hochberg" in report
    )

    assert (
        "resource_utilization" in report
    )

    assert (
        "cpu_scarcity" in report
    )

    assert (
        "significant_improvement_with_"
        "directional_tradeoffs"
        in report
    )

def test_build_mean_field_policy_supports_full_ablation() -> None:
    """Verify the default Mean-Field formulation remains unchanged."""
    from src.edge_game.experiments.runner import (
        build_mean_field_policy,
    )

    config = SimulationConfig(
        mean_field_state_points=21,
        mean_field_max_iterations=3,
        fpk_max_iterations=5,
    )

    _, diagnostics = build_mean_field_policy(
        config,
    )

    assert diagnostics["ablation_variant"] == "full"


def test_build_mean_field_policy_supports_no_latency_ablation() -> None:
    """Verify the latency ablation reaches the Mean-Field model."""
    from src.edge_game.experiments.runner import (
        build_mean_field_policy,
    )

    config = SimulationConfig(
        mean_field_state_points=21,
        mean_field_max_iterations=3,
        fpk_max_iterations=5,
    )

    _, diagnostics = build_mean_field_policy(
        config,
        ablation_variant="no_latency",
    )

    assert diagnostics["ablation_variant"] == "no_latency"


def test_invalid_ablation_variant_is_rejected() -> None:
    """Verify unsupported ablation variants fail explicitly."""
    from src.edge_game.experiments.runner import (
        build_mean_field_policy,
    )

    config = SimulationConfig(
        mean_field_state_points=21,
        mean_field_max_iterations=1,
        fpk_max_iterations=2,
    )

    with pytest.raises(ValueError, match="Unsupported ablation variant"):
        build_mean_field_policy(
            config,
            ablation_variant="invalid_variant",
        )

def test_ablation_variants_are_complete() -> None:
    """Verify the configured ablation variants cover the full formulation."""
    from src.edge_game.models.mean_field_model import (
        ABLATION_VARIANTS,
    )

    expected = {
        "full",
        "no_priority",
        "no_priority_reward",
        "no_latency",
        "no_resource",
        "no_queue",
        "no_energy",
    }

    assert ABLATION_VARIANTS == expected


def test_ablation_comparison_uses_full_as_reference() -> None:
    """Verify ablation comparisons are calculated relative to full."""
    from src.edge_game.experiments.ablation import (
        build_ablation_comparison,
    )

    rows = []

    for seed in (1, 2, 3):
        base = {
            "scenario": "default",
            "variant": "full",
            "seed": seed,
        }

        for metric in [
            "utility_mean",
            "response_time_mean",
            "throughput",
            "success_ratio",
            "rejected_tasks",
            "resource_utilization",
            "load_variance",
            "jains_fairness_index",
            "average_queue_length",
            "priority_success_ratio",
        ]:
            base[metric] = 10.0

        rows.append(base)

        ablation = base.copy()
        ablation["variant"] = "no_latency"
        ablation["response_time_mean"] = 8.0

        rows.append(ablation)

    comparison = build_ablation_comparison(
        pd.DataFrame(rows)
    )

    response = comparison.loc[
        comparison["metric"] == "response_time_mean"
    ].iloc[0]

    assert response["baseline_variant"] == "full"
    assert np.isclose(
        response["baseline_mean"],
        10.0,
    )
    assert np.isclose(
        response["comparison_mean"],
        8.0,
    )
    assert np.isclose(
        response["relative_change_percent"],
        -20.0,
    )
    assert response["direction"] == "improvement"


def test_ablation_comparison_detects_regression() -> None:
    """Verify an unfavorable ablation change is classified correctly."""
    from src.edge_game.experiments.ablation import (
        build_ablation_comparison,
    )

    rows = []

    for seed in (1, 2, 3):
        full = {
            "scenario": "default",
            "variant": "full",
            "seed": seed,
        }

        ablation = {
            "scenario": "default",
            "variant": "no_queue",
            "seed": seed,
        }

        for metric in [
            "utility_mean",
            "response_time_mean",
            "throughput",
            "success_ratio",
            "rejected_tasks",
            "resource_utilization",
            "load_variance",
            "jains_fairness_index",
            "average_queue_length",
            "priority_success_ratio",
        ]:
            full[metric] = 10.0
            ablation[metric] = 10.0

        full["utility_mean"] = 10.0
        ablation["utility_mean"] = 8.0

        rows.extend(
            [
                full,
                ablation,
            ]
        )

    comparison = build_ablation_comparison(
        pd.DataFrame(rows)
    )

    utility = comparison.loc[
        comparison["metric"] == "utility_mean"
    ].iloc[0]

    assert utility["direction"] == "regression"
    assert np.isclose(
        utility["relative_change_percent"],
        -20.0,
    )


def test_ablation_requires_full_reference() -> None:
    """Verify ablation experiments reject missing full reference."""
    from src.edge_game.experiments.ablation import (
        run_ablation_experiment,
    )

    config = SimulationConfig(
        mean_field_state_points=11,
        mean_field_max_iterations=1,
        fpk_max_iterations=2,
    )

    with pytest.raises(
        ValueError,
        match="requires the 'full' variant",
    ):
        run_ablation_experiment(
            config=config,
            seeds=[1],
            scenarios=["default"],
            variants=["no_latency"],
        )


def test_ablation_invalid_variant_is_rejected() -> None:
    """Verify unsupported ablation variants fail explicitly."""
    from src.edge_game.experiments.ablation import (
        run_ablation_experiment,
    )

    config = SimulationConfig(
        mean_field_state_points=11,
        mean_field_max_iterations=1,
        fpk_max_iterations=2,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported ablation variant",
    ):
        run_ablation_experiment(
            config=config,
            seeds=[1],
            scenarios=["default"],
            variants=[
                "full",
                "invalid_variant",
            ],
        )


def test_ablation_summary_classifies_significant_effects() -> None:
    """Verify ablation summary classifies significant effects."""
    from src.edge_game.experiments.ablation import (
        build_ablation_summary,
    )

    comparison = pd.DataFrame(
        [
            {
                "scenario": "default",
                "variant": "no_latency",
                "metric": "response_time_mean",
                "effect": "significant_improvement",
            },
            {
                "scenario": "default",
                "variant": "no_latency",
                "metric": "utility_mean",
                "effect": "significant_regression",
            },
        ]
    )

    summary = build_ablation_summary(
        comparison
    )

    row = summary.iloc[0]

    assert row["significant_improvements"] == 1
    assert row["significant_regressions"] == 1
    assert row["net_significant_effects"] == 0
    assert row["overall_classification"] == "mixed"

def test_module2_generates_resource_filtering_selection_audit(tmp_path) -> None:
    """Verify Module-2 writes the professor-facing filtering evidence."""
    from src.edge_game.experiments.module2 import run_module2_experiment

    config = SimulationConfig(
        simulation_steps=2,
        tasks_per_step=2,
        number_of_servers=1,
        nodes_per_server=3,
        mean_field_state_points=11,
        mean_field_max_iterations=3,
        fpk_max_iterations=5,
        output_directory=str(tmp_path),
    )

    output_directory = tmp_path / "module2"
    run_module2_experiment(
        config=config,
        seeds=(1,),
        output_directory=output_directory,
    )

    assert (
        output_directory
        / "raw"
        / "resource_filtering_audit.csv"
    ).exists()
    assert (
        output_directory
        / "aggregated"
        / "resource_filtering_summary.csv"
    ).exists()
    assert (
        output_directory
        / "figures"
        / "resource_filtering_selection_audit.png"
    ).exists()


def test_load_balancer_benchmark_generates_single_and_ten_run_outputs(tmp_path) -> None:
    """Benchmark must generate tabular, plotting, and filtering evidence."""
    from src.edge_game.config import SimulationConfig
    from src.edge_game.experiments.benchmark import run_load_balancer_benchmark

    config = SimulationConfig(
        simulation_steps=4,
        tasks_per_step=2,
        number_of_servers=1,
        nodes_per_server=3,
        mean_field_state_points=11,
        mean_field_max_iterations=5,
        fpk_max_iterations=20,
        benchmark_simulation_steps=4,
        benchmark_number_of_servers=1,
        benchmark_nodes_per_server=3,
        benchmark_mean_field_state_points=11,
        benchmark_mean_field_max_iterations=5,
        benchmark_fpk_max_iterations=20,
    )

    output_directory = tmp_path / "benchmark"
    outputs = run_load_balancer_benchmark(
        config=config,
        seeds=(42, 43),
        output_directory=output_directory,
    )

    assert outputs["output_directory"] == output_directory
    for path in outputs.values():
        assert path.exists()
    assert (output_directory / "raw" / "single_run_metrics.csv").exists()
    assert (output_directory / "raw" / "single_run_node_utilization.csv").exists()
    assert (output_directory / "raw" / "ten_run_benchmark_raw.csv").exists()
    assert (output_directory / "raw" / "ten_run_node_utilization_raw.csv").exists()
    assert (output_directory / "aggregated" / "ten_run_benchmark_summary.csv").exists()
    assert (output_directory / "aggregated" / "ten_run_node_utilization_summary.csv").exists()
    assert (output_directory / "figures" / "single_run_node_utilization.png").exists()
    assert (output_directory / "figures" / "ten_run_node_utilization_time_series.png").exists()
    assert (output_directory / "figures" / "resource_filtering_selection_audit.png").exists()
