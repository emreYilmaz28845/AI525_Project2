"""
Ablation studies and experiments for multi-agent debate system.
Tests different configurations to understand what contributes to performance.
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils import save_metrics
from evaluation import (
    load_gsm8k,
    evaluate_debate_system,
    evaluate_single_agent,
    analyze_debate_patterns
)


def experiment_num_agents(
    samples: List[Dict[str, Any]],
    base_models: List[str] = None,
    n_rounds: int = 2,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Ablation: Compare 2 agents vs 3 agents.

    Args:
        samples: GSM8K samples
        base_models: List of 3 models to use
        n_rounds: Number of debate rounds
        verbose: Print detailed output

    Returns:
        Experiment results
    """
    if base_models is None:
        base_models = ["llama3.1:8b", "qwen2.5:7b", "phi3:medium"]

    print("=" * 60)
    print("EXPERIMENT: Number of Agents")
    print("=" * 60)

    results = {}

    # 2 agents
    print("\n--- 2 Agents ---")
    two_agent_results = evaluate_debate_system(
        samples=samples,
        models=base_models[:2],
        n_rounds=n_rounds,
        verbose=verbose,
        log_results=False
    )
    results["2_agents"] = {
        "models": base_models[:2],
        "accuracy": two_agent_results["accuracy"],
        "correct": two_agent_results["correct"],
        "total": two_agent_results["total_samples"]
    }

    # 3 agents
    print("\n--- 3 Agents ---")
    three_agent_results = evaluate_debate_system(
        samples=samples,
        models=base_models[:3],
        n_rounds=n_rounds,
        verbose=verbose,
        log_results=False
    )
    results["3_agents"] = {
        "models": base_models[:3],
        "accuracy": three_agent_results["accuracy"],
        "correct": three_agent_results["correct"],
        "total": three_agent_results["total_samples"]
    }

    # Summary
    print("\n" + "=" * 60)
    print("AGENT COUNT COMPARISON")
    print("=" * 60)
    print(f"2 Agents: {results['2_agents']['accuracy']:.2%}")
    print(f"3 Agents: {results['3_agents']['accuracy']:.2%}")
    diff = results['3_agents']['accuracy'] - results['2_agents']['accuracy']
    print(f"Difference: {diff:+.2%}")

    return {
        "experiment": "num_agents",
        "n_rounds": n_rounds,
        "results": results
    }


def experiment_num_rounds(
    samples: List[Dict[str, Any]],
    models: List[str] = None,
    max_rounds: int = 3,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Ablation: Compare different numbers of debate rounds.

    Args:
        samples: GSM8K samples
        models: List of models for agents
        max_rounds: Maximum number of rounds to test
        verbose: Print detailed output

    Returns:
        Experiment results
    """
    if models is None:
        models = ["llama3.1:8b", "qwen2.5:7b", "phi3:medium"]

    print("=" * 60)
    print("EXPERIMENT: Number of Rounds")
    print("=" * 60)

    results = {}

    for n_rounds in range(1, max_rounds + 1):
        print(f"\n--- {n_rounds} Round(s) ---")
        round_results = evaluate_debate_system(
            samples=samples,
            models=models,
            n_rounds=n_rounds,
            verbose=verbose,
            log_results=False
        )
        results[f"{n_rounds}_rounds"] = {
            "n_rounds": n_rounds,
            "accuracy": round_results["accuracy"],
            "correct": round_results["correct"],
            "total": round_results["total_samples"]
        }

    # Summary
    print("\n" + "=" * 60)
    print("ROUNDS COMPARISON")
    print("=" * 60)
    for key, val in results.items():
        print(f"{val['n_rounds']} round(s): {val['accuracy']:.2%}")

    return {
        "experiment": "num_rounds",
        "models": models,
        "results": results
    }


def experiment_model_combinations(
    samples: List[Dict[str, Any]],
    n_rounds: int = 2,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Ablation: Compare different model combinations.

    Args:
        samples: GSM8K samples
        n_rounds: Number of debate rounds
        verbose: Print detailed output

    Returns:
        Experiment results
    """
    combinations = [
        # Homogeneous (same model)
        ("homogeneous_llama", ["llama3.1:8b", "llama3.1:8b", "llama3.1:8b"]),
        ("homogeneous_qwen", ["qwen2.5:7b", "qwen2.5:7b", "qwen2.5:7b"]),
        # Heterogeneous (different models)
        ("heterogeneous_default", ["llama3.1:8b", "qwen2.5:7b", "phi3:medium"]),
        ("heterogeneous_llama_qwen", ["llama3.1:8b", "llama3.1:8b", "qwen2.5:7b"]),
    ]

    print("=" * 60)
    print("EXPERIMENT: Model Combinations")
    print("=" * 60)

    results = {}

    for name, models in combinations:
        print(f"\n--- {name} ---")
        print(f"Models: {models}")

        try:
            combo_results = evaluate_debate_system(
                samples=samples,
                models=models,
                n_rounds=n_rounds,
                verbose=verbose,
                log_results=False
            )
            results[name] = {
                "models": models,
                "accuracy": combo_results["accuracy"],
                "correct": combo_results["correct"],
                "total": combo_results["total_samples"]
            }
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = {
                "models": models,
                "error": str(e)
            }

    # Summary
    print("\n" + "=" * 60)
    print("MODEL COMBINATIONS COMPARISON")
    print("=" * 60)
    for name, val in results.items():
        if "accuracy" in val:
            print(f"{name}: {val['accuracy']:.2%}")
        else:
            print(f"{name}: ERROR - {val.get('error', 'Unknown')}")

    return {
        "experiment": "model_combinations",
        "n_rounds": n_rounds,
        "results": results
    }


def experiment_baseline_comparison(
    samples: List[Dict[str, Any]],
    models: List[str] = None,
    n_rounds: int = 2,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compare each individual model as baseline vs debate.

    Args:
        samples: GSM8K samples
        models: List of models
        n_rounds: Number of debate rounds
        verbose: Print detailed output

    Returns:
        Comparison results
    """
    if models is None:
        models = ["llama3.1:8b", "qwen2.5:7b", "phi3:medium"]

    print("=" * 60)
    print("EXPERIMENT: Baseline vs Debate")
    print("=" * 60)

    results = {"baselines": {}, "debate": None}

    # Evaluate each model as single agent
    for model in models:
        print(f"\n--- Baseline: {model} ---")
        try:
            baseline = evaluate_single_agent(
                samples=samples,
                model=model,
                verbose=verbose
            )
            results["baselines"][model] = {
                "accuracy": baseline["accuracy"],
                "correct": baseline["correct"],
                "total": baseline["total_samples"]
            }
        except Exception as e:
            print(f"Error with {model}: {e}")
            results["baselines"][model] = {"error": str(e)}

    # Evaluate debate system
    print(f"\n--- Multi-Agent Debate ---")
    try:
        debate = evaluate_debate_system(
            samples=samples,
            models=models,
            n_rounds=n_rounds,
            verbose=verbose,
            log_results=False
        )
        results["debate"] = {
            "models": models,
            "accuracy": debate["accuracy"],
            "correct": debate["correct"],
            "total": debate["total_samples"]
        }
    except Exception as e:
        print(f"Debate error: {e}")
        results["debate"] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE VS DEBATE COMPARISON")
    print("=" * 60)

    best_baseline = 0
    best_model = None
    for model, val in results["baselines"].items():
        if "accuracy" in val:
            print(f"{model}: {val['accuracy']:.2%}")
            if val["accuracy"] > best_baseline:
                best_baseline = val["accuracy"]
                best_model = model

    if results["debate"] and "accuracy" in results["debate"]:
        debate_acc = results["debate"]["accuracy"]
        print(f"\nDebate: {debate_acc:.2%}")
        print(f"\nBest baseline ({best_model}): {best_baseline:.2%}")
        print(f"Debate improvement over best baseline: {debate_acc - best_baseline:+.2%}")

    return {
        "experiment": "baseline_comparison",
        "n_rounds": n_rounds,
        "results": results
    }


def run_all_experiments(
    n_samples: int = 50,
    save_results: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run all ablation experiments.

    Args:
        n_samples: Number of GSM8K samples to use
        save_results: Whether to save results to file
        verbose: Print detailed output

    Returns:
        All experiment results
    """
    print("=" * 60)
    print("RUNNING ALL EXPERIMENTS")
    print(f"Samples: {n_samples}")
    print("=" * 60)

    # Load data once
    samples = load_gsm8k(split="test", n_samples=n_samples)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "experiments": {}
    }

    # Experiment 1: Baseline comparison
    print("\n\n" + "=" * 80)
    print("EXPERIMENT 1: Baseline Comparison")
    print("=" * 80)
    exp1 = experiment_baseline_comparison(samples, verbose=verbose)
    all_results["experiments"]["baseline_comparison"] = exp1

    # Experiment 2: Number of agents
    print("\n\n" + "=" * 80)
    print("EXPERIMENT 2: Number of Agents")
    print("=" * 80)
    exp2 = experiment_num_agents(samples, verbose=verbose)
    all_results["experiments"]["num_agents"] = exp2

    # Experiment 3: Number of rounds
    print("\n\n" + "=" * 80)
    print("EXPERIMENT 3: Number of Rounds")
    print("=" * 80)
    exp3 = experiment_num_rounds(samples, max_rounds=3, verbose=verbose)
    all_results["experiments"]["num_rounds"] = exp3

    # Experiment 4: Model combinations
    print("\n\n" + "=" * 80)
    print("EXPERIMENT 4: Model Combinations")
    print("=" * 80)
    exp4 = experiment_model_combinations(samples, verbose=verbose)
    all_results["experiments"]["model_combinations"] = exp4

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_experiments_{timestamp}.json"
        save_metrics(all_results, filename)

    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print_experiment_summary(all_results)

    return all_results


def print_experiment_summary(results: Dict[str, Any]):
    """Print a summary of all experiment results."""

    experiments = results.get("experiments", {})

    # Baseline comparison
    if "baseline_comparison" in experiments:
        exp = experiments["baseline_comparison"]["results"]
        print("\n--- Baseline vs Debate ---")
        for model, val in exp.get("baselines", {}).items():
            if "accuracy" in val:
                print(f"  {model}: {val['accuracy']:.2%}")
        if exp.get("debate") and "accuracy" in exp["debate"]:
            print(f"  Debate: {exp['debate']['accuracy']:.2%}")

    # Number of agents
    if "num_agents" in experiments:
        exp = experiments["num_agents"]["results"]
        print("\n--- Agent Count ---")
        for key, val in exp.items():
            if "accuracy" in val:
                print(f"  {key}: {val['accuracy']:.2%}")

    # Number of rounds
    if "num_rounds" in experiments:
        exp = experiments["num_rounds"]["results"]
        print("\n--- Round Count ---")
        for key, val in exp.items():
            if "accuracy" in val:
                print(f"  {val['n_rounds']} round(s): {val['accuracy']:.2%}")

    # Model combinations
    if "model_combinations" in experiments:
        exp = experiments["model_combinations"]["results"]
        print("\n--- Model Combinations ---")
        for name, val in exp.items():
            if "accuracy" in val:
                print(f"  {name}: {val['accuracy']:.2%}")


if __name__ == "__main__":
    # Run a small test
    run_all_experiments(n_samples=10, save_results=True, verbose=False)
