"""
Main entry point for the Multi-Agent Debate System.
Solves GSM8K math problems using multiple LLM agents via Ollama.

Usage:
    python main.py --mode test           # Quick test with 5 questions
    python main.py --mode evaluate       # Full evaluation (100 samples)
    python main.py --mode experiments    # Run all ablation studies
    python main.py --mode demo           # Interactive demo with single question
"""

import argparse
import sys
from typing import Optional

from debate_agent import (
    MultiAgentDebate,
    single_agent_solve,
    check_models_available,
    pull_model_if_needed
)
from evaluation import (
    load_gsm8k,
    evaluate_debate_system,
    evaluate_single_agent,
    compare_methods,
    analyze_debate_patterns,
    quick_test
)
from experiments import run_all_experiments
from utils import save_metrics, answers_match


# Default models
DEFAULT_MODELS = ["llama3.1:8b", "mistral:7b", "phi3:medium"]


def check_and_setup_models(models: list) -> bool:
    """
    Check if required models are available and offer to pull them.

    Args:
        models: List of model names to check

    Returns:
        True if all models are available
    """
    print("Checking model availability...")
    availability = check_models_available(models)

    missing = [m for m, available in availability.items() if not available]

    if not missing:
        print("All models are available!")
        return True

    print(f"\nMissing models: {missing}")
    print("Would you like to pull them? (y/n): ", end="")

    response = input().strip().lower()
    if response != 'y':
        print("Cannot proceed without all models.")
        return False

    for model in missing:
        success = pull_model_if_needed(model)
        if not success:
            print(f"Failed to pull {model}. Please pull manually with: ollama pull {model}")
            return False

    print("All models are now available!")
    return True


def interactive_demo():
    """Run an interactive demo with a single question."""
    print("=" * 60)
    print("MULTI-AGENT DEBATE DEMO")
    print("=" * 60)

    # Get question from user or use default
    print("\nEnter a math problem (or press Enter for example):")
    user_input = input().strip()

    if not user_input:
        question = """Janet's ducks lay 16 eggs per day. She eats three for breakfast
every morning and bakes muffins for her friends every day with four. She sells the remainder
at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make
every day at the farmers' market?"""
        print(f"\nUsing example question:\n{question}")
    else:
        question = user_input

    print("\n" + "-" * 60)

    # Check models
    if not check_and_setup_models(DEFAULT_MODELS):
        return

    # Run single agent first
    print("\n--- SINGLE AGENT BASELINE ---")
    single_result = single_agent_solve(question, model=DEFAULT_MODELS[0], verbose=True)
    single_answer = single_result["answer"]

    # Run debate
    print("\n--- MULTI-AGENT DEBATE ---")
    debate = MultiAgentDebate(models=DEFAULT_MODELS)
    debate_result = debate.debate(question, n_rounds=2, verbose=True)
    debate_answer = debate_result["consensus"]

    # Summary
    print("\n" + "=" * 60)
    print("DEMO SUMMARY")
    print("=" * 60)
    print(f"Single Agent Answer: {single_answer}")
    print(f"Debate Consensus:    {debate_answer}")

    if single_answer != debate_answer:
        print("\n*** Agents reached different conclusions! ***")
        print("The debate process may have helped correct an error or")
        print("there may be legitimate disagreement on interpretation.")


def run_test_mode(n_samples: int = 5, verbose: bool = True):
    """Run quick test mode."""
    print("=" * 60)
    print(f"TEST MODE ({n_samples} samples)")
    print("=" * 60)

    if not check_and_setup_models(DEFAULT_MODELS):
        return

    quick_test(n_questions=n_samples, verbose=verbose)


def run_evaluation_mode(n_samples: int = 100, n_rounds: int = 2, verbose: bool = False):
    """Run full evaluation mode."""
    print("=" * 60)
    print(f"EVALUATION MODE ({n_samples} samples, {n_rounds} rounds)")
    print("=" * 60)

    if not check_and_setup_models(DEFAULT_MODELS):
        return

    samples = load_gsm8k(split="test", n_samples=n_samples)

    # Run comparison
    comparison = compare_methods(
        samples=samples,
        debate_models=DEFAULT_MODELS,
        baseline_model=DEFAULT_MODELS[0],
        n_rounds=n_rounds,
        verbose=verbose
    )

    # Analyze debate patterns
    if comparison.get("debate"):
        analyze_debate_patterns(comparison["debate"])

    # Save results
    save_metrics(comparison, f"evaluation_n{n_samples}_r{n_rounds}.json")

    print("\nEvaluation complete! Results saved to results/metrics/")


def run_experiments_mode(n_samples: int = 50, verbose: bool = False):
    """Run ablation experiments."""
    print("=" * 60)
    print(f"EXPERIMENTS MODE ({n_samples} samples)")
    print("=" * 60)

    if not check_and_setup_models(DEFAULT_MODELS):
        return

    results = run_all_experiments(
        n_samples=n_samples,
        save_results=True,
        verbose=verbose
    )

    print("\nExperiments complete! Results saved to results/metrics/")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Debate System for GSM8K Math Problems"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "evaluate", "experiments", "demo"],
        default="test",
        help="Execution mode (default: test)"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: mode-dependent)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Custom models to use (space-separated)"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)"
    )

    args = parser.parse_args()

    # Update default models if custom ones provided
    global DEFAULT_MODELS
    if args.models:
        DEFAULT_MODELS = args.models
        print(f"Using custom models: {DEFAULT_MODELS}")

    # Determine sample count based on mode
    if args.samples is not None:
        n_samples = args.samples
    else:
        n_samples = {
            "test": 5,
            "evaluate": 100,
            "experiments": 50,
            "demo": 1
        }.get(args.mode, 10)

    # Run appropriate mode
    if args.mode == "demo":
        interactive_demo()
    elif args.mode == "test":
        run_test_mode(n_samples=n_samples, verbose=args.verbose)
    elif args.mode == "evaluate":
        run_evaluation_mode(n_samples=n_samples, n_rounds=args.rounds, verbose=args.verbose)
    elif args.mode == "experiments":
        run_experiments_mode(n_samples=n_samples, verbose=args.verbose)
    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
