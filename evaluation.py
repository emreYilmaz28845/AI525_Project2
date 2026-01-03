"""
Evaluation functions for the multi-agent debate system.
Handles loading GSM8K dataset and computing accuracy metrics.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datasets import load_dataset
from tqdm import tqdm

from utils import (
    extract_gsm8k_answer,
    answers_match,
    DebateLogger,
    save_metrics
)
from debate_agent import MultiAgentDebate, single_agent_solve


def load_gsm8k(split: str = "test", n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.

    Args:
        split: Dataset split to load ("train" or "test")
        n_samples: Number of samples to load (None for all)

    Returns:
        List of question-answer pairs
    """
    print(f"Loading GSM8K {split} split...")
    dataset = load_dataset("gsm8k", "main", split=split)

    samples = []
    for i, item in enumerate(dataset):
        if n_samples and i >= n_samples:
            break

        ground_truth = extract_gsm8k_answer(item["answer"])
        samples.append({
            "id": i,
            "question": item["question"],
            "full_answer": item["answer"],
            "ground_truth": ground_truth
        })

    print(f"Loaded {len(samples)} samples")
    return samples


def evaluate_debate_system(
    samples: List[Dict[str, Any]],
    models: List[str] = None,
    n_rounds: int = 2,
    verbose: bool = False,
    log_results: bool = True
) -> Dict[str, Any]:
    """
    Evaluate multi-agent debate system on GSM8K samples.

    Args:
        samples: List of GSM8K samples
        models: List of models for agents
        n_rounds: Number of debate rounds
        verbose: Whether to print detailed output
        log_results: Whether to log detailed results

    Returns:
        Dictionary with evaluation metrics
    """
    if models is None:
        models = ["llama3.1:8b", "qwen2.5:7b", "phi3:medium"]

    debate_system = MultiAgentDebate(models=models)
    logger = DebateLogger() if log_results else None

    correct = 0
    total = 0
    results = []

    print(f"\nEvaluating multi-agent debate ({len(models)} agents, {n_rounds} rounds)")
    print(f"Models: {models}")
    print("-" * 60)

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        try:
            debate_result = debate_system.debate(
                question=question,
                n_rounds=n_rounds,
                verbose=verbose
            )

            predicted = debate_result["consensus"]
            is_correct = answers_match(predicted, ground_truth)

            if is_correct:
                correct += 1
            total += 1

            result_entry = {
                "id": sample["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct,
                "final_answers": debate_result["final_answers"],
                "answer_changes": debate_result["answer_changes"]
            }
            results.append(result_entry)

            if logger:
                logger.log_debate(
                    question_id=sample["id"],
                    question=question,
                    ground_truth=ground_truth,
                    rounds=debate_result["rounds"],
                    final_answers=list(debate_result["final_answers"].values()),
                    consensus=predicted,
                    is_correct=is_correct
                )

            if verbose:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"Q{sample['id']}: {status} (pred={predicted}, truth={ground_truth})")

        except Exception as e:
            print(f"Error on question {sample['id']}: {e}")
            total += 1
            results.append({
                "id": sample["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": None,
                "is_correct": False,
                "error": str(e)
            })

    if logger:
        logger.save()

    accuracy = correct / total if total > 0 else 0

    metrics = {
        "method": "multi_agent_debate",
        "models": models,
        "n_agents": len(models),
        "n_rounds": n_rounds,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }

    print(f"\n{'='*60}")
    print(f"DEBATE SYSTEM RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

    return metrics


def evaluate_single_agent(
    samples: List[Dict[str, Any]],
    model: str = "llama3.1:8b",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Evaluate single agent baseline on GSM8K samples.

    Args:
        samples: List of GSM8K samples
        model: Model to use
        verbose: Whether to print detailed output

    Returns:
        Dictionary with evaluation metrics
    """
    correct = 0
    total = 0
    results = []

    print(f"\nEvaluating single agent baseline")
    print(f"Model: {model}")
    print("-" * 60)

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        try:
            result = single_agent_solve(
                question=question,
                model=model,
                verbose=verbose
            )

            predicted = result["answer"]
            is_correct = answers_match(predicted, ground_truth)

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "id": sample["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "is_correct": is_correct
            })

            if verbose:
                status = "CORRECT" if is_correct else "WRONG"
                print(f"Q{sample['id']}: {status} (pred={predicted}, truth={ground_truth})")

        except Exception as e:
            print(f"Error on question {sample['id']}: {e}")
            total += 1
            results.append({
                "id": sample["id"],
                "question": question,
                "ground_truth": ground_truth,
                "predicted": None,
                "is_correct": False,
                "error": str(e)
            })

    accuracy = correct / total if total > 0 else 0

    metrics = {
        "method": "single_agent",
        "model": model,
        "total_samples": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results
    }

    print(f"\n{'='*60}")
    print(f"SINGLE AGENT RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")

    return metrics


def compare_methods(samples: List[Dict[str, Any]],
                    debate_models: List[str] = None,
                    baseline_model: str = "llama3.1:8b",
                    n_rounds: int = 2,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    Compare single agent vs multi-agent debate.

    Args:
        samples: GSM8K samples to evaluate
        debate_models: Models for debate system
        baseline_model: Model for single agent baseline
        n_rounds: Number of debate rounds
        verbose: Print detailed output

    Returns:
        Comparison results
    """
    print("=" * 60)
    print("COMPARISON: Single Agent vs Multi-Agent Debate")
    print("=" * 60)

    # Baseline evaluation
    baseline_results = evaluate_single_agent(
        samples=samples,
        model=baseline_model,
        verbose=verbose
    )

    # Debate evaluation
    debate_results = evaluate_debate_system(
        samples=samples,
        models=debate_models,
        n_rounds=n_rounds,
        verbose=verbose
    )

    # Calculate improvement
    baseline_acc = baseline_results["accuracy"]
    debate_acc = debate_results["accuracy"]
    improvement = debate_acc - baseline_acc

    comparison = {
        "baseline": baseline_results,
        "debate": debate_results,
        "improvement": improvement,
        "improvement_pct": improvement * 100,
        "relative_improvement": improvement / baseline_acc if baseline_acc > 0 else 0
    }

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Baseline Accuracy:    {baseline_acc:.2%}")
    print(f"Debate Accuracy:      {debate_acc:.2%}")
    print(f"Improvement:          {improvement:+.2%}")
    print(f"Relative Improvement: {comparison['relative_improvement']:+.1%}")

    return comparison


def analyze_debate_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze patterns in debate results.

    Args:
        results: Results from evaluate_debate_system

    Returns:
        Analysis of debate patterns
    """
    detailed_results = results.get("results", [])

    if not detailed_results:
        return {}

    # Count how often agents changed answers
    total_debates = len(detailed_results)
    debates_with_changes = 0
    correct_after_change = 0
    correct_without_change = 0

    for r in detailed_results:
        changes = r.get("answer_changes", {})
        had_changes = any(changes.values())

        if had_changes:
            debates_with_changes += 1
            if r["is_correct"]:
                correct_after_change += 1
        else:
            if r["is_correct"]:
                correct_without_change += 1

    no_changes = total_debates - debates_with_changes

    analysis = {
        "total_debates": total_debates,
        "debates_with_changes": debates_with_changes,
        "debates_without_changes": no_changes,
        "change_rate": debates_with_changes / total_debates if total_debates > 0 else 0,
        "accuracy_when_changed": correct_after_change / debates_with_changes if debates_with_changes > 0 else 0,
        "accuracy_when_no_change": correct_without_change / no_changes if no_changes > 0 else 0
    }

    print("\n" + "=" * 60)
    print("DEBATE PATTERN ANALYSIS")
    print("=" * 60)
    print(f"Debates with answer changes: {debates_with_changes}/{total_debates} ({analysis['change_rate']:.1%})")
    print(f"Accuracy when answers changed: {analysis['accuracy_when_changed']:.1%}")
    print(f"Accuracy when no changes: {analysis['accuracy_when_no_change']:.1%}")

    return analysis


def quick_test(n_questions: int = 5, verbose: bool = True):
    """
    Quick test to verify the system works.

    Args:
        n_questions: Number of questions to test
        verbose: Print detailed output
    """
    print("=" * 60)
    print("QUICK TEST")
    print("=" * 60)

    # Load a few samples
    samples = load_gsm8k(split="test", n_samples=n_questions)

    # Test single agent
    print("\n--- Testing Single Agent ---")
    single_result = single_agent_solve(
        question=samples[0]["question"],
        model="llama3.1:8b",
        verbose=True
    )
    print(f"Ground truth: {samples[0]['ground_truth']}")
    print(f"Predicted: {single_result['answer']}")
    print(f"Match: {answers_match(single_result['answer'], samples[0]['ground_truth'])}")

    # Test multi-agent debate
    print("\n--- Testing Multi-Agent Debate ---")
    debate = MultiAgentDebate(models=["llama3.1:8b", "qwen2.5:7b", "phi3:medium"])
    debate_result = debate.debate(
        question=samples[0]["question"],
        n_rounds=2,
        verbose=True
    )
    print(f"Ground truth: {samples[0]['ground_truth']}")
    print(f"Consensus: {debate_result['consensus']}")
    print(f"Match: {answers_match(debate_result['consensus'], samples[0]['ground_truth'])}")

    print("\n" + "=" * 60)
    print("Quick test completed!")
    print("=" * 60)
