"""
Utility functions for the multi-agent debate system.
Includes answer extraction, logging, and helper functions.
"""

import re
import json
import os
from datetime import datetime
from collections import Counter
from typing import Optional, List, Dict, Any


def extract_answer(text: str) -> Optional[float]:
    """
    Extract numerical answer from model response.

    Looks for patterns like:
    - "the answer is X"
    - "= X"
    - "#### X" (GSM8K format)
    - Final number in the text

    Args:
        text: Model response text

    Returns:
        Extracted numerical answer or None if not found
    """
    if not text:
        return None

    # Clean the text
    text = text.strip()

    # Pattern 1: GSM8K format "#### X"
    pattern_gsm8k = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern_gsm8k, text)
    if match:
        return parse_number(match.group(1))

    # Pattern 2: "the answer is X" or "answer: X"
    pattern_answer = r'(?:the\s+)?answer\s*(?:is|:)\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern_answer, text, re.IGNORECASE)
    if match:
        return parse_number(match.group(1))

    # Pattern 3: "= X" at end of equation
    pattern_equals = r'=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:$|[.\n])'
    matches = re.findall(pattern_equals, text)
    if matches:
        return parse_number(matches[-1])

    # Pattern 4: "therefore X" or "so X" or "thus X"
    pattern_therefore = r'(?:therefore|so|thus|hence)[,]?\s*(?:the\s+)?(?:answer\s+is\s+)?(?:it\s+is\s+)?\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern_therefore, text, re.IGNORECASE)
    if match:
        return parse_number(match.group(1))

    # Pattern 5: Look for boxed answer (LaTeX style)
    pattern_boxed = r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}'
    match = re.search(pattern_boxed, text)
    if match:
        return parse_number(match.group(1))

    # Pattern 6: Final number in the text (fallback)
    pattern_final = r'(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    matches = re.findall(pattern_final, text)
    if matches:
        # Return the last number found
        return parse_number(matches[-1])

    return None


def parse_number(num_str: str) -> float:
    """
    Parse a number string, handling commas and decimals.

    Args:
        num_str: String representation of a number

    Returns:
        Parsed float value
    """
    # Remove commas
    num_str = num_str.replace(',', '')
    return float(num_str)


def extract_gsm8k_answer(answer_text: str) -> Optional[float]:
    """
    Extract the ground truth answer from GSM8K dataset format.
    GSM8K answers end with "#### <number>"

    Args:
        answer_text: The answer field from GSM8K dataset

    Returns:
        Numerical answer
    """
    pattern = r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)'
    match = re.search(pattern, answer_text)
    if match:
        return parse_number(match.group(1))
    return None


def majority_vote(answers: List[Optional[float]]) -> Optional[float]:
    """
    Determine consensus answer using majority voting.

    Args:
        answers: List of numerical answers from agents

    Returns:
        Most common answer, or first answer if tie
    """
    # Filter out None values
    valid_answers = [a for a in answers if a is not None]

    if not valid_answers:
        return None

    # Count occurrences
    counter = Counter(valid_answers)

    # Get most common
    most_common = counter.most_common()

    # Return the most common answer
    return most_common[0][0]


def answers_match(pred: Optional[float], truth: Optional[float], tolerance: float = 0.01) -> bool:
    """
    Check if predicted answer matches ground truth.

    Args:
        pred: Predicted answer
        truth: Ground truth answer
        tolerance: Acceptable difference for floating point comparison

    Returns:
        True if answers match within tolerance
    """
    if pred is None or truth is None:
        return False

    # For integers, check exact match
    if pred == int(pred) and truth == int(truth):
        return int(pred) == int(truth)

    # For floats, use tolerance
    return abs(pred - truth) < tolerance


class DebateLogger:
    """Logger for tracking debate progress and results."""

    def __init__(self, log_dir: str = "results/logs"):
        """
        Initialize logger.

        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"debate_log_{timestamp}.json")

        self.logs = []

    def log_debate(self, question_id: int, question: str, ground_truth: float,
                   rounds: List[Dict[str, Any]], final_answers: List[float],
                   consensus: float, is_correct: bool):
        """
        Log a complete debate session.

        Args:
            question_id: Index of the question
            question: The math problem
            ground_truth: Correct answer
            rounds: List of round data (each round has agent responses)
            final_answers: Final answers from each agent
            consensus: Majority vote answer
            is_correct: Whether consensus matches ground truth
        """
        entry = {
            "question_id": question_id,
            "question": question,
            "ground_truth": ground_truth,
            "rounds": rounds,
            "final_answers": final_answers,
            "consensus": consensus,
            "is_correct": is_correct,
            "timestamp": datetime.now().isoformat()
        }
        self.logs.append(entry)

    def save(self):
        """Save logs to file."""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"Logs saved to {self.log_file}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics from logs."""
        if not self.logs:
            return {}

        correct = sum(1 for log in self.logs if log["is_correct"])
        total = len(self.logs)

        return {
            "total_questions": total,
            "correct": correct,
            "accuracy": correct / total if total > 0 else 0
        }


def format_prompt_initial(question: str, agent_id: str) -> str:
    """
    Format prompt for initial answer generation.

    Args:
        question: Math problem to solve
        agent_id: Identifier for the agent

    Returns:
        Formatted prompt string
    """
    return f"""You are Agent {agent_id}, a mathematical reasoning expert. Solve the following math problem step by step.

Problem: {question}

Instructions:
1. Show your work clearly with each calculation step
2. Double-check your arithmetic
3. State your final answer clearly as "The answer is X"

Solve this problem:"""


def format_prompt_debate(question: str, agent_id: str, own_answer: str,
                         other_responses: List[Dict[str, str]], round_num: int) -> str:
    """
    Format prompt for debate round.

    Args:
        question: Original math problem
        agent_id: This agent's identifier
        own_answer: This agent's previous answer
        other_responses: List of other agents' responses with their IDs
        round_num: Current debate round number

    Returns:
        Formatted prompt string
    """
    other_text = ""
    for resp in other_responses:
        other_text += f"\n--- Agent {resp['agent_id']}'s response ---\n{resp['response']}\n"

    return f"""You are Agent {agent_id}. This is Round {round_num} of a mathematical debate.

Original Problem: {question}

Your previous answer:
{own_answer}

Other agents' answers:{other_text}

Instructions:
1. Review other agents' solutions carefully
2. If you find errors in your reasoning, correct them
3. If you find errors in others' reasoning, explain why
4. If you're confident in your answer and others have the same, confirm it
5. State your final answer clearly as "The answer is X"

Your response for Round {round_num}:"""


def save_metrics(metrics: Dict[str, Any], filename: str,
                 metrics_dir: str = "results/metrics"):
    """
    Save experiment metrics to JSON file.

    Args:
        metrics: Dictionary of metrics to save
        filename: Name for the metrics file
        metrics_dir: Directory to save metrics
    """
    os.makedirs(metrics_dir, exist_ok=True)
    filepath = os.path.join(metrics_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to {filepath}")


def load_metrics(filename: str, metrics_dir: str = "results/metrics") -> Dict[str, Any]:
    """
    Load metrics from JSON file.

    Args:
        filename: Name of the metrics file
        metrics_dir: Directory containing metrics

    Returns:
        Dictionary of metrics
    """
    filepath = os.path.join(metrics_dir, filename)

    with open(filepath, 'r') as f:
        return json.load(f)
