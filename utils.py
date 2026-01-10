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


def extract_confidence(text: str) -> str:
    """
    Extract confidence level from model response.

    Args:
        text: Model response text

    Returns:
        Confidence level: "HIGH", "MEDIUM", or "LOW". Defaults to "MEDIUM" if not found.
    """
    if not text:
        return "MEDIUM"

    text_upper = text.upper()

    # Look for explicit confidence statements
    patterns = [
        r'CONFIDENCE:\s*(HIGH|MEDIUM|LOW)',
        r'CONFIDENCE\s*(?:LEVEL)?(?:IS)?:?\s*(HIGH|MEDIUM|LOW)',
        r'(?:I\s+AM\s+)?(HIGH|MEDIUM|LOW)(?:LY)?\s+CONFIDENT',
        r'(HIGH|MEDIUM|LOW)\s+CONFIDENCE',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_upper)
        if match:
            return match.group(1)

    # Heuristic: Look for confidence indicators in text
    high_indicators = ['certain', 'sure', 'definitely', 'clearly', 'obvious', 'straightforward']
    low_indicators = ['unsure', 'uncertain', 'not sure', 'might be', 'could be', 'possibly', 'maybe']

    text_lower = text.lower()

    high_count = sum(1 for indicator in high_indicators if indicator in text_lower)
    low_count = sum(1 for indicator in low_indicators if indicator in text_lower)

    if high_count > low_count and high_count >= 2:
        return "HIGH"
    elif low_count > high_count and low_count >= 1:
        return "LOW"

    return "MEDIUM"


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


def weighted_vote(answers: List[Optional[float]], confidences: List[str],
                  responses: List[str] = None) -> Dict[str, Any]:
    """
    Determine consensus answer using confidence-weighted voting.

    Args:
        answers: List of numerical answers from agents
        confidences: List of confidence levels ("HIGH", "MEDIUM", "LOW")
        responses: Optional list of response texts for tie-breaking analysis

    Returns:
        Dictionary with:
            - 'answer': The winning answer
            - 'method': How the answer was determined
            - 'vote_details': Breakdown of votes and weights
            - 'is_tie': Whether there was a tie
    """
    # Confidence weights
    CONFIDENCE_WEIGHTS = {
        "HIGH": 3.0,
        "MEDIUM": 2.0,
        "LOW": 1.0
    }

    # Filter out None values and pair with confidences
    valid_pairs = [(a, c) for a, c in zip(answers, confidences) if a is not None]

    if not valid_pairs:
        return {
            'answer': None,
            'method': 'no_valid_answers',
            'vote_details': {},
            'is_tie': False
        }

    # Calculate weighted votes for each unique answer
    weighted_counts = {}
    raw_counts = {}

    for answer, confidence in valid_pairs:
        weight = CONFIDENCE_WEIGHTS.get(confidence, 2.0)
        if answer not in weighted_counts:
            weighted_counts[answer] = 0.0
            raw_counts[answer] = 0
        weighted_counts[answer] += weight
        raw_counts[answer] += 1

    # Sort by weighted count (descending)
    sorted_answers = sorted(weighted_counts.items(), key=lambda x: (-x[1], -raw_counts[x[0]]))

    vote_details = {
        answer: {
            'weighted_score': weighted_counts[answer],
            'raw_count': raw_counts[answer]
        }
        for answer in weighted_counts
    }

    # Check for tie (top two have same weighted score)
    is_tie = len(sorted_answers) > 1 and sorted_answers[0][1] == sorted_answers[1][1]

    if is_tie:
        # Tie-breaking: prefer answer with more raw votes
        top_weight = sorted_answers[0][1]
        tied_answers = [a for a, w in sorted_answers if w == top_weight]

        # Break by raw count
        tied_answers_sorted = sorted(tied_answers, key=lambda a: -raw_counts[a])

        if raw_counts[tied_answers_sorted[0]] > raw_counts[tied_answers_sorted[1]]:
            return {
                'answer': tied_answers_sorted[0],
                'method': 'tie_broken_by_raw_count',
                'vote_details': vote_details,
                'is_tie': True
            }

        # Still tied - will need judge model (handled in debate_agent.py)
        return {
            'answer': tied_answers_sorted[0],  # Return first for now
            'method': 'unresolved_tie',
            'vote_details': vote_details,
            'is_tie': True,
            'tied_answers': tied_answers
        }

    return {
        'answer': sorted_answers[0][0],
        'method': 'weighted_majority',
        'vote_details': vote_details,
        'is_tie': False
    }


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
1. Read the problem carefully and identify all given information
2. Break down the problem into clear steps
3. Show each calculation explicitly
4. After solving, VERIFY your answer by:
   - Re-reading the problem to ensure you answered what was asked
   - Checking each arithmetic operation
   - Making sure units and quantities make sense
5. State your confidence level (HIGH, MEDIUM, or LOW) based on how certain you are
6. State your final answer clearly as "The answer is X" followed by "Confidence: [LEVEL]"

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
1. CRITICALLY analyze each agent's solution (including your own):
   - Identify the specific step where errors may have occurred
   - Check if the problem was interpreted correctly
   - Verify all arithmetic calculations
   - Ensure the final answer addresses what was asked

2. For EACH solution that differs from yours:
   - Point out the specific line/step that seems incorrect
   - Explain WHY it is incorrect with a counter-calculation if needed

3. If you find an error in YOUR OWN reasoning:
   - Acknowledge the mistake explicitly
   - Show the corrected calculation
   - Update your answer

4. If multiple agents agree on an answer different from yours:
   - Carefully reconsider - they may have found something you missed
   - Only change if you find a genuine error in your work

5. State your confidence level (HIGH, MEDIUM, or LOW)
6. State your final answer clearly as "The answer is X" followed by "Confidence: [LEVEL]"

Your response for Round {round_num}:"""


def format_prompt_verification(question: str, proposed_answer: float,
                               agent_solutions: List[Dict[str, str]]) -> str:
    """
    Format prompt for verification round.

    Args:
        question: Original math problem
        proposed_answer: The consensus answer to verify
        agent_solutions: List of agent solutions with their reasoning

    Returns:
        Formatted prompt string
    """
    solutions_text = ""
    for i, sol in enumerate(agent_solutions):
        solutions_text += f"\n--- Solution {i+1} ---\n{sol.get('response', '')}\n"

    return f"""You are a mathematical verification expert. Your job is to verify if the proposed answer is correct.

Original Problem: {question}

Proposed Answer: {proposed_answer}

Solutions from different agents:{solutions_text}

Your task:
1. Solve the problem independently from scratch
2. Show your complete step-by-step solution
3. Compare your answer with the proposed answer of {proposed_answer}
4. If your answer MATCHES the proposed answer: respond with "VERIFIED: The answer {proposed_answer} is correct"
5. If your answer DIFFERS: respond with "REJECTED: The correct answer is X" (where X is your answer)

Important: Be thorough and check each calculation carefully. The proposed answer may be wrong.

Your verification:"""


def format_prompt_tiebreaker(question: str, tied_answers: List[float],
                             agent_solutions: List[Dict[str, Any]]) -> str:
    """
    Format prompt for tie-breaking when agents disagree.

    Args:
        question: Original math problem
        tied_answers: List of answers that are tied
        agent_solutions: List of solutions with agent_id and response

    Returns:
        Formatted prompt string
    """
    solutions_text = ""
    for sol in agent_solutions:
        agent_id = sol.get('agent_id', 'Unknown')
        response = sol.get('response', '')
        answer = sol.get('answer', 'N/A')
        solutions_text += f"\n--- Agent {agent_id} (Answer: {answer}) ---\n{response}\n"

    answers_str = ", ".join(str(a) for a in tied_answers)

    return f"""You are a mathematical judge. Multiple agents have different answers and you must determine the correct one.

Original Problem: {question}

Tied Answers: {answers_str}

Agent Solutions:{solutions_text}

Your task:
1. Carefully analyze EACH agent's solution step by step
2. Identify any arithmetic errors, misinterpretations, or logical mistakes
3. Solve the problem yourself to verify
4. Determine which answer (if any) is correct

For each agent's solution, state:
- Whether their interpretation of the problem is correct
- Whether each calculation step is correct
- The specific error (if any)

Finally, state: "The correct answer is X"

Your judgment:"""


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
