"""
DebateAgent class for multi-agent debate system.
Each agent uses a different LLM model via Ollama.
"""

import ollama
from typing import List, Dict, Optional, Any
import time

from utils import (
    extract_answer,
    format_prompt_initial,
    format_prompt_debate
)


class DebateAgent:
    """
    An agent that participates in multi-agent debates.
    Uses Ollama for LLM inference.
    """

    def __init__(self, agent_id: str, model_name: str, temperature: float = 0.7):
        """
        Initialize a debate agent.

        Args:
            agent_id: Unique identifier for this agent (e.g., "A", "B", "C")
            model_name: Ollama model to use (e.g., "llama3.1:8b")
            temperature: Sampling temperature for diversity
        """
        self.agent_id = agent_id
        self.model_name = model_name
        self.temperature = temperature
        self.history: List[Dict[str, Any]] = []

    def _generate(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate response using Ollama.

        Args:
            prompt: Input prompt
            max_retries: Number of retries on failure

        Returns:
            Generated text response
        """
        for attempt in range(max_retries):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        "temperature": self.temperature,
                        "num_predict": 1024,  # Max tokens to generate
                    }
                )
                return response['response']
            except Exception as e:
                print(f"Agent {self.agent_id} ({self.model_name}) error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                else:
                    raise RuntimeError(f"Failed to generate response after {max_retries} attempts: {e}")

    def initial_answer(self, question: str) -> Dict[str, Any]:
        """
        Generate initial answer for a math problem.

        Args:
            question: The math problem to solve

        Returns:
            Dictionary with response text and extracted answer
        """
        prompt = format_prompt_initial(question, self.agent_id)
        response = self._generate(prompt)

        # Extract numerical answer
        extracted = extract_answer(response)

        # Store in history
        result = {
            "round": 0,
            "prompt": prompt,
            "response": response,
            "extracted_answer": extracted
        }
        self.history.append(result)

        return result

    def debate_round(self, question: str, other_responses: List[Dict[str, str]],
                     round_num: int) -> Dict[str, Any]:
        """
        Generate response for a debate round, considering other agents' answers.

        Args:
            question: Original math problem
            other_responses: List of dicts with 'agent_id' and 'response' from other agents
            round_num: Current round number (1-indexed)

        Returns:
            Dictionary with response text and extracted answer
        """
        # Get own previous response
        own_previous = self.history[-1]["response"] if self.history else ""

        prompt = format_prompt_debate(
            question=question,
            agent_id=self.agent_id,
            own_answer=own_previous,
            other_responses=other_responses,
            round_num=round_num
        )

        response = self._generate(prompt)

        # Extract numerical answer
        extracted = extract_answer(response)

        # Store in history
        result = {
            "round": round_num,
            "prompt": prompt,
            "response": response,
            "extracted_answer": extracted,
            "other_responses": other_responses
        }
        self.history.append(result)

        return result

    def get_final_answer(self) -> Optional[float]:
        """
        Get the agent's final answer (from most recent response).

        Returns:
            Extracted numerical answer or None
        """
        if not self.history:
            return None
        return self.history[-1].get("extracted_answer")

    def get_answer_history(self) -> List[Optional[float]]:
        """
        Get history of extracted answers across all rounds.

        Returns:
            List of answers for each round
        """
        return [h.get("extracted_answer") for h in self.history]

    def changed_answer(self) -> bool:
        """
        Check if the agent changed their answer during debate.

        Returns:
            True if final answer differs from initial answer
        """
        answers = self.get_answer_history()
        if len(answers) < 2:
            return False

        initial = answers[0]
        final = answers[-1]

        if initial is None or final is None:
            return True  # Consider None as a change

        return initial != final

    def reset(self):
        """Clear agent's history for a new problem."""
        self.history = []


def check_models_available(models: List[str]) -> Dict[str, bool]:
    """
    Check which models are available in Ollama.

    Args:
        models: List of model names to check

    Returns:
        Dictionary mapping model name to availability
    """
    try:
        available = ollama.list()

        # Handle different ollama library versions
        # Newer versions return objects with 'model' attribute
        # Older versions return dicts with 'name' key
        available_names = []
        models_list = available.get('models', []) if isinstance(available, dict) else getattr(available, 'models', [])

        for m in models_list:
            if isinstance(m, dict):
                # Old format: {'name': 'model:tag', ...}
                available_names.append(m.get('name', ''))
            else:
                # New format: Model object with 'model' attribute
                available_names.append(getattr(m, 'model', getattr(m, 'name', '')))

        result = {}
        for model in models:
            # Check if model or model with tag is available
            result[model] = any(
                model in name or name.startswith(model.split(':')[0])
                for name in available_names
            )
        return result
    except Exception as e:
        print(f"Error checking models: {e}")
        return {model: False for model in models}


def pull_model_if_needed(model_name: str) -> bool:
    """
    Pull a model if it's not available locally.

    Args:
        model_name: Name of the model to pull

    Returns:
        True if model is now available
    """
    available = check_models_available([model_name])

    if available.get(model_name, False):
        print(f"Model {model_name} is already available")
        return True

    print(f"Pulling model {model_name}...")
    try:
        ollama.pull(model_name)
        print(f"Successfully pulled {model_name}")
        return True
    except Exception as e:
        print(f"Failed to pull {model_name}: {e}")
        return False


class MultiAgentDebate:
    """
    Orchestrates multi-agent debate for solving problems.
    """

    def __init__(self, models: List[str] = None, temperature: float = 0.7):
        """
        Initialize the debate system.

        Args:
            models: List of model names for agents. Default uses 3 different models.
            temperature: Sampling temperature for all agents
        """
        if models is None:
            models = ["llama3.1:8b", "qwen2.5:7b", "phi3:medium"]

        self.models = models
        self.temperature = temperature
        self.agents: List[DebateAgent] = []

        # Create agents
        agent_ids = ["A", "B", "C", "D", "E"][:len(models)]
        for agent_id, model in zip(agent_ids, models):
            self.agents.append(DebateAgent(agent_id, model, temperature))

    def debate(self, question: str, n_rounds: int = 2,
               verbose: bool = True) -> Dict[str, Any]:
        """
        Run a multi-agent debate on a question.

        Args:
            question: The math problem to solve
            n_rounds: Number of debate rounds (including initial)
            verbose: Whether to print progress

        Returns:
            Dictionary with debate results
        """
        # Reset all agents
        for agent in self.agents:
            agent.reset()

        rounds_data = []

        # Round 0: Initial answers
        if verbose:
            print(f"\n{'='*60}")
            print("ROUND 0: Initial Answers")
            print('='*60)

        round_0_data = {}
        for agent in self.agents:
            result = agent.initial_answer(question)
            round_0_data[agent.agent_id] = {
                "model": agent.model_name,
                "response": result["response"],
                "answer": result["extracted_answer"]
            }
            if verbose:
                print(f"\nAgent {agent.agent_id} ({agent.model_name}):")
                print(f"Answer: {result['extracted_answer']}")
                print(f"Response preview: {result['response'][:200]}...")

        rounds_data.append(round_0_data)

        # Debate rounds
        for round_num in range(1, n_rounds):
            if verbose:
                print(f"\n{'='*60}")
                print(f"ROUND {round_num}: Debate")
                print('='*60)

            round_data = {}

            for agent in self.agents:
                # Gather other agents' responses
                other_responses = []
                for other_agent in self.agents:
                    if other_agent.agent_id != agent.agent_id:
                        other_responses.append({
                            "agent_id": other_agent.agent_id,
                            "response": other_agent.history[-1]["response"]
                        })

                result = agent.debate_round(question, other_responses, round_num)
                round_data[agent.agent_id] = {
                    "model": agent.model_name,
                    "response": result["response"],
                    "answer": result["extracted_answer"]
                }

                if verbose:
                    print(f"\nAgent {agent.agent_id} ({agent.model_name}):")
                    print(f"Answer: {result['extracted_answer']}")
                    changed = "CHANGED" if agent.changed_answer() else "unchanged"
                    print(f"Status: {changed}")

            rounds_data.append(round_data)

        # Collect final answers and determine consensus
        final_answers = [agent.get_final_answer() for agent in self.agents]

        from utils import majority_vote
        consensus = majority_vote(final_answers)

        # Track which agents changed their answers
        changes = {
            agent.agent_id: agent.changed_answer()
            for agent in self.agents
        }

        result = {
            "question": question,
            "rounds": rounds_data,
            "final_answers": {
                agent.agent_id: agent.get_final_answer()
                for agent in self.agents
            },
            "consensus": consensus,
            "answer_changes": changes,
            "n_agents": len(self.agents),
            "n_rounds": n_rounds
        }

        if verbose:
            print(f"\n{'='*60}")
            print("FINAL RESULTS")
            print('='*60)
            print(f"Final answers: {result['final_answers']}")
            print(f"Consensus answer: {consensus}")
            print(f"Agents that changed answer: {[k for k, v in changes.items() if v]}")

        return result


def single_agent_solve(question: str, model: str = "llama3.1:8b",
                       temperature: float = 0.7, verbose: bool = False) -> Dict[str, Any]:
    """
    Solve a problem with a single agent (baseline).

    Args:
        question: Math problem to solve
        model: Ollama model to use
        temperature: Sampling temperature
        verbose: Whether to print output

    Returns:
        Dictionary with response and extracted answer
    """
    agent = DebateAgent("single", model, temperature)
    result = agent.initial_answer(question)

    if verbose:
        print(f"\nSingle Agent ({model}):")
        print(f"Answer: {result['extracted_answer']}")
        print(f"Response: {result['response'][:300]}...")

    return {
        "question": question,
        "model": model,
        "response": result["response"],
        "answer": result["extracted_answer"]
    }
