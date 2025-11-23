"""Markov chain model for weather forecasting"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class WeatherMarkovChain:
    """Markov chain model for weather states"""

    # Weather states (index-based)
    STATES = ["sunny", "partly_cloudy", "cloudy", "rainy", "stormy"]
    STATE_TO_INDEX = {state: idx for idx, state in enumerate(STATES)}
    INDEX_TO_STATE = {idx: state for idx, state in enumerate(STATES)}

    def __init__(self, region: str, month: str):
        """
        Initializes the Markov chain for a region and a month

        Args:
            region: Name of the region (e.g. 'schleswig_holstein')
            month: Month name (e.g. 'january')
        """
        self.region = region
        self.month = month
        self.transition_matrix = None
        self._load_matrix()

    def _load_matrix(self):
        """Loads the transition matrix from JSON file"""
        matrix_path = Path(__file__).parent.parent / "data" / "matrices" / self.region / f"{self.month}.json"

        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix not found: {matrix_path}")

        with open(matrix_path, 'r') as f:
            data = json.load(f)

        self.transition_matrix = np.array(data['matrix'], dtype=np.float64)

    def predict_next_day(self, current_state: str) -> Dict[str, float]:
        """
        Predicts probabilities for the next day

        Args:
            current_state: Current weather state (string)

        Returns:
            Dictionary with probabilities for all states
        """
        if current_state not in self.STATE_TO_INDEX:
            raise ValueError(f"Invalid state: {current_state}")

        state_idx = self.STATE_TO_INDEX[current_state]
        # Get the row of current state probabilities
        next_probs = self.transition_matrix[state_idx]

        return {state: float(prob) for state, prob in zip(self.STATES, next_probs)}

    def predict_n_days(self, current_state: str, n_days: int) -> List[Dict[str, float]]:
        """
        Predicts probabilities for N days

        Args:
            current_state: Current weather state
            n_days: Number of days for the forecast

        Returns:
            List of probability dicts for each day
        """
        state_idx = self.STATE_TO_INDEX[current_state]

        # Start with one-hot vector for current state
        current_vector = np.zeros(len(self.STATES))
        current_vector[state_idx] = 1.0

        predictions = []

        for day in range(n_days):
            # Multiply with transition matrix
            current_vector = current_vector @ self.transition_matrix

            # Convert to dictionary
            probs_dict = {state: float(prob) for state, prob in zip(self.STATES, current_vector)}
            predictions.append(probs_dict)

        return predictions

    def get_most_likely_state(self, probs: Dict[str, float]) -> Tuple[str, float]:
        """
        Determines the most likely state from probability distribution

        Args:
            probs: Dictionary with state probabilities

        Returns:
            Tuple of (state, probability)
        """
        state = max(probs.items(), key=lambda x: x[1])
        return state[0], state[1]

    @staticmethod
    def verify_matrix_validity(matrix: np.ndarray) -> bool:
        """
        Verifies that a transition matrix is valid
        (each row sums to 1)

        Args:
            matrix: Transition matrix

        Returns:
            True if valid
        """
        row_sums = matrix.sum(axis=1)
        # Tolerance for floating-point errors
        return np.allclose(row_sums, 1.0, atol=1e-6)
