"""2nd order Markov chain model for improved weather forecasting"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class HigherOrderMarkovChain:
    """2nd order Markov chain model - considers the last 2 days"""

    # Weather states (index-based)
    STATES = ["sunny", "partly_cloudy", "cloudy", "rainy", "stormy"]
    STATE_TO_INDEX = {state: idx for idx, state in enumerate(STATES)}
    INDEX_TO_STATE = {idx: state for idx, state in enumerate(STATES)}

    def __init__(self, region: str, month: str, order: int = 2):
        """
        Initializes the higher order Markov chain

        Args:
            region: Name of the region (e.g. 'schleswig_holstein')
            month: Month name (e.g. 'january')
            order: Order of the Markov chain (2 or 3)
        """
        self.region = region
        self.month = month
        self.order = order
        self.transition_matrix = None
        self.state_pairs = []  # List of all possible state pairs
        self._build_state_pairs()
        self._load_matrix()

    def _build_state_pairs(self):
        """Creates list of all possible state combinations for the order"""
        if self.order == 2:
            # 2nd order: (prev_state, current_state)
            self.state_pairs = [
                (s1, s2) for s1 in self.STATES for s2 in self.STATES
            ]
        elif self.order == 3:
            # 3rd order: (prev_prev_state, prev_state, current_state)
            self.state_pairs = [
                (s1, s2, s3) for s1 in self.STATES for s2 in self.STATES for s3 in self.STATES
            ]
        else:
            raise ValueError(f"Unsupported order: {self.order}. Use 2 or 3.")

    def _get_pair_index(self, states_tuple: Tuple) -> int:
        """Finds index of a state pair in the list"""
        return self.state_pairs.index(states_tuple)

    def _load_matrix(self):
        """Loads the transition matrix from JSON file"""
        # Use a subfolder for higher order
        order_str = f"order{self.order}"
        matrix_path = (
            Path(__file__).parent.parent / "data" / "matrices" / order_str /
            self.region / f"{self.month}.json"
        )

        if not matrix_path.exists():
            raise FileNotFoundError(f"Matrix not found: {matrix_path}")

        with open(matrix_path, 'r') as f:
            data = json.load(f)

        self.transition_matrix = np.array(data['matrix'], dtype=np.float64)

    def predict_next_day(self, prev_state: str, current_state: str) -> Dict[str, float]:
        """
        Predicts probabilities for the next day (2nd order)

        Args:
            prev_state: Weather state from previous day
            current_state: Current weather state

        Returns:
            Dictionary with probabilities for all states
        """
        if self.order != 2:
            raise ValueError("This method is only available for 2nd order")

        if current_state not in self.STATE_TO_INDEX or prev_state not in self.STATE_TO_INDEX:
            raise ValueError(f"Invalid state: ({prev_state}, {current_state})")

        # Find index of the state pair
        pair_idx = self._get_pair_index((prev_state, current_state))

        # Get the row of transition probabilities
        next_probs = self.transition_matrix[pair_idx]

        return {state: float(prob) for state, prob in zip(self.STATES, next_probs)}

    def predict_next_day_3order(self, prev_prev_state: str, prev_state: str, current_state: str) -> Dict[str, float]:
        """
        Predicts probabilities for the next day (3rd order)

        Args:
            prev_prev_state: Weather state from 2 days ago
            prev_state: Weather state from previous day
            current_state: Current weather state

        Returns:
            Dictionary with probabilities for all states
        """
        if self.order != 3:
            raise ValueError("This method is only available for 3rd order")

        # Validation
        for state in [prev_prev_state, prev_state, current_state]:
            if state not in self.STATE_TO_INDEX:
                raise ValueError(f"Invalid state: {state}")

        # Find index of the state triplet
        pair_idx = self._get_pair_index((prev_prev_state, prev_state, current_state))

        # Get the row of transition probabilities
        next_probs = self.transition_matrix[pair_idx]

        return {state: float(prob) for state, prob in zip(self.STATES, next_probs)}

    def predict_n_days(self, prev_state: str, current_state: str, n_days: int) -> List[Dict[str, float]]:
        """
        Predicts probabilities for N days (2nd order)
        Simulates a sequence with stochastic transitions

        Args:
            prev_state: State from previous day
            current_state: Current state
            n_days: Number of days for the forecast

        Returns:
            List of probability dicts for each day
        """
        predictions = []
        prev = prev_state
        curr = current_state

        for _ in range(n_days):
            # Prediction for next day
            probs = self.predict_next_day(prev, curr)
            predictions.append(probs)

            # For next iteration: shift the states
            # Use the most likely next state
            next_state = max(probs.items(), key=lambda x: x[1])[0]
            prev = curr
            curr = next_state

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
        return np.allclose(row_sums, 1.0, atol=1e-6)
