"""Builder for 2nd and 3rd order transition matrices"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from .model_2order import HigherOrderMarkovChain


class HigherOrderMatrixBuilder:
    """Creates higher order transition matrices from weather data"""

    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]

    MONTH_MAP = {
        1: "january", 2: "february", 3: "march", 4: "april",
        5: "may", 6: "june", 7: "july", 8: "august",
        9: "september", 10: "october", 11: "november", 12: "december"
    }

    def __init__(self, region: str, order: int = 2, output_dir: str = "data/matrices"):
        """
        Initializes the higher-order matrix builder

        Args:
            region: Name of the region
            order: Order of the Markov chain (2 or 3)
            output_dir: Output directory for matrices
        """
        self.region = region
        self.order = order
        order_str = f"order{order}"
        self.output_dir = Path(output_dir) / order_str / region
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create state combinations based on order
        self.states = HigherOrderMarkovChain.STATES
        self.state_to_idx = HigherOrderMarkovChain.STATE_TO_INDEX
        self._build_state_tuples()

    def _build_state_tuples(self):
        """Creates all possible state tuples based on order"""
        if self.order == 2:
            # (prev_state, current_state)
            self.state_tuples = [
                (s1, s2) for s1 in self.states for s2 in self.states
            ]
        elif self.order == 3:
            # (prev_prev_state, prev_state, current_state)
            self.state_tuples = [
                (s1, s2, s3) for s1 in self.states for s2 in self.states for s3 in self.states
            ]
        else:
            raise ValueError(f"Unsupported order: {self.order}")

    def build_transition_matrix(self, weather_data: pd.DataFrame, month: int) -> np.ndarray:
        """
        Creates a higher order transition matrix from weather data for a month

        Args:
            weather_data: DataFrame with 'date' and 'weather_state' columns
            month: Month number (1-12)

        Returns:
            NxN transition matrix (normalized), where N = 5^order
        """
        # Filter data for the given month
        df = weather_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        month_data = df[df['month'] == month].copy()
        month_data = month_data.sort_values('date').reset_index(drop=True)

        # Initialize transition count matrix
        n_tuples = len(self.state_tuples)
        transition_counts = np.zeros((n_tuples, len(self.states)))

        if self.order == 2:
            # Count (prev_state, current_state) → next_state transitions
            if len(month_data) < 3:
                return self._create_uniform_matrix()

            for i in range(len(month_data) - 2):
                prev_state = month_data.iloc[i]['weather_state']
                curr_state = month_data.iloc[i + 1]['weather_state']
                next_state = month_data.iloc[i + 2]['weather_state']

                # Find index of tuple (prev, curr)
                tuple_idx = self.state_tuples.index((prev_state, curr_state))
                next_idx = self.state_to_idx[next_state]

                transition_counts[tuple_idx, next_idx] += 1

        elif self.order == 3:
            # Count (prev2, prev, curr) → next transitions
            if len(month_data) < 4:
                return self._create_uniform_matrix()

            for i in range(len(month_data) - 3):
                prev_prev_state = month_data.iloc[i]['weather_state']
                prev_state = month_data.iloc[i + 1]['weather_state']
                curr_state = month_data.iloc[i + 2]['weather_state']
                next_state = month_data.iloc[i + 3]['weather_state']

                tuple_idx = self.state_tuples.index((prev_prev_state, prev_state, curr_state))
                next_idx = self.state_to_idx[next_state]

                transition_counts[tuple_idx, next_idx] += 1

        # Normalize to probabilities
        transition_matrix = self._normalize_matrix(transition_counts)

        return transition_matrix

    def _normalize_matrix(self, counts_matrix: np.ndarray, smoothing: float = 1e-6) -> np.ndarray:
        """
        Normalizes a count matrix to probabilities with Laplace smoothing

        Args:
            counts_matrix: Matrix with transition counts (shape: n_tuples x n_states)
            smoothing: Smoothing factor for zero probabilities

        Returns:
            Normalized matrix (row sum = 1)
        """
        smoothed_matrix = counts_matrix + smoothing
        row_sums = smoothed_matrix.sum(axis=1, keepdims=True)
        prob_matrix = smoothed_matrix / row_sums

        return prob_matrix

    def _create_uniform_matrix(self) -> np.ndarray:
        """Creates a uniform transition matrix (for insufficient data)"""
        n_tuples = len(self.state_tuples)
        n_states = len(self.states)
        return np.ones((n_tuples, n_states)) / n_states

    def save_matrix_to_json(self, transition_matrix: np.ndarray, month: str):
        """
        Saves a transition matrix as JSON file with rounding

        Args:
            transition_matrix: The matrix to save
            month: Month name
        """
        output_file = self.output_dir / f"{month}.json"

        # Round to 2 decimal places
        rounded_matrix = np.round(transition_matrix, 2)

        # Correct row sums
        for i in range(len(rounded_matrix)):
            row_sum = rounded_matrix[i].sum()
            if not np.isclose(row_sum, 1.0, atol=1e-6):
                max_idx = np.argmax(rounded_matrix[i])
                rounded_matrix[i, max_idx] += (1.0 - row_sum)

        rounded_matrix_list = rounded_matrix.tolist()

        # Create informative description of state tuples
        if self.order == 2:
            state_pairs_str = [f"{s1}→{s2}" for s1, s2 in self.state_tuples]
        else:
            state_pairs_str = [f"{s1}→{s2}→{s3}" for s1, s2, s3 in self.state_tuples]

        data = {
            "region": self.region,
            "month": month,
            "order": self.order,
            "states": self.states,
            "state_tuples": state_pairs_str,
            "matrix": rounded_matrix_list
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def build_all_months(self, weather_data: pd.DataFrame):
        """
        Creates matrices for all 12 months

        Args:
            weather_data: DataFrame with weather data
        """
        for month_num in range(1, 13):
            month_name = self.MONTH_MAP[month_num]
            matrix = self.build_transition_matrix(weather_data, month_num)
            self.save_matrix_to_json(matrix, month_name)
            print(f"✓ {month_name:10} - Matrix order {self.order} created ({len(self.state_tuples)}×{len(self.states)} matrix)")

    def build_from_file(self, csv_file: str):
        """
        Creates all matrices from a CSV file

        Args:
            csv_file: Path to CSV file with weather data
        """
        df = pd.read_csv(csv_file)
        self.build_all_months(df)

    def print_matrix_info(self):
        """Prints information about matrix size"""
        print(f"\nMatrix information (order {self.order}):")
        print(f"   States: {len(self.states)}")
        print(f"   State tuples: {len(self.state_tuples)}")
        print(f"   Matrix size: {len(self.state_tuples)}×{len(self.states)}")
        print(f"   Region: {self.region}")
