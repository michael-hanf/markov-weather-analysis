"""Builder for transition matrices from weather data"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from .model import WeatherMarkovChain


class MatrixBuilder:
    """Creates transition matrices from weather data"""

    MONTHS = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]

    MONTH_MAP = {
        1: "january", 2: "february", 3: "march", 4: "april",
        5: "may", 6: "june", 7: "july", 8: "august",
        9: "september", 10: "october", 11: "november", 12: "december"
    }

    def __init__(self, region: str, output_dir: str = "data/matrices"):
        """
        Initializes the matrix builder

        Args:
            region: Name of the region
            output_dir: Output directory for matrices
        """
        self.region = region
        self.output_dir = Path(output_dir) / region
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_transition_matrix(self, weather_data: pd.DataFrame, month: int) -> np.ndarray:
        """
        Creates a transition matrix from weather data for a month

        Args:
            weather_data: DataFrame with 'date' and 'weather_state' columns
            month: Month number (1-12)

        Returns:
            5x5 transition matrix (normalized)
        """
        # Filter data for the given month
        df = weather_data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month

        month_data = df[df['month'] == month].copy()
        month_data = month_data.sort_values('date').reset_index(drop=True)

        if len(month_data) < 2:
            # Fallback: Uniform matrix
            return self._create_uniform_matrix()

        # Initialize transition count matrix
        states = WeatherMarkovChain.STATES
        state_to_idx = WeatherMarkovChain.STATE_TO_INDEX

        transition_counts = np.zeros((len(states), len(states)))

        # Count transitions (day i → day i+1)
        for i in range(len(month_data) - 1):
            current_state = month_data.iloc[i]['weather_state']
            next_state = month_data.iloc[i + 1]['weather_state']

            current_idx = state_to_idx[current_state]
            next_idx = state_to_idx[next_state]

            transition_counts[current_idx, next_idx] += 1

        # Normalize to probabilities
        transition_matrix = self._normalize_matrix(transition_counts)

        return transition_matrix

    def _normalize_matrix(self, counts_matrix: np.ndarray, smoothing: float = 1e-6) -> np.ndarray:
        """
        Normalizes a count matrix to probabilities with Laplace smoothing

        Args:
            counts_matrix: Matrix with transition counts
            smoothing: Smoothing factor for zero probabilities

        Returns:
            Normalized matrix (row sum = 1)
        """
        # Laplace smoothing: Add small constant to all cells
        # to avoid states being impossible
        smoothed_matrix = counts_matrix + smoothing

        row_sums = smoothed_matrix.sum(axis=1, keepdims=True)

        prob_matrix = smoothed_matrix / row_sums

        return prob_matrix

    def _create_uniform_matrix(self) -> np.ndarray:
        """Creates a uniform transition matrix (for insufficient data)"""
        n_states = len(WeatherMarkovChain.STATES)
        return np.ones((n_states, n_states)) / n_states

    def save_matrix_to_json(self, transition_matrix: np.ndarray, month: str):
        """
        Saves a transition matrix as JSON file with rounding

        Args:
            transition_matrix: The matrix to save
            month: Month name
        """
        output_file = self.output_dir / f"{month}.json"

        # Round to 2 decimal places for better readability
        # But increase the smallest probability to ensure row sum = 1.0
        rounded_matrix = np.round(transition_matrix, 2)

        # Correct row sums to ensure they are 1.0
        for i in range(len(rounded_matrix)):
            row_sum = rounded_matrix[i].sum()
            if not np.isclose(row_sum, 1.0, atol=1e-6):
                # Find index with max value and correct it
                max_idx = np.argmax(rounded_matrix[i])
                rounded_matrix[i, max_idx] += (1.0 - row_sum)

        rounded_matrix_list = rounded_matrix.tolist()

        data = {
            "region": self.region,
            "month": month,
            "states": WeatherMarkovChain.STATES,
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
            print(f"✓ Matrix for {month_name} created and saved")

    def build_from_file(self, csv_file: str):
        """
        Creates all matrices from a CSV file

        Args:
            csv_file: Path to CSV file with weather data
        """
        df = pd.read_csv(csv_file)
        self.build_all_months(df)
