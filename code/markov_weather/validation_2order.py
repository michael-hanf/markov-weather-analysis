"""Validation for higher order Markov models"""

import pandas as pd
import numpy as np
from pathlib import Path
from .model_2order import HigherOrderMarkovChain


class HigherOrderValidationAnalyzer:
    """Validates higher order Markov models against real weather data"""

    def __init__(self, region: str, data_file: str, order: int = 2):
        """
        Initializes the validator

        Args:
            region: Name of the region
            data_file: Path to CSV file with real data
            order: Order of the Markov chain (2 or 3)
        """
        self.region = region
        self.order = order
        self.data = pd.read_csv(data_file)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['month'] = self.data['date'].dt.month
        self.states = HigherOrderMarkovChain.STATES

    def test_next_day_prediction(self) -> float:
        """
        Tests 1-day prediction accuracy (2nd or 3rd order)

        Returns:
            Accuracy as percentage (0-100)
        """
        correct = 0
        total = 0

        for month_num in range(1, 13):
            month_data = self.data[self.data['month'] == month_num].copy()
            month_data = month_data.sort_values('date').reset_index(drop=True)

            min_length = self.order + 1
            if len(month_data) < min_length:
                continue

            try:
                model = HigherOrderMarkovChain(self.region, self._month_name(month_num), order=self.order)
            except FileNotFoundError:
                continue

            # Test for 2nd order
            if self.order == 2:
                for i in range(len(month_data) - 2):
                    prev_state = month_data.iloc[i]['weather_state']
                    curr_state = month_data.iloc[i + 1]['weather_state']
                    actual_next = month_data.iloc[i + 2]['weather_state']

                    try:
                        probs = model.predict_next_day(prev_state, curr_state)
                        predicted_next = max(probs.items(), key=lambda x: x[1])[0]

                        if predicted_next == actual_next:
                            correct += 1
                        total += 1
                    except:
                        continue

            # Test for 3rd order
            elif self.order == 3:
                for i in range(len(month_data) - 3):
                    prev_prev_state = month_data.iloc[i]['weather_state']
                    prev_state = month_data.iloc[i + 1]['weather_state']
                    curr_state = month_data.iloc[i + 2]['weather_state']
                    actual_next = month_data.iloc[i + 3]['weather_state']

                    try:
                        probs = model.predict_next_day_3order(prev_prev_state, prev_state, curr_state)
                        predicted_next = max(probs.items(), key=lambda x: x[1])[0]

                        if predicted_next == actual_next:
                            correct += 1
                        total += 1
                    except:
                        continue

        if total == 0:
            return 0.0

        accuracy = (correct / total) * 100
        return accuracy

    def test_sequence_accuracy(self, seq_length: int = 3) -> float:
        """
        Tests the accuracy of longer sequences

        Args:
            seq_length: Length of the sequence to test

        Returns:
            Accuracy as percentage (0-100)
        """
        correct = 0
        total = 0

        for month_num in range(1, 13):
            month_data = self.data[self.data['month'] == month_num].copy()
            month_data = month_data.sort_values('date').reset_index(drop=True)

            if len(month_data) < seq_length + self.order:
                continue

            try:
                model = HigherOrderMarkovChain(self.region, self._month_name(month_num), order=self.order)
            except FileNotFoundError:
                continue

            if self.order == 2:
                for i in range(len(month_data) - seq_length - 1):
                    prev_state = month_data.iloc[i]['weather_state']
                    curr_state = month_data.iloc[i + 1]['weather_state']

                    # Prediction for seq_length days
                    predicted_seq = []
                    p, c = prev_state, curr_state

                    for j in range(seq_length):
                        probs = model.predict_next_day(p, c)
                        next_state = max(probs.items(), key=lambda x: x[1])[0]
                        predicted_seq.append(next_state)
                        p, c = c, next_state

                    # Comparison with real data
                    actual_seq = [
                        month_data.iloc[i + j + 2]['weather_state']
                        for j in range(seq_length)
                    ]

                    if predicted_seq == actual_seq:
                        correct += 1
                    total += 1

            elif self.order == 3:
                for i in range(len(month_data) - seq_length - 2):
                    pp_state = month_data.iloc[i]['weather_state']
                    p_state = month_data.iloc[i + 1]['weather_state']
                    c_state = month_data.iloc[i + 2]['weather_state']

                    # Prediction for seq_length days
                    predicted_seq = []
                    pp, p, c = pp_state, p_state, c_state

                    for j in range(seq_length):
                        probs = model.predict_next_day_3order(pp, p, c)
                        next_state = max(probs.items(), key=lambda x: x[1])[0]
                        predicted_seq.append(next_state)
                        pp, p, c = p, c, next_state

                    # Comparison with real data
                    actual_seq = [
                        month_data.iloc[i + j + 3]['weather_state']
                        for j in range(seq_length)
                    ]

                    if predicted_seq == actual_seq:
                        correct += 1
                    total += 1

        if total == 0:
            return 0.0

        accuracy = (correct / total) * 100
        return accuracy

    def compare_state_distribution(self) -> dict:
        """
        Compares the real state distribution with simulated

        Returns:
            Dictionary with MAE and other metrics
        """
        results = {}

        for month_num in range(1, 13):
            month_data = self.data[self.data['month'] == month_num]
            month_name = self._month_name(month_num)

            # Real distribution
            real_counts = month_data['weather_state'].value_counts()
            real_dist = {state: (real_counts.get(state, 0) / len(month_data) * 100) for state in self.states}

            # Simulated distribution (from the matrices)
            # For 2nd order: start with average state pair
            sim_counts = {state: 0 for state in self.states}

            # Simple heuristic: use the diagonal of the 2nd order matrix
            try:
                model = HigherOrderMarkovChain(self.region, month_name, order=self.order)

                # Average over all state pairs
                for prev_state in self.states:
                    for curr_state in self.states:
                        probs = model.predict_next_day(prev_state, curr_state)
                        for state, prob in probs.items():
                            sim_counts[state] += prob / (len(self.states) ** 2)

            except FileNotFoundError:
                continue

            sim_dist = {state: (sim_counts[state] * 100) for state in self.states}

            # Calculate MAE
            mae = sum(abs(real_dist[state] - sim_dist[state]) for state in self.states) / len(self.states)

            results[month_name] = {
                'real': real_dist,
                'simulated': sim_dist,
                'mae': mae
            }

        return results

    def get_confidence_score(self) -> float:
        """
        Calculates a confidence score for the model (0-100)

        Returns:
            Confidence score
        """
        accuracy = self.test_next_day_prediction()

        # Normalize to 0-100 confidence score
        # 50% accuracy = 0% confidence, 100% accuracy = 100% confidence
        confidence = max(0, (accuracy - 50) * 2)

        return min(100, confidence)

    def _month_name(self, month_num: int) -> str:
        """Converts month number to name"""
        months = [
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        return months[month_num - 1]

    def print_validation_report(self):
        """Prints a detailed validation report"""
        print(f"\n{'=' * 100}")
        print(f"VALIDATION REPORT: {self.order}. Order Markov model for {self.region}")
        print(f"{'=' * 100}\n")

        # 1-day accuracy
        accuracy_1day = self.test_next_day_prediction()
        print(f"1-day prediction accuracy: {accuracy_1day:.1f}%")

        # Sequence accuracy
        accuracy_3day = self.test_sequence_accuracy(seq_length=3)
        print(f"3-day prediction accuracy: {accuracy_3day:.1f}%")

        # State distribution
        print(f"\nState frequency comparison (MAE per month):")
        dist_results = self.compare_state_distribution()
        overall_mae = 0
        for month_name, metrics in sorted(dist_results.items()):
            mae = metrics['mae']
            overall_mae += mae
            print(f"  {month_name:10} MAE: {mae:.2f}%")

        overall_mae /= 12
        print(f"\nOverall MAE: {overall_mae:.2f}%")

        # Confidence score
        confidence = self.get_confidence_score()
        print(f"\nConfidence score: {confidence:.1f}%")

        print(f"\n{'=' * 100}\n")
