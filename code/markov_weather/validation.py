"""Validation of the simulator model against real data"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from .simulator import WeatherSimulator
from .model import WeatherMarkovChain


class ValidationAnalyzer:
    """Validates the Markov model against real historical data"""

    def __init__(self, region: str, data_file: str):
        """
        Initializes the validator

        Args:
            region: Name of the region
            data_file: Path to CSV file with real weather data
        """
        self.region = region
        self.data_file = data_file
        self.data = self._load_data()
        self.simulator = WeatherSimulator(region)

    def _load_data(self) -> pd.DataFrame:
        """Loads real weather data"""
        df = pd.read_csv(self.data_file)
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['month_name'] = df['date'].dt.strftime('%B').str.lower()
        return df.sort_values('date').reset_index(drop=True)

    def compare_state_distribution(self) -> Dict[str, Dict]:
        """
        Compares frequency distribution: Real data vs. simulation

        Returns:
            {month: {state: {real: %, simulated: %}}}
        """
        comparison = {}

        for month_name in self.data['month_name'].unique():
            month_data = self.data[self.data['month_name'] == month_name]

            # Real data: frequencies
            real_counts = month_data['weather_state'].value_counts()
            real_probs = (real_counts / len(month_data) * 100).to_dict()

            # Simulation: 10,000x for this month
            sequences = self.simulator.simulate_many("cloudy", 30, month_name, 500)
            stats = WeatherSimulator.get_sequence_statistics(sequences)
            simulated_probs = {
                state: prob * 100
                for state, prob in stats['state_probabilities'].items()
            }

            # Fill in missing states
            for state in WeatherMarkovChain.STATES:
                if state not in real_probs:
                    real_probs[state] = 0.0
                if state not in simulated_probs:
                    simulated_probs[state] = 0.0

            comparison[month_name] = {
                state: {
                    'real': real_probs[state],
                    'simulated': simulated_probs[state],
                    'diff': abs(real_probs[state] - simulated_probs[state])
                }
                for state in WeatherMarkovChain.STATES
            }

        return comparison

    def calculate_mae(self, comparison: Dict) -> Tuple[float, Dict]:
        """
        Calculates Mean Absolute Error (MAE) per month

        Args:
            comparison: Comparison data

        Returns:
            (overall_mae, {month: mae})
        """
        month_errors = {}
        all_errors = []

        for month, states in comparison.items():
            month_errors_list = [v['diff'] for v in states.values()]
            mae = np.mean(month_errors_list)
            month_errors[month] = mae
            all_errors.extend(month_errors_list)

        overall_mae = np.mean(all_errors)
        return overall_mae, month_errors

    def compare_transitions(self) -> Dict[str, Dict]:
        """
        Compares transition patterns: Real data vs. simulation

        Returns:
            {transition: {real: %, simulated: %}}
        """
        # Count real transitions
        real_transitions = defaultdict(int)
        total_transitions = 0

        for month in self.data['month_name'].unique():
            month_data = self.data[self.data['month_name'] == month].sort_values('date')
            for i in range(len(month_data) - 1):
                from_state = month_data.iloc[i]['weather_state']
                to_state = month_data.iloc[i + 1]['weather_state']
                transition = f"{from_state} → {to_state}"
                real_transitions[transition] += 1
                total_transitions += 1

        real_probs = {t: (c / total_transitions * 100) for t, c in real_transitions.items()}

        # Count simulated transitions
        sim_transitions = defaultdict(int)
        sim_total = 0

        for month in self.data['month_name'].unique():
            sequences = self.simulator.simulate_many("cloudy", 30, month, 1000)
            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    transition = f"{sequence[i]} → {sequence[i + 1]}"
                    sim_transitions[transition] += 1
                    sim_total += 1

        sim_probs = {t: (c / sim_total * 100) for t, c in sim_transitions.items()}

        # Combine
        all_transitions = set(real_probs.keys()) | set(sim_probs.keys())
        comparison = {
            t: {
                'real': real_probs.get(t, 0),
                'simulated': sim_probs.get(t, 0),
                'diff': abs(real_probs.get(t, 0) - sim_probs.get(t, 0))
            }
            for t in all_transitions
        }

        return comparison

    def predict_vs_actual(self, days: int = 7) -> Dict[str, Dict]:
        """
        Compares predictions against actual next days

        Args:
            days: Number of days for forecast

        Returns:
            Accuracy metrics
        """
        results = {
            'correct_next_day': 0,
            'incorrect_next_day': 0,
            'accuracy_next_day': 0,
            'predictions_by_state': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'correct_sequences': 0,
            'total_sequences': 0,
            'sequence_accuracy': 0
        }

        for i in range(len(self.data) - days):
            current_state = self.data.iloc[i]['weather_state']
            month = self.data.iloc[i]['month_name']

            # Prediction for next day
            simulator = WeatherSimulator(self.region)
            next_day_probs = simulator.predictor.predict(current_state, 1, month)[0]
            predicted_state = max(next_day_probs.items(), key=lambda x: x[1])[0]

            # Actual next state
            actual_next = self.data.iloc[i + 1]['weather_state']

            # 1-day forecast
            if predicted_state == actual_next:
                results['correct_next_day'] += 1
            else:
                results['incorrect_next_day'] += 1

            results['predictions_by_state'][current_state]['total'] += 1
            if predicted_state == actual_next:
                results['predictions_by_state'][current_state]['correct'] += 1

            # N-day sequence comparison
            actual_sequence = list(self.data.iloc[i:i+days]['weather_state'])
            sim_sequence = simulator.simulate_sequence(current_state, days - 1, month)

            # Count correct predictions in the sequence
            correct_in_seq = sum(1 for a, s in zip(actual_sequence[1:], sim_sequence[1:]) if a == s)
            if correct_in_seq == days - 1:
                results['correct_sequences'] += 1

            results['total_sequences'] += 1

        # Calculate accuracy
        total_predictions = results['correct_next_day'] + results['incorrect_next_day']
        if total_predictions > 0:
            results['accuracy_next_day'] = (results['correct_next_day'] / total_predictions) * 100

        if results['total_sequences'] > 0:
            results['sequence_accuracy'] = (results['correct_sequences'] / results['total_sequences']) * 100

        # Accuracy per state
        for state, stats in results['predictions_by_state'].items():
            if stats['total'] > 0:
                stats['accuracy'] = (stats['correct'] / stats['total']) * 100

        return results

    def get_best_worst_transitions(self, comparison: Dict, n: int = 5) -> Tuple[List, List]:
        """
        Finds best/worst predicted transitions

        Args:
            comparison: Transition comparison
            n: Number of transitions

        Returns:
            (best, worst)
        """
        sorted_trans = sorted(comparison.items(), key=lambda x: x[1]['diff'])
        return sorted_trans[:n], sorted_trans[-n:]

    def get_confidence_score(self) -> float:
        """
        Calculates an overall confidence score of the model (0-100%)

        Returns:
            Confidence score
        """
        dist_comparison = self.compare_state_distribution()
        mae, _ = self.calculate_mae(dist_comparison)

        # Convert MAE to confidence (higher MAE = lower confidence)
        confidence = max(0, 100 - mae * 5)
        return confidence

    @staticmethod
    def print_comparison_table(
        comparison: Dict[str, Dict],
        title: str = "Comparison: Real Data vs. Simulation"
    ):
        """Prints a nice comparison table"""
        print(f"\n{title}")
        print("=" * 90)
        print(f"{'Month':<12} {'State':<15} {'Real Data':<15} {'Simulation':<15} {'Diff':<10}")
        print("-" * 90)

        for month in sorted(comparison.keys()):
            states = comparison[month]
            month_printed = False
            for state in WeatherMarkovChain.STATES:
                if state in states:
                    data = states[state]
                    month_str = month if not month_printed else ""
                    print(
                        f"{month_str:<12} {state:<15} "
                        f"{data['real']:>6.1f}%         "
                        f"{data['simulated']:>6.1f}%         "
                        f"{data['diff']:>6.1f}%"
                    )
                    month_printed = True

        print("=" * 90)
