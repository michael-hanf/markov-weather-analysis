"""Stochastic weather simulator based on Markov chains"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from .model import WeatherMarkovChain
from .predictor import Predictor


class WeatherSimulator:
    """Simulates realistic weather sequences based on transition matrices"""

    def __init__(self, region: str):
        """
        Initializes the simulator

        Args:
            region: Name of the region
        """
        self.region = region
        self.predictor = Predictor(region)

    def simulate_sequence(
        self,
        start_state: str,
        days: int,
        month: str,
        random_seed: Optional[int] = None
    ) -> List[str]:
        """
        Simulates a single weather sequence

        Args:
            start_state: Starting state
            days: Number of days to simulate
            month: Month for the matrix
            random_seed: Reproducibility seed

        Returns:
            List of weather states
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        chain = self.predictor._get_chain(month)
        states = chain.STATES

        sequence = [start_state]
        current_state = start_state

        for _ in range(days):
            # Get transition probabilities
            probs = chain.predict_next_day(current_state)
            probs_array = np.array([probs[s] for s in states])

            # Roll next state based on probabilities
            next_state = np.random.choice(states, p=probs_array)
            sequence.append(next_state)
            current_state = next_state

        return sequence

    def simulate_many(
        self,
        start_state: str,
        days: int,
        month: str,
        num_simulations: int = 1000
    ) -> List[List[str]]:
        """
        Simulates multiple weather sequences

        Args:
            start_state: Starting state
            days: Number of days per sequence
            month: Month for the matrix
            num_simulations: Number of simulations

        Returns:
            List of sequences
        """
        sequences = []
        for _ in range(num_simulations):
            seq = self.simulate_sequence(start_state, days, month)
            sequences.append(seq)
        return sequences

    @staticmethod
    def get_sequence_statistics(sequences: List[List[str]]) -> Dict:
        """
        Calculates statistics from simulated sequences

        Args:
            sequences: List of weather sequences

        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_sequences': len(sequences),
            'sequence_length': len(sequences[0]) if sequences else 0,
            'state_counts': defaultdict(int),
            'state_probabilities': {},
            'consecutive_runs': defaultdict(list),  # Longest runs per state
            'transition_counts': defaultdict(int),
        }

        for sequence in sequences:
            # Count states
            for state in sequence:
                stats['state_counts'][state] += 1

            # Count transitions
            for i in range(len(sequence) - 1):
                transition = f"{sequence[i]} → {sequence[i+1]}"
                stats['transition_counts'][transition] += 1

            # Count longest runs per state
            for state in WeatherMarkovChain.STATES:
                max_consecutive = WeatherSimulator._count_max_consecutive(sequence, state)
                if max_consecutive > 0:
                    stats['consecutive_runs'][state].append(max_consecutive)

        # Calculate probabilities
        total_state_count = sum(stats['state_counts'].values())
        for state, count in stats['state_counts'].items():
            stats['state_probabilities'][state] = count / total_state_count

        return stats

    @staticmethod
    def _count_max_consecutive(sequence: List[str], state: str) -> int:
        """Counts longest consecutive occurrences of a state"""
        max_count = 0
        current_count = 0
        for s in sequence:
            if s == state:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        return max_count

    @staticmethod
    def compare_month_statistics(
        sequences_by_month: Dict[str, List[List[str]]]
    ) -> Dict[str, Dict]:
        """
        Compares statistics across multiple months

        Args:
            sequences_by_month: {month: [sequences]}

        Returns:
            {month: statistics}
        """
        comparison = {}
        for month, sequences in sequences_by_month.items():
            comparison[month] = WeatherSimulator.get_sequence_statistics(sequences)
        return comparison

    @staticmethod
    def find_rare_events(
        sequences: List[List[str]],
        event_pattern: str,
        min_length: int = 3
    ) -> Tuple[float, int]:
        """
        Finds how often an event pattern occurs

        Args:
            sequences: Simulated sequences
            event_pattern: E.g. "rainy" (at least min_length days) or "sunny→cloudy" (transitions)
            min_length: Minimum run length for a state

        Returns:
            (Probability in %, Number of sequences with event)
        """
        count = 0

        # If pattern contains a state (no arrows)
        if "→" not in event_pattern:
            state = event_pattern.strip()
            for sequence in sequences:
                # Count longest run of this state
                max_consecutive = WeatherSimulator._count_max_consecutive(sequence, state)
                if max_consecutive >= min_length:
                    count += 1
        else:
            # Transition pattern
            from_state, to_state = [s.strip() for s in event_pattern.split("→")]
            for sequence in sequences:
                for i in range(len(sequence) - 1):
                    if sequence[i] == from_state and sequence[i + 1] == to_state:
                        count += 1
                        break  # Count sequence only once

        probability = (count / len(sequences)) * 100
        return probability, count

    @staticmethod
    def trend_analysis(
        sequences: List[List[str]],
        state: str,
        window_size: int = 7
    ) -> List[float]:
        """
        Analyzes trend: Average of a state per time window

        Args:
            sequences: Simulated sequences
            state: State to analyze
            window_size: Window width in days

        Returns:
            List of averages per window
        """
        sequence_length = len(sequences[0]) if sequences else 0
        trends = []

        for window_start in range(0, sequence_length - window_size + 1, window_size):
            window_end = window_start + window_size
            count = 0

            for sequence in sequences:
                window = sequence[window_start:window_end]
                count += window.count(state)

            avg_in_window = count / (len(sequences) * window_size)
            trends.append(avg_in_window)

        return trends

    @staticmethod
    def get_state_at_day(sequences: List[List[str]], day: int) -> Dict[str, float]:
        """
        Returns probability distribution for a specific day

        Args:
            sequences: Simulated sequences
            day: Day number (0-indexed)

        Returns:
            {state: probability}
        """
        if day >= len(sequences[0]):
            return {}

        states_at_day = [seq[day] for seq in sequences]
        counter = Counter(states_at_day)
        total = len(sequences)

        return {state: count / total for state, count in counter.items()}

    @staticmethod
    def monte_carlo_probability(
        simulator: 'WeatherSimulator',
        start_state: str,
        target_state: str,
        days: int,
        month: str,
        num_simulations: int = 10000
    ) -> float:
        """
        Calculates probability using Monte Carlo method

        Args:
            simulator: WeatherSimulator instance
            start_state: Starting state
            target_state: Target state
            days: Time window in days
            month: Month
            num_simulations: Number of simulations

        Returns:
            Probability that target_state occurs within 'days'
        """
        count = 0
        for _ in range(num_simulations):
            sequence = simulator.simulate_sequence(start_state, days, month)
            if target_state in sequence[1:]:  # [1:] = exclude start_state
                count += 1

        return (count / num_simulations) * 100
