"""Prediction interface for weather forecasting"""

from typing import Dict, List
from .model import WeatherMarkovChain


class Predictor:
    """Wrapper for simple predictions"""

    def __init__(self, region: str):
        """
        Initializes the predictor

        Args:
            region: Name of the region
        """
        self.region = region
        self._chains_cache = {}

    def _get_chain(self, month: str) -> WeatherMarkovChain:
        """Loads or caches a Markov chain"""
        if month not in self._chains_cache:
            self._chains_cache[month] = WeatherMarkovChain(self.region, month)
        return self._chains_cache[month]

    def predict(self, current_state: str, n_days: int, month: str) -> List[Dict[str, float]]:
        """
        Makes a prediction for N days

        Args:
            current_state: Current weather state
            n_days: Number of forecast days
            month: Month for the forecast

        Returns:
            List of probability dicts
        """
        chain = self._get_chain(month)
        return chain.predict_n_days(current_state, n_days)

    def predict_most_likely(self, current_state: str, n_days: int, month: str) -> List[str]:
        """
        Makes a prediction with the most likely states

        Args:
            current_state: Current weather state
            n_days: Number of forecast days
            month: Month for the forecast

        Returns:
            List of most likely states for each day
        """
        chain = self._get_chain(month)
        predictions = chain.predict_n_days(current_state, n_days)

        most_likely = []
        for probs in predictions:
            state, _ = chain.get_most_likely_state(probs)
            most_likely.append(state)

        return most_likely
