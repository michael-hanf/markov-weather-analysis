"""Markov-Ketten Wettervorhersage für Bundesländer"""

__version__ = "0.1.0"

from .model import WeatherMarkovChain
from .predictor import Predictor

__all__ = ["WeatherMarkovChain", "Predictor"]
