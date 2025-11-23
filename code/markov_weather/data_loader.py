"""Data loader for weather data"""

import pandas as pd
from pathlib import Path
from typing import Optional
import numpy as np


class WeatherDataLoader:
    """Loads and processes weather data"""

    def __init__(self, data_dir: str = "data/raw"):
        """
        Initializes the data loader

        Args:
            data_dir: Directory with raw data
        """
        self.data_dir = Path(data_dir)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Loads a CSV file with weather data

        Args:
            filename: Name of the CSV file

        Returns:
            DataFrame with weather data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, parse_dates=['date'])
        return df.sort_values('date')

    @staticmethod
    def categorize_weather(temperature: float, precipitation: float, wind_speed: float) -> str:
        """
        Categorizes weather based on meteorological data

        Args:
            temperature: Temperature in Celsius
            precipitation: Precipitation in mm
            wind_speed: Wind speed in km/h

        Returns:
            Weather state as string
        """
        # Priority: Storm > Rain > Cloudy > Sun

        if wind_speed > 50:  # Storm
            return "stormy"

        if precipitation > 5:  # Rain (> 5mm)
            return "rainy"

        if temperature > 20 and precipitation < 1 and wind_speed < 20:
            # Warm, dry, calm → sunny (based on thresholds)
            return "sunny"

        if precipitation < 1 and wind_speed < 15:
            return "partly_cloudy"

        # Default: Cloudy
        return "cloudy"

    def create_sample_data(self, output_file: str = "sample_weather.csv", n_records: int = 365):
        """
        Creates sample data for testing (realistic random weather patterns)

        Args:
            output_file: Output file
            n_records: Number of records
        """
        np.random.seed(42)
        dates = pd.date_range(start="2022-01-01", periods=n_records, freq='D')

        # Generate realistic weather patterns
        base_temp = 10
        temps = base_temp + 10 * np.sin(np.arange(n_records) * 2 * np.pi / 365) + np.random.normal(0, 3, n_records)

        # Rain: random, but clustered
        precips = np.random.exponential(2, n_records)
        precips = np.where(np.random.random(n_records) < 0.3, precips, 0)  # 30% rainy days

        # Wind
        winds = np.abs(np.random.normal(15, 8, n_records))

        data = pd.DataFrame({
            'date': dates,
            'temperature': temps,
            'precipitation': precips,
            'wind_speed': winds
        })

        # Categorize
        data['weather_state'] = data.apply(
            lambda row: self.categorize_weather(row['temperature'], row['precipitation'], row['wind_speed']),
            axis=1
        )

        output_path = self.data_dir / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(output_path, index=False)

        return data
