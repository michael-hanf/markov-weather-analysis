"""Converts DWD weather data to Markov format"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class DWDConverter:
    """Converts DWD daily climate data (KL data) to Markov format"""

    # DWD KL format columns (semicolon-separated)
    # Documentation: https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/
    DWD_COLUMNS = {
        'STATIONS_ID': int,
        'MESS_DATUM': str,  # YYYYMMDD
        'QN_3': int,        # Quality precipitation
        'FX': float,        # Maximum wind gust (m/s)
        'FM': float,        # Daily mean wind speed (m/s)
        'QN_4': int,        # Quality temperature/humidity/pressure
        'RSK': float,       # Precipitation (mm)
        'RSKF': int,        # Precipitation indicator
        'SDK': float,       # Sunshine duration (hours)
        'SHK_TAG': float,   # Snow height (cm)
        'NM': float,        # Cloud cover (%)
        'VPM': float,       # Vapor pressure (hPa)
        'PM': float,        # Air pressure (hPa)
        'TMK': float,       # Mean temperature (°C)
        'UPM': float,       # Relative humidity (%)
        'TXK': float,       # Max temperature (°C)
        'TNK': float,       # Min temperature (°C)
        'TGK': float,       # Min ground temperature (°C)
    }

    @staticmethod
    def read_dwd_file(filepath: str) -> pd.DataFrame:
        """
        Reads a DWD KL file and parses it

        Args:
            filepath: Path to DWD file

        Returns:
            DataFrame with parsed data
        """
        # DWD uses semicolon as separator and has 'eor' at the end of each line
        df = pd.read_csv(
            filepath,
            sep=';',
            skipinitialspace=True
        )

        # Remove the 'eor' column
        if 'eor' in df.columns:
            df = df.drop('eor', axis=1)

        # Parse date
        df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d')

        # Convert -999 to NaN (missing values)
        df = df.replace(-999.0, np.nan)
        df = df.replace(-999, np.nan)

        # Sort by date
        df = df.sort_values('MESS_DATUM').reset_index(drop=True)

        return df

    @staticmethod
    def categorize_weather(temperature: float, precipitation: float, wind_speed: float) -> Optional[str]:
        """
        Categorizes weather based on DWD metrics

        Args:
            temperature: Temperature in Celsius (daily mean)
            precipitation: Precipitation in mm
            wind_speed: Wind speed in m/s (not km/h!)

        Returns:
            Weather state as string or None for missing data
        """
        # Skip missing values
        if pd.isna(temperature) or pd.isna(wind_speed):
            return None

        # Convert m/s to km/h
        wind_kmh = wind_speed * 3.6

        # Default precipitation to 0 if NaN
        precip = precipitation if not pd.isna(precipitation) else 0.0

        # Categorization (same logic as sample data)
        if wind_kmh > 50:  # Storm
            return "stormy"

        if precip > 5:  # Rain
            return "rainy"

        if temperature > 20 and precip < 1 and wind_kmh < 20:
            return "sunny"

        if precip < 1 and wind_kmh < 15:
            return "partly_cloudy"

        return "cloudy"

    @classmethod
    def convert_to_markov_format(
        cls,
        dwd_filepath: str,
        output_csv: str = "data/raw/nordfriesland_2024.csv"
    ) -> pd.DataFrame:
        """
        Converts DWD data to Markov input format

        Args:
            dwd_filepath: Path to DWD file
            output_csv: Output file for Markov format

        Returns:
            DataFrame in Markov format
        """
        # Read DWD file
        df = cls.read_dwd_file(dwd_filepath)

        # Convert columns
        markov_df = pd.DataFrame({
            'date': df['MESS_DATUM'],
            'temperature': df['TMK'],  # Daily mean
            'precipitation': df['RSK'],  # Precipitation
            'wind_speed': df['FM'],  # Daily mean wind speed in m/s
        })

        # Categorize
        markov_df['weather_state'] = markov_df.apply(
            lambda row: cls.categorize_weather(
                row['temperature'],
                row['precipitation'],
                row['wind_speed']
            ),
            axis=1
        )

        # Remove rows with missing weather states
        markov_df = markov_df.dropna(subset=['weather_state']).reset_index(drop=True)

        # Save CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markov_df.to_csv(output_path, index=False)

        return markov_df

    @staticmethod
    def print_statistics(df: pd.DataFrame):
        """Shows statistics about the weather data"""
        print(f"\nData Statistics")
        print(f"Period: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"Number of days: {len(df)}")
        print(f"\nWeather state distribution:")

        for state, count in df['weather_state'].value_counts().items():
            percent = (count / len(df)) * 100
            print(f"  - {state:15}: {count:4} days ({percent:5.1f}%)")

        print(f"\nTemperature statistics:")
        print(f"  Min:     {df['temperature'].min():6.1f} °C")
        print(f"  Max:     {df['temperature'].max():6.1f} °C")
        print(f"  Mean:    {df['temperature'].mean():6.1f} °C")

        print(f"\nPrecipitation statistics:")
        precip_days = (df['precipitation'] > 0).sum()
        print(f"  Rainy days: {precip_days} ({precip_days/len(df)*100:.1f}%)")
        print(f"  Mean:       {df['precipitation'].mean():6.1f} mm")
        print(f"  Total:      {df['precipitation'].sum():6.1f} mm")

        print(f"\nWind speed statistics (daily mean):")
        wind_kmh = df['wind_speed'] * 3.6  # Convert m/s to km/h
        print(f"  Min:     {wind_kmh.min():6.1f} km/h")
        print(f"  Max:     {wind_kmh.max():6.1f} km/h")
        print(f"  Mean:    {wind_kmh.mean():6.1f} km/h")
