# Markov Chain Weather Prediction - Code

Python-based implementation of Markov chain models for weather state prediction. This system uses historical weather data to compute transition probabilities between different weather states.

## Project Structure

```
code/
├── markov_weather/           # Python package
│   ├── __init__.py
│   ├── model.py              # Markov chain model (1st order)
│   ├── model_2order.py       # Higher-order Markov models (2nd/3rd order)
│   ├── data_loader.py        # Data loader and categorization
│   ├── matrix_builder.py     # Transition matrix builder (1st order)
│   ├── matrix_builder_2order.py  # Higher-order matrix builder
│   ├── predictor.py          # Prediction interface
│   ├── simulator.py          # Monte Carlo simulation
│   ├── validation.py         # Model validation (1st order)
│   ├── validation_2order.py  # Validation for higher-order models
│   └── dwd_converter.py      # DWD data format converter
├── data/
│   ├── raw/                  # Raw weather data (CSV)
│   └── matrices/             # Transition matrices (JSON)
├── cli.py                    # Command-line interface
├── requirements.txt
└── README.md
```

## Weather States

The model uses 5 discrete weather states:

- **sunny** (☀️)
- **partly_cloudy** (⛅)
- **cloudy** (☁️)
- **rainy** (🌧️)
- **stormy** (⛈️)

## Setup

### Create Virtual Environment and Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Generate Sample Data

```bash
python cli.py generate-sample-data --records 1095
```

This creates 3 years (1095 days) of realistic sample weather data.

### Generate Transition Matrices

```bash
python cli.py generate-matrices
```

This computes 12 month-specific transition matrices (one per month) from weather data and saves them as JSON files.

## CLI Commands

### Make Predictions

```bash
# Simple prediction (most likely states only)
python cli.py predict --state sunny --days 3 --month january

# With probability distributions
python cli.py predict --state cloudy --days 5 --month june --probabilities
```

### Simulation & Analysis

```bash
# Simulate realistic weather sequences
python cli.py simulate --state cloudy --days 30 --month november --runs 1000 --show-sequences

# Calculate probability (Monte Carlo)
python cli.py probability --state rainy --target sunny --days 14 --month august

# Compare months
python cli.py compare-months --state sunny --months january --months july --runs 1000 --days 30

# Find rare events
python cli.py rare-event --state sunny --event rainy --min-length 5 --month september --days 30
```

### Model Validation

```bash
# Validate against real DWD data
python cli.py validate --region nordfriesland --data-file data/raw/nordfriesland_2024.csv
```

**Options:**
- `--state`: Current weather state (sunny|partly_cloudy|cloudy|rainy|stormy)
- `--days`: Number of forecast days (default: 3)
- `--month`: Month (e.g., january, february, ...). If not specified: current month
- `--probabilities`: Show all probability distributions instead of only most likely state
- `--region`: Region name (default: schleswig_holstein)

### Show Transition Matrix

```bash
python cli.py show-matrix --month january
```

Displays the 5×5 transition matrix for the specified month.

### Generate Sample Data

```bash
python cli.py generate-sample-data --records 1095
```

Creates realistic sample weather data based on:
- Seasonal temperature patterns
- Random precipitation events (~30% rainy days)
- Random wind speeds

### Create Matrices from Weather Data

```bash
python cli.py generate-matrices --data-file data/raw/sample_weather.csv
```

## Data Format

### CSV Format (Input)

```csv
date,temperature,precipitation,wind_speed,weather_state
2022-01-01,5.2,0.0,12.3,partly_cloudy
2022-01-02,4.8,2.1,18.5,rainy
...
```

The system automatically categorizes based on meteorological values:
- **Stormy**: wind_speed > 50 km/h
- **Rainy**: precipitation > 5 mm
- **Sunny**: temp > 20°C, precip < 1mm, wind < 20 km/h
- **Partly cloudy**: precip < 1mm, wind < 15 km/h
- **Cloudy**: Default (other combinations)

### JSON Format (Transition Matrices)

```json
{
  "region": "schleswig_holstein",
  "month": "january",
  "states": ["sunny", "partly_cloudy", "cloudy", "rainy", "stormy"],
  "matrix": [
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.444, 0.528, 0.028, 0.0],
    ...
  ]
}
```

The matrix is a 5×5 matrix where:
- **Rows** = current state
- **Columns** = next state
- **Values** = transition probabilities (0-1)

## Algorithm: Markov Chains

The model uses 1st order Markov chains:

1. **Transition Probability**: P(tomorrow = B | today = A)
2. **N-Day Forecast**: Repeated matrix multiplication
   - Vector (current state) × Matrix^n = Probability distribution after n days

### Laplace Smoothing

To prevent impossible states, **Laplace smoothing** is used:
- Each transition count is increased by a small constant (1e-6)
- Guarantees that all states are possible with minimal probability

## Extensions

The system is designed to be easily extended:

### Add New Regions

```python
from markov_weather.matrix_builder import MatrixBuilder

# For Bayern (Bavaria)
builder = MatrixBuilder("bayern")
builder.build_from_file("data/raw/bayern_weather.csv")
```

### More Weather States

Modify `WeatherMarkovChain.STATES` in `model.py` and `categorize_weather()` in `data_loader.py`.

### Real Weather Data

Download historical data from DWD (Deutscher Wetterdienst):
- https://www.dwd.de/DE/leistungen/met_verfahren_mosmix/mosmix_stationskatalog.cfg

Or use Open-Meteo API:
- https://open-meteo.com/

## Examples

### Scenario 1: Fair Weather Forecast
```bash
python cli.py predict --state sunny --days 7 --month august --probabilities
```

### Scenario 2: Autumn Weather Trends
```bash
python cli.py predict --state cloudy --days 5 --month october
python cli.py show-matrix --month october
```

### Scenario 3: Matrix Comparison
```bash
python cli.py show-matrix --month january  # Winter
python cli.py show-matrix --month july     # Summer
```

## Mathematical Foundations

### Transition Matrix Properties
- **Stochastic Matrix**: Each row sums to 1.0
- **Month-Specific**: Accounts for seasonal variations
- **Memoryless (Markov Property)**: Future depends only on present, not past

### Prediction Calculation
```
p(n) = p(0) × M^n

where:
- p(0) = initial state vector (one-hot for current state)
- M = transition matrix
- n = number of days
- p(n) = probability distribution after n days
```

## Limitations

1. **1st Order**: Model considers only immediate previous day, not longer-term trends
2. **Seasonality**: Only month-specific, not year-round seasonality for transition months
3. **Independence from Temperature**: States are categorized, no continuous values
4. **No External Factors**: Does not consider large-scale weather patterns or climate models

## Improvement Potential

- Higher-order Markov chains (2nd-3rd order)
- Daily matrices instead of monthly
- Integration of real DWD data
- Hidden Markov Models (HMM) for better predictions
- Multiple regions with spatial correlation

## Related Research

This code accompanies the research paper:

**"Markov Chains for Weather Prediction: A Geographic Spectrum Analysis"**

See [`../paper/Markov_Weather_Analysis.md`](../paper/Markov_Weather_Analysis.md) for detailed analysis of:
- 336 trained models across 3 regions
- 10-30 years of historical data
- Applicability framework for Markov models
- Performance analysis across stability spectrum

## License

Research code - see main repository README for details.

