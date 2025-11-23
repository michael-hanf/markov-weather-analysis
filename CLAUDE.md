# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project analyzing Markov chain applicability for weather prediction across three German climatic regions (coastal, inland, alpine). Includes 336 trained models spanning 10-30 years of weather data, with a full Python implementation and accompanying research paper.

## Environment Setup

```bash
# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

**Working directory**: Always `cd` to `code/` directory before running commands.

## Core CLI Commands

### Data Generation and Matrix Building

```bash
# Generate sample weather data (3 years by default)
python cli.py generate-sample-data --records 1095

# Build 1st order transition matrices from data
python cli.py generate-matrices --data-file data/raw/sample_weather.csv --region schleswig_holstein

# Build higher-order matrices (2nd or 3rd order)
python cli.py generate-matrices-higher --data-file data/raw/nordfriesland_2024.csv --region nordfriesland --order 2
```

### Weather Prediction

```bash
# Simple prediction (most likely states)
python cli.py predict --state sunny --days 3 --month january --region schleswig_holstein

# With full probability distributions
python cli.py predict --state cloudy --days 5 --month june --probabilities
```

### Simulation and Analysis

```bash
# Monte Carlo simulation
python cli.py simulate --state cloudy --days 30 --month november --runs 1000 --show-sequences

# Calculate event probability
python cli.py probability --state rainy --target sunny --days 14 --month august

# Compare months
python cli.py compare-months --months january --months july --runs 1000 --state sunny --days 30

# Find rare events
python cli.py rare-event --event rainy --min-length 5 --month september --state sunny --days 30
```

### Model Validation

```bash
# Validate 1st order model against real DWD data
python cli.py validate --region nordfriesland --data-file data/raw/nordfriesland_2024.csv

# Validate higher-order model
python cli.py validate-higher --region nordfriesland --data-file data/raw/nordfriesland_2024.csv --order 2
```

### Utilities

```bash
# Show transition matrix
python cli.py show-matrix --month january --region schleswig_holstein
```

## Architecture Overview

### Multi-Order Markov Implementation

The codebase supports **three different Markov model orders**:

1. **1st Order** (`model.py`, `matrix_builder.py`): Prediction based on current state only
   - Matrix size: 5×5 (states × states)
   - Example: P(tomorrow | today)

2. **2nd Order** (`model_2order.py`, `matrix_builder_2order.py`): Uses last 2 days
   - Matrix size: 25×5 (state pairs × states)
   - Example: P(tomorrow | yesterday, today)

3. **3rd Order** (same modules as 2nd order): Uses last 3 days
   - Matrix size: 125×5 (state triples × states)
   - Example: P(tomorrow | 2 days ago, yesterday, today)

**Key Insight**: Higher orders require exponentially more data (evident in paper findings that 30 years of data shows no improvement over 10 years).

### Module Architecture

```
markov_weather/
├── model.py                    # 1st order Markov chain
├── model_2order.py             # 2nd/3rd order Markov chain (configurable)
├── matrix_builder.py           # Builds 1st order matrices from CSV data
├── matrix_builder_2order.py    # Builds higher-order matrices
├── data_loader.py              # CSV loading & weather state categorization
├── predictor.py                # High-level prediction interface (1st order)
├── simulator.py                # Monte Carlo simulation & statistics
├── validation.py               # 1st order model validation
├── validation_2order.py        # Higher-order model validation
└── dwd_converter.py            # DWD data format converter
```

### Data Flow

1. **Raw CSV** → `data_loader.py` → Categorized weather states (sunny, cloudy, rainy, stormy, partly_cloudy)
2. **Categorized data** → `matrix_builder.py` or `matrix_builder_2order.py` → JSON transition matrices
3. **Matrices** → `model.py` or `model_2order.py` → Predictions via matrix multiplication
4. **Predictions** → `simulator.py` → Monte Carlo analysis & statistics

### Weather State Categorization

Logic in `data_loader.py`:
- **Stormy**: wind_speed > 50 km/h
- **Rainy**: precipitation > 5 mm
- **Sunny**: temp > 20°C, precip < 1mm, wind < 20 km/h
- **Partly cloudy**: precip < 1mm, wind < 15 km/h
- **Cloudy**: Default (all other combinations)

### Matrix Storage Convention

```
data/matrices/
├── {region}/              # 1st order matrices
│   └── {month}.json
├── order2/{region}/       # 2nd order matrices
│   └── {month}.json
└── order3/{region}/       # 3rd order matrices
    └── {month}.json
```

Each JSON contains:
- `region`: Region name
- `month`: Month name
- `states`: List of weather states
- `matrix`: 2D array of transition probabilities (rows sum to 1.0)

### Laplace Smoothing

Applied in `matrix_builder.py` to prevent zero-probability transitions:
- Adds small constant (1e-6) to all transition counts before normalization
- Ensures all state transitions remain theoretically possible

## Available Regions

Current datasets in `code/data/raw/`:
- `nordfriesland` (coastal - stable, low entropy)
- `schleswig_holstein` (coastal)
- `bayern` (alpine - complex, high entropy)

**Note**: Region names use German names (Bayern, not Bavaria) for consistency with the research data.

To add a new region: Place CSV in `data/raw/{region}_*.csv`, then run `generate-matrices` with `--region {region}`.

## Testing & Development

**No formal test suite currently exists.** Validation is done via:
- `validate` command: Compares model predictions against real weather data
- `validate-higher`: Same for higher-order models
- Metrics: MAE (Mean Absolute Error), prediction accuracy, state distribution comparison

When developing new features:
1. Test with sample data first (`generate-sample-data`)
2. Validate against real DWD data if available
3. Use `show-matrix` to inspect transition probabilities
4. Run simulations to verify statistical behavior

## Research Paper Context

Full manuscript: `paper/Markov_Weather_Analysis.md`

Key research findings (relevant when modifying models):
- **Data quantity plateau**: 10y → 30y yields negligible accuracy improvement
- **Problem structure limits**: External factors (>3) impose hard accuracy ceiling
- **Lift paradox**: Markov models add most value (vs. baseline) in chaotic regions, despite lower absolute accuracy
- **State persistence**: High-entropy regions show rapid transitions; low-entropy regions favor state persistence

## Extension Points

### Adding New Weather States

1. Modify `WeatherMarkovChain.STATES` in `model.py`
2. Update `categorize_weather()` in `data_loader.py`
3. Regenerate all matrices (matrix dimensions will expand)

### Adding New Months/Seasonality

Currently month-specific matrices are used. For finer granularity:
- Modify `MatrixBuilder.build_transition_matrix()` to split by date ranges
- Update file naming convention

### Integration with Real Weather APIs

Placeholder in `dwd_converter.py` for DWD (Deutscher Wetterdienst) data.
Alternative: Open-Meteo API (https://open-meteo.com/)

## Common Pitfalls

- **Matrix not found errors**: Run `generate-matrices` before predictions
- **Month names**: Use lowercase English month names (e.g., "january", not "January" or "Januar")
- **Working directory**: CLI expects to be run from `code/` directory
- **State names**: Must match exactly: `sunny`, `partly_cloudy`, `cloudy`, `rainy`, `stormy`
- **Higher-order models**: Require sufficient data density (sparse data → unreliable matrices)

## Development Language

Code comments and CLI output are in German. Variable names and function signatures are in English/German mix.
