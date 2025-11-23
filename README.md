# Markov Chains for Weather Prediction  
**A Geographic Spectrum Analysis Across Three Climatic Regimes**

This repository contains the full research manuscript, analysis code, and accompanying figures for the study:

**"Markov Chains for Weather Prediction: A Geographic Spectrum Analysis"**  
Author: Michael Hanf  
Version: 2.4 (2025)

The goal of this work is to understand *where* and *why* Markov models succeed or fail by analyzing real-world weather data across three structurally distinct regions.

---

## Motivation

This project started as a personal attempt to understand Markov chains in a practical, intuitive way.  
Living in a coastal region, I began by testing whether a simple Markov model could predict local daily weather states.  
The accuracy turned out unexpectedly high, which raised a deeper question:

**Was this a coincidence of the local climate, or a structural property of Markov models?**

To explore this, I extended the analysis to inland and alpine regions with very different atmospheric dynamics.  
What started as a learning experiment evolved into a systematic study of when and why Markov chains succeed — and where their structural limits become visible.

## ✨ Summary

Markov chains are simple, interpretable models that show inconsistent performance across domains.  
This study investigates the structural factors that determine whether Markov models are suitable for a problem.

We analyze **336 models** trained on **10–30 years** of weather data across three German regions:

- **Region A – Coastal-Stable** (low entropy, high persistence)  
- **Region B – Inland-Transitional**  
- **Region C – Alpine-Complex** (chaotic, multi-factor)

Key findings:

- Markov models perform best in stable regions (high absolute accuracy).  
- Their **relative value** (Lift) is highest in chaotic regions.  
- Increasing data (10y → 30y) yields **no measurable improvement**.  
- Problem *structure* limits accuracy, not data volume.

---

## 📊 Figures

### 1. Stability Spectrum  
State distribution from dominance to chaos.

![Figure 1](paper/figures/Figure1_Spectrum.png)

### 2. Accuracy vs. Baseline ("Lift" Paradox)  
Markov adds most value where absolute accuracy is lowest.

![Figure 2](paper/figures/Figure2_Lift.png)

### 3. Data Quantity Plateau  
Tripling data volume yields negligible accuracy gain.

![Figure 3](paper/figures/Figure3_Data.png)

---

## 📄 Manuscript

The full paper is available in two formats:

- **PDF**: [`paper/Markov_Weather_Analysis.pdf`](paper/Markov_Weather_Analysis.pdf) ⬅️ **Recommended for reading**
- **Markdown**: [`paper/Markov_Weather_Analysis.md`](paper/Markov_Weather_Analysis.md)

It includes:

- Motivation & framework  
- Methodology (data, discretization, training)  
- Regional spectrum analysis  
- Detailed results (accuracy, lift, entropy)  
- Applicability conditions for Markov models  
- References  

---

## 🔍 Research Questions

1. Which structural factors define the boundary of Markov model applicability?  
2. Can additional data overcome these structural limits?  
3. Are these principles generalizable to other domains (e.g., finance, pandemics, system modeling)?  

---

## 🧠 Key Insights

- **Dominant states** drive high accuracy.  
- **Transition stability** is crucial for Markov success.  
- **External factors** (>3) impose hard accuracy limits.  
- **High-order models** require exponential data.  
- **Short memory systems** align best with Markov assumptions.  

---

## 📁 Repository Structure

```
.
├── paper/
│   ├── figures/
│   │   ├── Figure1_Spectrum.png
│   │   ├── Figure2_Lift.png
│   │   └── Figure3_Data.png
│   └── Markov_Weather_Analysis.md
├── code/
│   ├── markov_weather/         # Python package
│   ├── data/                   # Weather data and matrices
│   ├── cli.py                  # Command-line interface
│   ├── requirements.txt
│   └── README.md               # Code documentation
└── README.md
```

## 🚀 Getting Started

### Running the Code

See [`code/README.md`](code/README.md) for detailed instructions on:
- Setting up the Python environment
- Generating sample data
- Training Markov models
- Making predictions
- Validating against real weather data

