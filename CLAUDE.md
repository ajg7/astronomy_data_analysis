# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A Python learning project for ML/AI using a star dataset. The goal is to build hands-on experience with machine learning concepts applied to real astronomical data.

## Dataset: `star_dataset.csv`

Six columns describing named stars:

| Column | Description |
|---|---|
| `Name` | Star name (e.g., Sirius, Betelgeuse) |
| `Distance (ly)` | Distance from Earth in light years |
| `Luminosity (L/Lo)` | Luminosity relative to the Sun |
| `Radius (R/Ro)` | Radius relative to the Sun |
| `Temperature (K)` | Surface temperature in Kelvin |
| `Spectral Class` | Stellar classification (e.g., A1V, M2Iab, B7V) |

**Data quality notes:**
- Some rows have negative luminosity values — likely noise or data entry errors requiring cleaning before use.
- The same star name appears multiple times (e.g., Barnard's Star, Capella, Regulus) with slightly varying values, representing repeated/duplicate observations rather than distinct stars.
- Spectral classes present: O, B, A, F, G, K, M — useful as a multi-class classification target.

## Natural ML Tasks

- **Classification:** Predict spectral class from numerical features (temperature, luminosity, radius, distance)
- **Regression:** Predict luminosity or radius from other physical properties
- **Clustering:** Discover stellar groups without labels (unsupervised)
- **Data cleaning:** Handle negatives, duplicates, outliers before modeling

## Environment Setup

No environment is configured yet. Recommended stack for this project:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

Run a notebook:
```bash
jupyter notebook
```

Run a Python script:
```bash
python <script_name>.py
```
