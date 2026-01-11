# ML-Based Air Traffic Conflict Detection & Decision Support

## Overview

This project implements an end-to-end machine learning pipeline for analyzing real-world
air traffic data and supporting air traffic conflict detection and decision-making.

Using historical ADS-B aircraft state vectors, the system:

- Monitors live airspace snapshots
- Detects abnormal aircraft behavior
- Predicts current and future loss-of-separation risks
- Identifies congested airspace regions
- Generates actionable safety recommendations
- Presents results through an interpretable visualization dashboard

The focus of this project is decision support, not automation — all outputs are designed
to assist human operators.

## System Architecture (High Level)

The notebook follows a structured, modular pipeline:

- Data Ingestion & Sampling
- Data Cleaning & Feature Engineering
- Anomaly Detection (Isolation Forest)
- Conflict Prediction (Random Forest)
- Traffic Density & Congestion Analysis (DBSCAN)
- Short-Term Trajectory Forecasting (Kinematic Model)
- Look-Ahead Conflict Risk Assessment
- Conflict Resolution & Route Adjustment Strategies
- ML-Powered Decision Support
- Visualization & Situational Awareness Dashboard

Each stage builds on the previous one and can be inspected independently.

## Dataset Description

### Data Source

OpenSky Network
Historical ADS-B state vector data collected from real-world air traffic operations.

The data is used for research and demonstration purposes only.

### Dataset Characteristics

Each CSV file contains time-stamped aircraft state information, including:

- Aircraft identifier (`icao24`, `callsign`)
- Geographic position (latitude, longitude)
- Altitude (geoaltitude, baroaltitude)
- Velocity and heading
- Vertical rate
- Operational flags (on-ground status, alerts)
- Timestamps

Example dataset used in the notebook:

`states_2022-01-03-00.csv`

### Sampling Strategy

Due to the large size of raw ADS-B datasets, the notebook:

- Limits the number of loaded records during exploration
- Applies temporal downsampling per aircraft
- Filters only airborne aircraft for modeling

This ensures fast iteration while preserving realistic traffic behavior.

## Machine Learning Models Used

1. **Anomaly Detection — Isolation Forest**

   - Detects aircraft exhibiting unusual kinematic behavior
   - Unsupervised (no anomaly labels required)
   - Outputs anomaly flags and severity scores

2. **Conflict Prediction — Random Forest**

   - Supervised model trained on relative aircraft geometry
   - Predicts probability of loss of separation
   - Provides feature importance for interpretability
   - Reused for both current and future conflict assessment

3. **Traffic Clustering — DBSCAN**

   - Identifies high-density airspace regions
   - Detects congested zones and isolated aircraft
   - No predefined number of clusters required

4. **Trajectory Prediction — Kinematic Model**

   - Physics-based short-term forecasting (5-minute horizon)
   - Assumes constant velocity, heading, and vertical rate
   - Used for look-ahead conflict detection

## Decision Support & Recommendations

The system does not simply output predictions.
It translates model outputs into actionable guidance, including:

- Anomaly alerts for individual aircraft
- Immediate conflict warnings
- Predicted future conflicts
- Airspace congestion warnings
- Suggested altitude and heading adjustments

Recommendations are:

- Prioritized (CRITICAL / HIGH / MEDIUM)
- Labeled by source (ML / Unsupervised ML / Hybrid)
- Designed for human-in-the-loop decision-making

## Visualization Dashboard

The notebook includes a comprehensive dashboard that visualizes:

- Current and predicted aircraft positions
- Conflict probability distributions
- Traffic clustering results
- Anomaly statistics
- Altitude and velocity profiles
- Model status summary

The dashboard serves as the final situational awareness layer of the system.

## How to Run

### Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

Install dependencies:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Running the Notebook

- Place the ADS-B CSV file(s) in the project directory
- Open the notebook in Jupyter or Google Colab
- Run cells top to bottom to ensure proper state initialization

## Project Scope & Limitations

This system is designed for decision support, not autonomous control

- Trajectory prediction assumes constant motion over short horizons
- Conflict labels are derived from standard separation minima
- Results should not be used for real-world operational control

## Future Extensions

Possible improvements include:

- Intent-aware trajectory modeling
- Probabilistic motion uncertainty
- Reinforcement learning for resolution selection
- Streaming / real-time integration
- Sector-based airspace modeling
