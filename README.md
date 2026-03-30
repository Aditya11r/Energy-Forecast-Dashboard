Energy Demand Forecasting using LSTM + GRU (MLOps + Streamlit)
Overview

This project focuses on time-series forecasting of energy demand using deep learning models (LSTM & GRU), combined with MLOps practices (MLflow) and a Streamlit frontend for real-time predictions.
The system predicts future energy consumption based on historical patterns and temporal features like weekly and seasonal trends.

Key Features:
Time-series forecasting using LSTM + GRU hybrid model
Sliding window approach for sequential learning
Cyclical encoding for capturing weekly and seasonal patterns
Experiment tracking using MLflow
Interactive frontend using Streamlit
Visualization of Actual vs Predicted values
Future forecasting with user-selected date range

Learning Phases

Phase 1: Data Preparation:
Converted date column to datetime64
Sorted and indexed time-series data
Resampled hourly data to daily
Performed scaling (normalization)
Data split:
80% Train / 20% Test (chronological)
Train further split into 90% Train / 10% Validation

Phase 2: Model Development & Experimentation:
Built multiple models:
Stacked LSTM
Hybrid LSTM + GRU
Implemented:
Sliding window technique
Prevention of data leakage
Logged experiments using MLflow:
Hyperparameters (units, learning rate, dropout)
Metrics (MAE, RMSE, MAPE)
Model artifacts
Initial Performance
MAE ≈ 650–670 MW
RMSE ≈ 860 MW
MAPE ≈ 4.5%

Phase 3: Model Improvement:
Increased window size from 14 to 30
Identified missing temporal patterns
Introduced cyclical encoding
Encoded week/day using sine and cosine transformation
Enabled model to understand cyclic nature of time
Features increased from 1 to 7
Improved Performance
MAE: 543.67 MW
RMSE: 714.48 MW
SMAPE: 3.64%
Direction Accuracy: 70.99%

Phase 4: Deployment (Streamlit App):
Built interactive UI using Streamlit
Integrated MLflow model loading
Features:
Date range selection
Future forecasting
Visualization of predictions

Final Model Performance:
MAE: 499.65 MW
RMSE: 676.46 MW
MAPE: 3.18%
SMAPE: 3.18%
Direction Accuracy: 72.94%

Observations & Limitations:
Model performs well for short-term forecasting (approximately 10 days)
Beyond this range:
Predictions become smoother
Variability decreases
Common limitation of recursive forecasting in deep learning

Future Improvements:
Incorporate exogenous variables (weather, holidays)
Experiment with attention-based models or Transformers
Use multi-step forecasting instead of recursive prediction
Add uncertainty estimation (confidence intervals)

Author:
Aditya Raj
