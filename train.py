import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout,
                                     BatchNormalization, Input, GRU)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error)

import mlflow
import mlflow.keras

from utils import load_and_preprocess, create_sequences

# ---------------------------
# CONFIG
# ---------------------------
WINDOW_SIZE = 30
EXPERIMENT_NAME = "AEP_Energy_Forecasting"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------
# LOAD DATA
# ---------------------------
daily_df = load_and_preprocess("data/AEP_hourly.csv")

values = daily_df.values
N_FEATURES = values.shape[1]

# ---------------------------
# SPLIT
# ---------------------------
split_idx = int(len(values) * 0.80)

train_raw = values[:split_idx]
test_raw  = values[split_idx:]

feature_scaler = MinMaxScaler()
target_scaler  = MinMaxScaler()

X_train_scaled = feature_scaler.fit_transform(train_raw)
X_test_scaled  = feature_scaler.transform(test_raw)

y_train_scaled = target_scaler.fit_transform(train_raw[:, 0].reshape(-1, 1))
y_test_scaled  = target_scaler.transform(test_raw[:, 0].reshape(-1, 1))

# ---------------------------
# SEQUENCES
# ---------------------------
X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, WINDOW_SIZE)
X_test,  y_test  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW_SIZE)

val_split = int(len(X_train) * 0.90)
X_tr, X_val = X_train[:val_split], X_train[val_split:]
y_tr, y_val = y_train[:val_split], y_train[val_split:]

# ---------------------------
# MODEL (UNCHANGED)
# ---------------------------
def build_model(units, dropout_rate, learning_rate):

    inputs = Input(shape=(WINDOW_SIZE, N_FEATURES))

    x = GRU(units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)

    x = LSTM(units // 2, return_sequences=True,
             kernel_regularizer=l2(1e-4))(x)

    x = GRU(units//4, return_sequences=False)(x)

    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)

    output = Dense(1)(x)

    model = Model(inputs, output)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=Huber(delta=1.0),
        metrics=['mae']
    )
    return model


# ---------------------------
# EVALUATION (UNCHANGED)
# ---------------------------
def evaluate_model(model, X, y, scaler):
    y_pred_scaled = model.predict(X, verbose=0)

    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y.reshape(-1, 1))

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) /
                           np.clip(y_true, 1, None))) * 100

    return mae, rmse, mape, y_pred.flatten(), y_true.flatten()


# ---------------------------
# CONFIGS (ORIGINAL NAMES)
# ---------------------------
configs = [
    {"run_name": "LSTM_64_baseline",     "units": 64,  "dropout_rate": 0.2, "learning_rate": 0.001,  "batch_size": 32,  "epochs": 50},
    {"run_name": "LSTM_128_standard",    "units": 128, "dropout_rate": 0.2, "learning_rate": 0.001,  "batch_size": 64,  "epochs": 50},
    {"run_name": "LSTM_128_low_lr",      "units": 128, "dropout_rate": 0.3, "learning_rate": 0.0015, "batch_size": 32,  "epochs": 80},
    {"run_name": "LSTM_256_low_dropout", "units": 256, "dropout_rate": 0.1, "learning_rate": 0.0018, "batch_size": 64,  "epochs": 80},
    {"run_name": "LSTM_256_large",       "units": 256, "dropout_rate": 0.3, "learning_rate": 0.0015, "batch_size": 32,  "epochs": 80},
]

# ---------------------------
# CALLBACKS
# ---------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_mae',
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)

# ---------------------------
# TRAIN LOOP (UNCHANGED)
# ---------------------------
for cfg in configs:

    checkpoint = ModelCheckpoint(
        filepath=f"{cfg['run_name']}_best.keras",
        monitor='val_loss',
        save_best_only=True
    )

    with mlflow.start_run(run_name=cfg['run_name']):

        mlflow.log_params(cfg)
        mlflow.log_param("window_size", WINDOW_SIZE)

        model = build_model(
            cfg['units'],
            cfg['dropout_rate'],
            cfg['learning_rate']
        )

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            callbacks=[early_stop, lr_scheduler, checkpoint],
            verbose=1
        )

        model = tf.keras.models.load_model(f"{cfg['run_name']}_best.keras")

        mae, rmse, mape, _, _ = evaluate_model(
            model, X_test, y_test, target_scaler
        )

        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mape", mape)

        mlflow.keras.log_model(model, "model")

        print(f"{cfg['run_name']} → MAE: {mae:.2f}")