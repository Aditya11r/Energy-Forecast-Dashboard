import numpy as np
import pandas as pd


def load_and_preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime').sort_index().dropna()

    daily = df['AEP_MW'].resample('D').mean().dropna()

    daily_df = daily.to_frame(name='AEP_MW')

    # Cyclical encoding
    daily_df['dow_sin'] = np.sin(2 * np.pi * daily_df.index.dayofweek / 7)
    daily_df['dow_cos'] = np.cos(2 * np.pi * daily_df.index.dayofweek / 7)

    daily_df['month_sin'] = np.sin(2 * np.pi * daily_df.index.month / 12)
    daily_df['month_cos'] = np.cos(2 * np.pi * daily_df.index.month / 12)

    # Rolling features
    daily_df['rolling_mean_7'] = daily_df['AEP_MW'].rolling(7).mean()
    daily_df['rolling_std_7']  = daily_df['AEP_MW'].rolling(7).std()

    return daily_df.dropna()


def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)