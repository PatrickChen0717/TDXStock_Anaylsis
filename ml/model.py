import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
    
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

from datetime import datetime
import math

input_size = 120


def train_rf():
    csv_path = 'ml/historical_stock_data_vol.csv'
    df = pd.read_csv(csv_path)

    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)

    X_price = []
    X_vol = []
    y = []

    for index, row in df_price.iterrows():
        X_day_price = row.iloc[:40].values
        y_day = row.iloc[40:].values
        X_price.append(X_day_price)
        y.append(y_day)

    for index, row in df_vol.iterrows():
        X_day_vol = row.iloc[:40].values
        X_vol.append(X_day_vol)

    X = np.hstack((np.array(X_price), np.array(X_vol)))
    
    # Normalizing the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array(y)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'random_state': [50],
        'max_depth': [10, 20, 30, None]  # Added max_depth to the parameter grid
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')

    return grid_search.best_estimator_

def train_rf_rate():
    csv_path = 'ml/historical_stock_data_vol.csv'
    df = pd.read_csv(csv_path)

    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)

    X_price = []
    X_vol = []
    y = []

    for index, row in df_price.iterrows():
        X_day_price = row.iloc[:40].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).iloc[1:].values  # Drop the first value
        y_day = row.iloc[40:].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).iloc[1:].values  # Drop the first value
        if len(X_day_price) == 39 and len(y_day) > 0:
            X_price.append(X_day_price)
            y.append(y_day)

    for index, row in df_vol.iterrows():
        X_day_vol = row.iloc[:40].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).iloc[1:].values  # Drop the first value
        if len(X_day_vol) == 39:
            X_vol.append(X_day_vol)

    X = np.hstack((np.array(X_price), np.array(X_vol)))
    X = X.astype(np.float64)
    
    # Normalizing the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array(y)

    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_features': ['sqrt'],
        'random_state': [50]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=tscv, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best score: {grid_search.best_score_}')

    return grid_search.best_estimator_, scaler


def train_rf_voting():
    csv_path = 'ml/historical_stock_data.csv'

    df = pd.read_csv(csv_path)
    X = []
    y = []

    for index, row in df.iterrows():
        X_day = row.iloc[:40].values
        y_day = row.iloc[40:].values
        
        # Append to your lists
        X.append(X_day)
        y.append(y_day)

    # Convert your lists to NumPy arrays
    X = np.array(X) * 100
    # X = smooth_seqence(X)
    y = np.array(y) * 100

    print(X)
    print(y)

    models_for_each_output = []
    for i in range(y.shape[1]):  # assuming y has shape (485, 200)
        print(f"Training ensemble for output column {i+1}: {datetime.now()}")  # Log start time for each model
        y_single_output = y[:, i]

        rf = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
        lr = LinearRegression()
        svr = SVR(kernel='linear')

        ensemble_model = VotingRegressor(estimators=[('rf', rf), ('lr', lr), ('svr', svr)])
        ensemble_model.fit(X, y_single_output)

        models_for_each_output.append(ensemble_model)

    print(f"End Time: {datetime.now()}")  # Log end time
    return models_for_each_output



def train_nn():
    csv_path = 'ml/historical_stock_data.csv'
    
    df = pd.read_csv(csv_path)
    X = []
    y = []

    for index, row in df.iterrows():
        X_day = row.iloc[:120].values  # Adjust indices as needed
        y_day = row.iloc[120:].values  # Adjust indices as needed
        X.append(X_day)
        y.append(y_day)

    X = np.array(X)
    y = np.array(y)
    
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM
    y = np.reshape(y, (y.shape[0], y.shape[1]))  # Already okay for output layer

    scaler_x = MinMaxScaler()
    X_scaled = np.array([scaler_x.fit_transform(x) for x in X])

    scaler_y = MinMaxScaler()
    y_scaled = np.array([scaler_y.fit_transform(y.reshape(-1, 1)) for y in y])
    
    joblib.dump(scaler_x, 'scaler_x.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    model = tf.keras.Sequential([
        LSTM(128, activation='relu', input_shape=(120, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        LSTM(64, activation='relu', return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        LSTM(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(120)
    ])

    model.compile(optimizer='adam', loss='mse')

    # Learning rate scheduler
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * math.exp(-0.1)

    lr_schedule = LearningRateScheduler(scheduler)
    
    # Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    
    history = model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, 
                        validation_split=0.2, verbose=1, 
                        callbacks=[early_stopping, lr_schedule])

    loss = model.evaluate(X_scaled, y_scaled)
    print(f'Model Loss: {loss}')
    
    model.save("trained_stock_lstm_model.h5")

    return model


def test_prediction(clf, csv_path):
    df = pd.read_csv(csv_path)
    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)

    X_price_input = []
    X_vol_input = []
    y_expected = [] 

    for index, row in df_price.iterrows():
        X_day_price = row.iloc[:40].values
        y_day_expected = row.iloc[40:].values
        X_price_input.append(X_day_price)
        y_expected.append(y_day_expected)
    
    for index, row in df_vol.iterrows():
        X_day_vol = row.iloc[:40].values
        X_vol_input.append(X_day_vol)

    X_input = np.hstack((np.array(X_price_input), np.array(X_vol_input)))

    y_predicted = clf.predict(X_input)
    print("Test Mean Squared Error:", mean_squared_error(y_expected, y_predicted))

    # Make sure data_plot function exists and works as expected
    data_plot(X_price_input, y_expected, y_predicted)

def test_prediction(clf, csv_path):
    df = pd.read_csv(csv_path)
    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)

    X_price_input = []
    X_vol_input = []
    y_expected = [] 

    for index, row in df_price.iterrows():
        X_day_price = row.iloc[:40].pct_change().dropna().values  # Calculate growth rate
        y_day_expected = row.iloc[40:].pct_change().dropna().values  # Calculate growth rate
        if len(X_day_price) > 0 and len(y_day_expected) > 0:
            X_price_input.append(X_day_price)
            y_expected.append(y_day_expected)

    for index, row in df_vol.iterrows():
        X_day_vol = row.iloc[:40].pct_change().dropna().values  # Calculate growth rate
        if len(X_day_vol) > 0:
            X_vol_input.append(X_day_vol)

    X_input = np.hstack((np.array(X_price_input), np.array(X_vol_input)))

    # Assuming you've also scaled X_input as you did in the training function
    scaler = StandardScaler()
    X_input = scaler.fit_transform(X_input)

    y_predicted = clf.predict(X_input)
    print("Test Mean Squared Error:", mean_squared_error(y_expected, y_predicted))

    # Make sure data_plot function exists and works as expected
    data_plot(X_price_input, y_expected, y_predicted)


def test_prediction_rate(models, csv_path, ref_scaler):
    df = pd.read_csv(csv_path)
    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)

    X_price_input = []
    X_vol_input = []
    first_prices = []  # List to hold the first actual price for each sequence
    y_expected = [] 

    for index, row in df_price.iterrows():
        first_price = row.iloc[0]  # Store the first actual price
        first_prices.append(first_price)

        X_day_price = row.iloc[:40].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).iloc[1:].values
        y_day_expected = row.iloc[40:].values
        if len(X_day_price) == 39 and len(y_day_expected) > 0:
            X_price_input.append(X_day_price)
            y_expected.append(y_day_expected)
    
    for index, row in df_vol.iterrows():
        X_day_vol = row.iloc[:40].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0).iloc[1:].values  # Drop the first value
        if len(X_day_vol) == 39:
            X_vol_input.append(X_day_vol)

    X_input = np.hstack((np.array(X_price_input), np.array(X_vol_input)))

    # Use the same scaler object that you used for your training data
    X_input = ref_scaler.transform(X_input)
    y_predicted_rates = models.predict(X_input)

    y_predicted_price = []

    X_price_converted = []
    for fp, rates in zip(first_prices, X_input):  # half defined as before
        prices = [fp]
        for rate in rates:
            next_price = prices[-1] * (1 + rate)
            prices.append(next_price)
        X_price_converted.append(prices[1:])

    for fp, rates in zip(X_price_converted[-1], y_predicted_rates):
        prices = [fp]
        for rate in rates:
            next_price = prices[-1] * (1 + rate)
            prices.append(next_price)
        y_predicted_price.append(prices[1:])

    y_expected_sliced = [ye[1:] for ye in y_expected] 
    print("Test Mean Squared Error:", mean_squared_error(y_expected_sliced, y_predicted_price))
    full_array = np.concatenate((X_input, np.array(y_expected)), axis=1)  # Assuming that axis=1 is appropriate
    full_pred_array = np.concatenate((X_input, np.array(y_predicted_price)), axis=1)  # Assuming that axis=1 is appropriate
    
    # Assuming data_plot plots actual vs predicted sequences
    data_plot(X_price_converted, y_expected, y_predicted_price)

def data_plot_rate(X_input_array, X_expt_array, X_output):
    # Make sure the arrays are numpy arrays
    X_input_array = np.array(X_input_array)
    X_expt_array = np.array(X_expt_array)
    X_output = np.array(X_output)

    # If the arrays are 1D, expand dimensions to make them 2D
    if len(X_input_array.shape) == 1:
        X_input_array = np.expand_dims(X_input_array, axis=0)
    if len(X_expt_array.shape) == 1:
        X_expt_array = np.expand_dims(X_expt_array, axis=0)
    if len(X_output.shape) == 1:
        X_output = np.expand_dims(X_output, axis=0)

    # Concatenate arrays along the last axis (horizontally)
    full_array = np.concatenate((X_input_array, X_expt_array), axis=-1)
    full_pred_array = np.concatenate((X_input_array, X_output), axis=-1)

    # full_array = smooth_seqence(full_array)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot real data
    plt.plot(full_array[0], label='Real Data', marker='o')

    # Plot predicted data
    plt.plot(full_pred_array[0], label='Predicted Data', marker='x')

    plt.xlabel('Time Point')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def data_plot(X_input_array, X_expt_array, X_output):
    full_array = np.hstack((X_input_array, X_expt_array))
    full_pred_array = np.hstack((X_input_array, X_output))
    print(full_array)
    print(full_pred_array)
    # full_array = smooth_seqence(full_array)
    # full_pred_array = smooth_seqence(full_pred_array)
    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot real data (concatenation of input and expected output)
    plt.plot(full_array[0], label='Real Data', marker='o')  # Taking the first row as an example

    # Plot predicted data (concatenation of input and predicted output)
    plt.plot(full_pred_array[0], label='Predicted Data', marker='x')  # Taking the first row as an example

    plt.xlabel('Time Point')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def smooth_seqence(X):
    smoothed_X = []
    
    for sequence in X:
        smoothed_sequence = np.convolve(sequence, np.ones(2)/2, mode='valid')
        smoothed_X.append(smoothed_sequence)

    # Convert back to NumPy array
    smoothed_X = np.array(smoothed_X)
    return smoothed_X

if __name__ == '__main__':
    # random_forest_vote = train_rf_voting()
    # joblib.dump(random_forest_vote, 'ml/random_forest_vote')
    # random_forest_vote = joblib.load('ml/random_forest_vote')


    # test_prediction_voting(random_forest_vote, 'ml/test_data2.csv')

    # rf_model_rate, ref_scaler = train_rf_rate()
    # joblib.dump(rf_model_rate, 'ml/rf_model_rate')
    # joblib.dump(ref_scaler, 'ml/ref_scaler.pkl')
    rf_model_rate = joblib.load('ml/rf_model_rate')
    ref_scaler = joblib.load('ml/ref_scaler.pkl')
    test_prediction_rate(rf_model_rate, 'ml/test_data2.csv', ref_scaler)

    # nn_model = train_nn()
    # test_prediction(nn_model, 'ml/test_data1.csv')

