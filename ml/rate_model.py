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



def train_rf_rate():
    csv_path = 'ml/historical_stock_data_vol.csv'
    df = pd.read_csv(csv_path)

    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)

    df_price = df_price * 100
    df_vol = df_vol * 100
    # df_price = df_price.apply(smooth_df, axis=1)
    # df_vol = df_vol.apply(smooth_df, axis=1)

    X_price = []
    X_vol = []
    y = []

    for index, row in df_price.iterrows():
        price_rate_arr = price_rate(row)
        price_rate_arr = smooth_sequence(price_rate_arr)
        X_day_rate = price_rate_arr[:39]
        y_day_rate = price_rate_arr[39:]
        # print(np.array(X_day_rate).shape)

        X_price.append(X_day_rate)
        y.append(y_day_rate)

    for index, row in df_vol.iterrows():
        X_day_rate = price_rate(row.iloc[:40])

        X_vol.append(X_day_rate)

    X_price = np.array(X_price)
    X_vol = np.array(X_vol)
    y = np.array(y)

    X = np.array(X_price)
    # X = np.hstack((X_price, X_vol))
    X = X.astype(np.float64)
    print(X.shape)
    print(y.shape)

    # Normalizing the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
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


def test_prediction_rate(models, csv_path, ref_scaler):
    df = pd.read_csv(csv_path)
    df_price = df.iloc[::2, :].reset_index(drop=True)
    df_vol = df.iloc[1::2, :].reset_index(drop=True)
    df_price = df_price * 100
    df_vol = df_vol * 100
    df_price = df_price.apply(smooth_df, axis=1)
    df_vol = df_vol.apply(smooth_df, axis=1)

    X_price_input = []
    X_vol_input = []
    first_prices = []  # List to hold the first actual price for each sequence
    y_expected = [] 
    full_prices = []

    for index, row in df_price.iterrows():
        first_price = row.iloc[0]  # Store the first actual price 
        first_prices.append(first_price)
        full_prices = row
        
        price_rate_arr = price_rate(row)
        X_day_price = price_rate_arr[:39]
        y_day_expected = price_rate_arr[39:]

        X_price_input.append(X_day_price)
        y_expected.append(y_day_expected)
    
    for index, row in df_vol.iterrows():
        X_day_vol = price_rate(row.iloc[:40]) 
        X_vol_input.append(X_day_vol)
    
    print(f"X price shape : {np.array(X_price_input).shape}")
    print(f"X price shape : {np.array(X_vol_input).shape}")
    X_input = np.array(X_price_input)
    # X_input = np.hstack((np.array(X_price_input), np.array(X_vol_input)))
    print(f"X input shape : {X_input.shape}")
    

    X_input = ref_scaler.transform(X_input)
    y_predicted_rates = models.predict(X_input) 
    print(f"y_expected size : {np.array(y_expected).shape}")
    print(f"y_predicted_rates size : {np.array(y_predicted_rates).shape}")

    X_price_converted = []

    # print("Test Mean Squared Error:", mean_squared_error(y_expected_sliced, y_predicted_price))
    X_price_converted = rate_to_price(np.array(X_price_input[0]), first_price, 1)
    print(X_price_converted)
    print("X_price_converted:",np.array(X_price_converted).shape)
    
    first = X_price_converted[-1]

    y_predicted_price = rate_to_price(np.array(y_predicted_rates[0]), first, 2)
    print("y_predicted_price:",np.array(y_predicted_price).shape)
    # Assuming data_plot plots actual vs predicted sequences
    data_plot(X_price_converted, full_prices, y_predicted_price)

def data_plot(X_input_array, full_prices, X_output):
    full_pred_array = X_input_array + X_output
    full_prices_smoothed = smooth_sequence(full_prices)
    full_pred_array = smooth_sequence(full_pred_array)
    # Plotting
    print("full_prices:", np.array(full_prices).shape)
    print("full_pred_array:", np.array(full_pred_array).shape)
    plt.figure(figsize=(12, 6))

    # Plot real data (concatenation of input and expected output)
    plt.plot(full_prices, label='Real Data', alpha=0.5, linewidth=1)
    plt.plot(full_prices_smoothed, label='Real Data smoothed', linewidth=1)

    # Plot predicted data (concatenation of input and predicted output)
    plt.plot(full_pred_array, label='Predicted Data', linewidth=1)

    plt.xlabel('Time Point')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def price_rate(price_row):
    rate_list = []
    i = 0

    while i < len(price_row)-1:
        rate = price_row[i+1] - price_row[i]
        rate_list.append(rate)

        i+=1
    
    if len(rate_list) != len(price_row)-1:
        print(f'Error price rate len {len(rate_list)}')

    return rate_list

def rate_to_price(rate_list, first_price, tri):
    price_list = []
    if tri == 1:
        price_list.append(first_price)

    current_price = first_price
    
    for rate in rate_list:
        next_price = current_price + rate
        price_list.append(next_price)
        current_price = next_price

    return price_list


def smooth_sequence(X):
    return np.convolve(X, np.ones(5)/5, mode='valid')

def smooth_df(row):
    smoothed_row = np.convolve(row, np.ones(2)/2, mode='valid')
    return pd.Series(np.append(smoothed_row, np.nan))  

if __name__ == '__main__':
    # rf_model_rate, ref_scaler = train_rf_rate()
    # joblib.dump(rf_model_rate, 'ml/rf_model_rate')
    # joblib.dump(ref_scaler, 'ml/ref_scaler.pkl')
    rf_model_rate = joblib.load('ml/rf_model_rate')
    ref_scaler = joblib.load('ml/ref_scaler.pkl')
    test_prediction_rate(rf_model_rate, 'ml/test_data2.csv', ref_scaler)
