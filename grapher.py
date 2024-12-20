import akshare as ak
import pandas as pd
import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
import time
import numpy as np
import json
import os


res_path = "res"



#################################################################################################

def get_log_data(code, name):
    log_dir = f"{res_path}/{code}_{name}" 
    log_file = f"{log_dir}/price_log_{code}.txt" 

    with open(log_file, 'r') as file:
        data = json.load(file)

    price_list = [item[0] for item in data]
    time_list = [item[1] for item in data]
    volume_list = [item[3] for item in data]
    value_list = [item[4] for item in data]

    return price_list, time_list, volume_list, value_list


def plot_original(ax, time, price):
    ax.plot(time, price, label='Price Original Data', alpha=1, color='#7777f7', linewidth=0.75)

def plot_average(ax, time, value_data, volume_data):
    # Convert to NumPy arrays for element-wise operations
    time = np.array(time)
    value_data = np.array(value_data)
    volume_data = np.array(volume_data)

    mask = volume_data != 0
    # Filter out zero-volume data points
    filtered_time = time[mask]
    filtered_value_data = value_data[mask]
    filtered_volume_data = volume_data[mask]

    # Check for zero division
    # if np.any(filtered_volume_data == 0):
    #     print("Warning: Division by zero detected.")
    #     return

    # Compute the average
    average_data = filtered_value_data / filtered_volume_data / 100
    # print(average_data)
    # Plot
    ax.plot(filtered_time, average_data, label='Price Average Data', alpha=1, color='green', linewidth=0.75)


def plot_smoothed(ax, time, price, window_size=10):
    price_smooth = np.convolve(price, np.ones(window_size)/window_size, mode='valid')
    time_smooth = time[(window_size - 1)//2:-(window_size//2)]  # Truncate time to match smoothed data
    ax.plot(time_smooth, price_smooth, label='Price Smoothed Data', alpha=0.5, color='red', linewidth=0.75)

def plot_points(ax, x_values, y_values):
    ax.scatter(x_values, y_values, label='Trough', alpha=1, color='green', s=10)

def plot_graph(code, name):
    fig, ax = plt.subplots()
    # Set initial limits for the x-axis (time) to show only recent data
    ax.set_xlim(dt.datetime.now() - dt.timedelta(seconds=60), dt.datetime.now())
    ax.set_ylim(0, 100)  # Replace with appropriate y-axis limits

    # Get current time and price data (replace this with your data source)
    price_data, time_data, volume_data, value_data  = get_log_data(code, name)
    time_data = [dt.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S') for time_str in time_data]

    plot_original(ax, time_data, price_data)
    
    # Plot smoothed data
    # plot_smoothed(ax, time_data, price_data)

    plot_average(ax, time_data, value_data, volume_data)

    # is_filter, x_values, y_values = check_trough(price_data, time_data)
    # plot_points(ax, x_values, y_values)

    # Adjust x-axis limits to show only recent data
    ax.set_xlim(min(time_data), max(time_data))
    # ax.set_xlim(current_time - datetime.timedelta(seconds=60), current_time)
    
    y_min = min(price_data)
    y_max = max(price_data)
    y_margin = (y_max - y_min)/2  # Add a small margin for better visualization
    ax.set_ylim(y_min - y_margin, y_max + y_margin)


    # Save the updated graph to a local file
    plt.legend()
    plt.title(f"{name}  {code}")
    plt.savefig(f"{res_path}/{code}_{name}/price_graph_{code}_{name}.png")

    plt.close(fig)


def clear_log():
    retries = 0
    info = None
    while retries < 20:
        try:
            stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
            info = stock_zh_a_spot_em_df[['代码','名称']]
            print(info)
            break
        except Exception as e:
            retries += 1

    # if info == None:
    #     print("Fail to retrieve data (None)")
    #     return

    for index, row in info.iterrows():
        name = row['名称'].replace("*", "")
        code = row['代码'] 
        
        log_dir = f"{res_path}/{code}_{name}"  # Directory for the log
        log_file = f"{log_dir}/price_log_{code}.txt"  # File path for the log

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(log_file, 'w') as file:
            json.dump([], file)
    
    print("Cleared log")