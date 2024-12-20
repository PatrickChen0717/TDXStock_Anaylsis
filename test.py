import pandas as pd
from pytdx.hq import TdxHq_API
from collections import OrderedDict
import matplotlib.pyplot as plt
import datetime
import time
import json
import os

pd.set_option('display.max_rows', 10000)

api = TdxHq_API(multithread=True)

res_path = "res"
time_data = []
price_data = []

# Create a figure and axis for the graph
fig, ax = plt.subplots()

# Set initial limits for the x-axis (time) to show only recent data
ax.set_xlim(datetime.datetime.now() - datetime.timedelta(seconds=60), datetime.datetime.now())
ax.set_ylim(0, 100)  # Replace with appropriate y-axis limits

# Create a line plot
line, = ax.plot(time_data, price_data)

# def save_data(time_data, price_data):
    
def get_log_data(code, name):
    log_dir = f"{res_path}/{code}_{name}" 
    log_file = f"{log_dir}/price_log_{code}.txt" 

    with open(log_file, 'r') as file:
        data = json.load(file)

    price_list = [item[0] for item in data]
    time_list = [item[1] for item in data]

    return price_list, time_list

def plot_graph(code, name):
    while True:
        # Get current time and price data (replace this with your data source)
        price_data, time_data = get_log_data(code, name)

        # Update the line plot data
        line.set_data(time_data, price_data)

        # Adjust x-axis limits to show only recent data
        ax.set_xlim(min(time_data), max(time_data))
        # ax.set_xlim(current_time - datetime.timedelta(seconds=60), current_time)
        
        y_min = min(price_data)
        y_max = max(price_data)
        y_margin = 0.5  # Add a small margin for better visualization
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        # Save the updated graph to a local file
        plt.savefig(f"{res_path}/{code}_{name}/price_graph_{code}_{name}.png")
        plt.title(f"{name}  {code}")
        time.sleep(1)





