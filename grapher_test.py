import akshare as ak
import pandas as pd
import pandas as pd
from pytdx.hq import TdxHq_API
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np

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


############################################################################

# stock_zh_a_minute_df = ak.stock_zh_a_minute(symbol='sh600751', period='1', adjust="qfq")
stock_zh_a_hist_min_em = ak.stock_zh_a_hist_min_em(symbol="000001", start_date="2023-09-01 09:30:00", end_date="2023-09-01 15:00:00", period='1', adjust='')
# print(stock_zh_a_hist_min_em)
data = stock_zh_a_hist_min_em[['时间','开盘','成交量','成交额']]
data['时间'] = pd.to_datetime(data['时间'])

filtered_data = data[data['时间'].dt.strftime('%m-%d') == '09-01']

time = data['时间']
price = data['开盘']
value = data['成交额']
volume = data['成交量']

print(filtered_data)

# Create a line plot
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
    average_data = filtered_value_data / filtered_volume_data
    print(average_data)
    # Plot
    ax.plot(filtered_time, average_data, label='Price Average Data', alpha=1, color='green', linewidth=0.75)


def plot_smoothed(ax, time, price, window_size=10):
    price_smooth = np.convolve(price, np.ones(window_size)/window_size, mode='valid')
    time_smooth = time[(window_size - 1)//2:-(window_size//2)]  # Truncate time to match smoothed data
    ax.plot(time_smooth, price_smooth, label='Price Smoothed Data', alpha=0.5, color='red', linewidth=0.75)

def plot_points(ax, x_values, y_values):
    ax.scatter(x_values, y_values, label='Trough', alpha=1, color='green', s=10)

def plot_graph(code, name):
    # Get current time and price data (replace this with your data source)
    price_data = price
    time_data = time
    value_data = value
    volume_data = volume

    plot_original(ax, time, price)
    
    # Plot smoothed data
    plot_smoothed(ax, time, price)

    plot_average(ax, time, value_data, volume_data)

    is_filter, x_values, y_values = check_trough(price_data, time_data)
    plot_points(ax, x_values, y_values)

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
    plt.savefig(f"price_graph_{code}_{name}.png")


def check_trough(prices, times):
        y_values = []
        x_values = []

        for i in range(1, len(prices) - 1):
            if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                if prices.iloc[i] >= 0.2:
                    y_values.append(prices.iloc[i])
                    x_values.append(times.iloc[i])

        # Check if there are at least two troughs
        if len(y_values) >= 2:
            print("There are at least two troughs:")
            # for x, y in zip(x_values, y_values):
            #     print(f"Price: {y}, Time: {x}")

            # Check if the last trough is greater than the second last trough
            if y_values[-1] > y_values[-2]:
                print("The last trough is greater than the second last trough.")
                return True, x_values[-2:], y_values[-2:]
            else:
                print("The last trough is not greater than the second last trough.")
                return False, x_values[-2:], y_values[-2:]
        else:
            print("There are fewer than two troughs.")
            return False, [], []
            
# plot_graph('sh600751', 'Test')