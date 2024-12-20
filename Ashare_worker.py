from  Ashare.Ashare import *   
from  Ashare.MyTT import *
import akshare as ak
import os
import datetime
import time

# df = get_price('sh600519',frequency='1m',count=5)  
# print(df)
res_path = "res"
code_list = []

def get_cur_price():
    retries = 0
    while retries < 20:
        try:
            stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()
            info = stock_zh_a_spot_em_df[['代码','名称','最新价','涨跌幅','成交量','成交额']]
            return info
        except Exception as e:
            retries += 1

    return -1
        

def add_cur_log(info):
    if info.empty:
        print("Fail to retrieve data (None)")
        return

    for index, row in info.iterrows():
        name = row['名称'].replace("*", "")
        code = row['代码'] 

        price = row['最新价']
        rise = row['涨跌幅']
        trade_volume = row['成交量']
        trade_value = row['成交额']
        # open_price = row['开盘']

        # print(name)
        if(code not in code_list):
            code_list.append(f"{code}_{name}")

        log_dir = f"{res_path}/{code}_{name}"  # Directory for the log
        log_file = f"{log_dir}/price_log_{code}.txt"  # File path for the log
        
        # Create directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        data = []
        try:
            with open(log_file, 'r') as file:
                data = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        new_entry = (price, current_time, rise, trade_volume, trade_value)
        data.append(new_entry)

        with open(log_file, 'w') as file:
            json.dump(data, file)


if __name__ == '__main__':
    print('Init ',datetime.datetime.now())
    while True:
        info = get_cur_price()
        add_cur_log(info)
        print('Repeat ',datetime.datetime.now())



