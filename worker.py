import multiprocessing
import concurrent.futures
from pytdx.hq import TdxHq_API
import json
import datetime
import time
import os
from concurrent.futures import ThreadPoolExecutor

res_path = "res"

def get_price(info):
    api = TdxHq_API(multithread=True)
    retries = 0
    while retries < 20:
        try:
            with api.connect('119.147.212.81', 7709, time_out=10):
                data = api.get_security_quotes(info)
            
            try:
                return data
            except TypeError:
                retries += 1
        except Exception as e:
            print(f"Error fetching price: {e}   {info}")
            retries += 1
    
    print(f"Error fetching price: {info}")
    return -1
    # api = TdxHq_API(multithread=True)
    # retries = 0
    # sleep_time = 1  # starting sleep time in seconds

    # while retries < 10:
    #     try:
    #         with api.connect('119.147.212.81', 7709, time_out=10) as connection:  # increased timeout to 10 seconds
    #             data = api.get_security_quotes(info)
    #         return data
    #     except (ConnectionResetError, TimeoutError) as e:  # Catching specific exceptions
    #         print(f"Retrying due to error: {e}")
    #         retries += 1
    #         time.sleep(sleep_time)
    #         sleep_time *= 2  # exponential backoff
    #     except Exception as e:
    #         print(f"Other error occurred: {e}   {info}")
    #         return -1

    # print(f"Max retries reached for {info}")
    # return -1

def add_log(info):
    if info == None:
        print(None)
        return

    for sub_dict in info:
        price = sub_dict['price']
        code = sub_dict['code']  
        log_dir = f"{res_path}/{code}"  # Directory for the log
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

        data.append(price)
        with open(log_file, 'w') as file:
            # print(code,"   ",price)
            json.dump(data, file)

def child_process(item): 
    try:
        # print("Hello from the child process!")
        data = get_price(item)
        # print("Child PID:", multiprocessing.current_process().pid)
        add_log(data)
    except Exception as e:
        print(f"Error in child process: {e}  {item}")

if __name__ == '__main__':
    file_path = 'codelist.txt'
    items = []     
    with open(file_path, 'r') as file:
        cnt = 0
        for line in file:
            #if(cnt < 100):
            line_list = line.strip().split('|')
            formatted_item = (int(line_list[0]), f'{int(line_list[1]):06d}')
            items.append(formatted_item)
                # cnt += 1

    split_lists = [items[i:i + 50] for i in range(0, len(items), 50)]
    # print(items)
    print("init ",datetime.datetime.now())
    # Create a process pool with a specified number of workers
    while(True):
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     executor.map(child_process, split_lists)
        with ThreadPoolExecutor() as executor:
            executor.map(child_process, split_lists)
  
    # while(True):
    #     child_process(items)
    #     print("repeat ",datetime.datetime.now())