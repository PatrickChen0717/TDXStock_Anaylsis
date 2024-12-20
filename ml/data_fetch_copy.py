# import akshare as ak
# import requests
# from datetime import datetime, timedelta

# csv_path = 'ml/historical_stock_data.csv'
# stock_symbol = "000001"

# def get_data(start_date, end_date):
#     stock_zh_a_hist_min_em_df = None
#     try:
#         # stock_zh_a_hist_min_em_df = ak.stock_zh_a_hist_min_em(
#         #                                 symbol=stock_symbol, 
#         #                                 start_date=start_date, 
#         #                                 end_date=end_date, 
#         #                                 period='1', 
#         #                                 adjust='')
#         stock_zh_a_hist_min_em_df = ak.stock_zh_a_daily(symbol='sh000300', period='1')
#         print(stock_zh_a_hist_min_em_df)
#     except requests.exceptions.RequestException as e:
#         print(f"An HTTP error occurred: {e}")
#     except requests.exceptions.JSONDecodeError as e:
#         print(f"Failed to decode JSON: {e}")
    

#     # stock_zh_a_hist_min_em_df.columns = ['Timestamp', 'Open', 'Close', 'High', 'Low', 'Volume', 'Turnover', 'Latest']
#     # stock_zh_a_hist_min_em_df.to_csv(csv_path, index=False)



# start_date = datetime(2023, 8, 30)
# end_date = start_date

# one_day = timedelta(days=1)

# date_strings = []

# current_date = start_date
# while current_date <= end_date:
#     # Create the string format for the current date
#     date_str = current_date.strftime("%Y-%m-%d")
#     start_date_str = f"{date_str} 09:30:00"
#     end_date_str = f"{date_str} 15:00:00"
    
#     print(f"start_date=\"{start_date_str}\", end_date=\"{end_date_str}\"")

#     get_data(start_date_str, end_date_str)

#     current_date += one_day


from pytdx.exhq import *
from pytdx.hq import *
from datetime import datetime, timedelta
import time

def get_cur_price(date):
    retries = 0
    while retries < 20:
        try:
            api_hq = TdxHq_API()
            api_hq = api_hq.connect('119.147.212.81', 7709)
            time.sleep(1)
            data = pd.DataFrame.from_dict(api_hq.get_history_minute_time_data(TDXParams.MARKET_SZ, "002560", date)).T
            return data
        except Exception as e:
            retries += 1

    return -1
        

def get_data_day(date, csv_path, init):
    data = get_cur_price(date)

    if data.empty:
        print("The DataFrame is empty.")
        return

    price_row = data.loc[['price']]
    volumn_row = data.loc[['vol']]
    df1 = pd.DataFrame(price_row)
    df2 = pd.DataFrame(volumn_row)
    print(df1)
    print(df2)

    # Write the first DataFrame to CSV
    print(f'init = {init}')

    df1.to_csv(csv_path, mode='a', header=False, index=False)
    df2.to_csv(csv_path, mode='a', header=False, index=False)

    # df = pd.read_csv(csv_path)
    # print(df)
    # time.sleep(1)
    
def get_train_data():
    start_date = datetime(2022, 11, 9)
    end_date = datetime(2023, 9, 1)

    one_day = timedelta(days=1)

    date_strings = []

    current_date = start_date
    while current_date <= end_date:
        # Create the string format for the current date
        date = int(current_date.strftime("%Y-%m-%d").replace('-',''))
        
        print(f"date = {date}")

        get_data_day(date, csv_path = 'ml/GAN/historical_stock_data_vol.csv', init = (start_date == current_date))

        current_date += one_day

def get_test_data():
    csv_path = 'ml/test_data4.csv'
    #csv_path = 'ml/historical_stock_data.csv'
    get_data_day(20230908, csv_path, True)

get_train_data()
# get_test_data()