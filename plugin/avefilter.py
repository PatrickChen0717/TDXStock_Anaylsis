from plugin.basefilter import basefilter
import json


class avefilter(basefilter):
    code_list = []
    threshold_percentage = 0.8

    def __init__(self, code_list):
        super().__init__()
        self.code_list = code_list
        self.note = 'Check average price is greater than current price'

    def check_code_num(self):
        with open("codelist.txt", 'r') as file:
            data = json.load(file)

        return len(data) == self.code_num
    
    def run(self):
        
        if self.check_code_num() == False:
            print("Error number of codes!")
            return

        for code_sample in self.code_list:
            code = code_sample.split('_')[0]
            name = code_sample.split('_')[1]

            log_dir = f"{self.res_path}/{code}_{name}"  # Directory for the log
            log_file = f"{log_dir}/price_log_{code}.txt"  # File path for the log

            try:
                with open(log_file, 'r') as file:
                    data = json.load(file)
            except json.JSONDecodeError as e:
                if f"{code}_{name}" in self.result:
                    self.result.remove(f"{code}_{name}")
                print(f"JSON Decode Error: {log_file}")

            '''
                All item lists are arranged in incrementing time order. 
                The final data in the item list represents the most recent value
            '''
            prices = [item[0] for item in data]
            trade_volume = [item[3] for item in data if len(item) >= 4]
            trade_value = [item[4] for item in data if len(item) >= 5]

            if trade_volume != [] and trade_value != []:
                if self.check_ave(prices, trade_volume, trade_value) == True:
                    self.result.append(f"{code}_{name}")
                else:
                    if f"{code}_{name}" in self.result:
                        self.result.remove(f"{code}_{name}")
            else:
                if f"{code}_{name}" in self.result:
                    self.result.remove(f"{code}_{name}")
        print(f'Average filter done: result = {len(self.result)}')

    def check_ave(self, prices, trade_volume, trade_value):
        ave_price = []
        for value, volume in zip(trade_value, trade_volume):
            if volume == 0:
                ave_price.append(0)
            else:
                ave_price.append(float(value) / float(volume) / 100)
        comparison = [p >= a for p, a in zip(prices, ave_price)]

        # Count the number of True values
        true_count = sum(comparison)
        true_percentage = true_count / len(comparison)
        
        return true_percentage >= self.threshold_percentage