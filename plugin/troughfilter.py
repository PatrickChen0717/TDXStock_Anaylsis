from plugin.basefilter import basefilter
import json


class troughfilter(basefilter):
    code_list = []
    trough_threshold = 0.2

    def __init__(self, code_list):
        super().__init__()
        self.code_list = code_list
        self.note = 'The last trough is greater than the second last trough'
    
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
            times = [item[1] for item in data]

            if self.check_trough2(prices, times) == True:
                self.result.append(f"{code}_{name}")
            else:
                if f"{code}_{name}" in self.result:
                    self.result.remove(f"{code}_{name}")
        print(f'Trough filter done: result = {len(self.result)}')

    def check_trough(self, prices, times):
        troughs = []
        recent_peak = 0  # Initialize a variable to keep track of the most recent peak

        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                recent_peak = prices[i]  # Update the recent peak if current price is a peak

            is_trough = prices[i] < prices[i-1] and prices[i] < prices[i+1]
            satisfies_condition = prices[i] < 0.9 * recent_peak  # Check if the trough is < 90% of the recent peak

            if is_trough and satisfies_condition:
                troughs.append((prices[i], times[i]))

        if len(troughs) >= 2 and troughs[-1][0] > troughs[-2][0]:
            return True

        return False


    def check_trough2(self, prices, times):

        troughs = [(price, time) for i, price, time in zip(range(1, len(prices) - 1), prices[1:], times[1:])
                        if prices[i] < prices[i-1] and prices[i] < prices[i+1] and prices[i] >= self.trough_threshold]

        if len(troughs) >= 2 and troughs[-1][0] > troughs[-2][0]:
            return True
        return False
