from plugin.basefilter import basefilter
import json


class risefilter(basefilter):
    code_list = []

    def __init__(self, code_list, rise_threshold = 5):
        super().__init__()
        self.code_list = code_list
        self.rise_threshold = rise_threshold
        self.note = 'Check if rise is greater than rise_threshold'

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
            rise = [item[2] for item in data if len(item) >= 3]
            last_rise = rise[-1] if rise else None
            
            if last_rise != None:
                if self.check_rise(last_rise) == True:
                    self.result.append(f"{code}_{name}")
                else:
                    if f"{code}_{name}" in self.result:
                        self.result.remove(f"{code}_{name}")
            else:
                if f"{code}_{name}" in self.result:
                    self.result.remove(f"{code}_{name}")
        print(f'Rise filter done: result = {len(self.result)}')

    def check_rise(self, rise):
        return (rise >= self.rise_threshold)