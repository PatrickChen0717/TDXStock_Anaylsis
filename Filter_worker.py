import importlib
import requests
import json
import threading

filter_worker = None

with open("codelist.txt", 'r') as file:
    code_list = json.load(file)

class Filter_worker():
    result = []

    def __init__(self, filter_list, active = False):
        self.filter_list = filter_list
        self.plugin_list = []
        self.active = active
        for code in code_list:
            self.result.append(code)

        self.run()

    def add_filter(self, filter_list):
        if self.filter_list != filter_list:
            self.active = False
            self.run()

        self.active = True
        for filter in filter_list:
            module = importlib.import_module(f"plugin.{filter}")
            FilterClass = getattr(module, filter)
            
            # Create an instance of the class
            filter_instance = FilterClass(code_list)
            print("loaded: ",filter_instance)
            self.plugin_list.append(filter_instance)

        self.filter_list == filter_list
        

    # def run(self):
    #     if self.active:
    #         for plugin in self.plugin_list:
    #             print(f'Running: {plugin}')
    #             plugin.run()
    #             self.result = [x for x in self.result if x in plugin.result]
    #         print(f'result len: {len(self.result)}')
    #     else:
    #         self.result = []
    #         for code in code_list:
    #             self.result.append(code)
    #         print(f"Filter reset: result len = {len(self.result)}")
    def run_plugin_wrapper(self, filter_worker_instance):
        filter_worker_instance.run()
    
    def run(self):
        if self.active:
            threads = []
            temp_result = []
            for plugin in self.plugin_list:
                thread = threading.Thread(target=self.run_plugin_wrapper, args=(plugin,))
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            common_result = set(self.plugin_list[0].result)
            for plugin in self.plugin_list:
                common_result &= set(plugin.result)

            self.result = common_result 
            print(f'result len: {len(self.result)}')
            
        else:
            self.result = []
            for code in code_list:
                self.result.append(code)
            print(f"Filter reset: result len = {len(self.result)}")


    def send_result(self):
        print(f'sent result len: {len(self.result)}')
        return self.result


filter_worker = Filter_worker([], active = False)