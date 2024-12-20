import requests

def send_result(result):
    response = requests.get(f"http://127.0.0.1:8000/update_result_list?result={result}")

    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Failed:", response.status_code)


send_result(['000001', '000002', '000003', '000004', '000005'])