'''
uvicorn fastapi_app:app --reload
http://127.0.0.1:8000
'''

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse
from typing import List, Optional
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import subprocess
import importlib
import json
import base64
import logging

import Ashare_worker
import grapher
import Filter_worker

app = FastAPI()

log_worker = 'Ashare_worker.py'
process = None

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

# In-memory database represented as a list
items_db = []


@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html", "r") as f:
        content = f.read()
    return content

@app.post("/start-log")
def start_log():
    print("Start Log button clicked")
    process = subprocess.Popen(['python', log_worker])

    return {"message": "Start Log button clicked"}

@app.on_event("shutdown")
def shutdown_event():
    if process != None:
        print('Terminating logging process ...')
        process.terminate()

@app.post("/clear-log")
def clear_log():
    grapher.clear_log()

@app.post("/filter")
async def filter(request: Request):
    filter_list = await request.json()
    Filter_worker.filter_worker.plugin_list = []
    # Create an instance of the class
    Filter_worker.filter_worker.add_filter(filter_list)
    Filter_worker.filter_worker.active = True
    Filter_worker.filter_worker.run()
    print(Filter_worker)

    return {"message": "filter button clicked"}

@app.get("/new-image-url")
async def get_new_image_url(file_name):
    graph_path = file_name
    
    if file_name != 'no_img.png':
        graph_path = f"res/{file_name}/price_graph_{file_name}.png"
        grapher.plot_graph(file_name.split('_')[0], file_name.split('_')[1])
    logging.info(graph_path)

    try:
        with open(graph_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image file not found")
    return JSONResponse({'image_base64': base64_image})


@app.get("/update_result_list")
async def get_new_image_url(result: List[str] = Query(None, alias="result")):
    if Filter_worker.filter_worker == None:
        print("Filter worker is None")
        return {"result": []}
    print(Filter_worker.filter_worker)
    return {"result": Filter_worker.filter_worker.send_result()}

@app.route('/res/<path:filename>')
def serve_file(filename):
    print(filename)
    # return send_from_directory('your_directory_path_here', filename)
