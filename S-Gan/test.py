import os
import re
import time
import csv
import random
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from datetime import date, timedelta, datetime
from urllib.request import urlopen
import pandas as pd
import numpy as np
from time import sleep
from random import randint
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import yfinance as yf
import quandl

start_date = "2020-08-26"
end_date = "2023-08-26"

nyse = yf.download("^NYA", start=start_date, end=end_date, interval = "1d")

price_data = yf.download(tickers = 'AAPL',end=end_date, start=start_date, interval = "1d" ) 
print(nyse)