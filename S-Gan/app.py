import yfinance as yf
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import nltk
import string
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from urllib.parse import urlencode
from datetime import datetime, timedelta

nltk.download('punkt')
nltk.download('stopwords')

def NLTK_textproc(headline_text):
    tokens = nltk.word_tokenize(headline_text.lower())

    # Remove punctuations and stopwords
    tokens = [word for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]

    print(tokens)

def finBERT_Sentiment_Analysis(headline_text):
    model = AutoModelForSequenceClassification.from_pretrained("models", from_tf=False, config="config.json") 
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer/")

    # Preprocess the text
    inputs = tokenizer(headline_text, padding=True, truncation=True, return_tensors="pt")

    # Get model output
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label = torch.argmax(outputs.logits, dim=1)

    print(predicted_label)

def url_getter(start_date, end_date):
    base_url = "https://seekingalpha.com/symbol/AAPL/news?"
    params = {
        'from': f"{start_date}T00:00:00.000Z",
        'to': f"{end_date}T23:59:59.999Z"
    }
    url = base_url + urlencode(params)
    return url

start_date = datetime.strptime("2022-09-16", "%Y-%m-%d")
end_date = datetime.strptime("2023-09-16", "%Y-%m-%d")

period = timedelta(days=60)

while start_date <= end_date:
    period_end_date = start_date + period - timedelta(days=1)
    
    if period_end_date > end_date:
        period_end_date = end_date
    
    url = url_getter(start_date.strftime("%Y-%m-%d"), period_end_date.strftime("%Y-%m-%d"))
    print("-------------", url) 
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    headlines = soup.select('article')
    for headline in headlines:
        print(headline.text)
        NLTK_textproc(headline.text)
        finBERT_Sentiment_Analysis(headline.text)

    
    start_date += period

# response = requests.get(url)
# with open('response.html', 'w', encoding='utf-8') as f:
#     f.write(response.text)
# soup = BeautifulSoup(response.text, 'html.parser')

# # Define the initial start_date and end_date


# # print(response.text)
# # Locate the news headlines on the page (this selector will depend on the actual HTML structure of the page)
# headlines = soup.select('article')
# # Extract and print out the headlines
# for headline in headlines:
#     print(headline.text)
#     # NLTK_textproc(headline.text)
#     # finBERT_Sentiment_Analysis(headline.text)

