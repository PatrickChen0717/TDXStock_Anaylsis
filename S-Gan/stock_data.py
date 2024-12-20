import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from keras.layers import Dropout, Activation 
from keras import layers
from keras.callbacks import EarlyStopping
import keras
import time
import random
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objs as go 
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler 
import quandl

class stock_data():
    def __init__(self, ticket, num_article, finbert_model, tokenizer):
        self.ticket = ticket
        self.num_article = num_article
        self.finbert_model = finbert_model
        self.tokenizer = tokenizer

    def article_parser(self):
        '''
        This method gets as input a stock ticker, creates the www.marketwatch.com link of this stock
        and returns a dataframe with the last 17 articles' headers.
        '''
        company_ticker = self.ticket
        link_1 = 'https://www.marketwatch.com/investing/stock/'
        link_2 = '?mod=search_symbol'
        # Pasting the above 3 parts to create the URL
        final_link = link_1 + company_ticker + link_2


        page = requests.get(final_link)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find("div", class_="tab__pane is-active j-tabPane")
        articles = results.find_all("a", class_="link")

        headerList = ["ticker", "headline"]
        rows = []
        counter = 1
        df_headers = pd.DataFrame()

        for art in articles:
            if counter <= self.num_article:
                ticker = company_ticker
                title = art.text.strip()
                if title is None:
                        break
                rows.append([ticker, title])
                counter = counter + 1

            df_headers = pd.DataFrame(rows, columns=headerList)
            
        self.article_headers = df_headers
        return df_headers
    
    def MACD(self, ticker):
        '''
        This method calculates the MACD (Mean Average Convergence Divergence) for a stock. The decision of whether to buy or sell
        a stock when using this method, depends on the difference of two "lines". The 1st one is called "MACD" and is equal to the 
        difference between the Exponential Moving Average of the adjusted closing price of the last 12 days, and the Moving Average
        of the adjusted closing price of the last 26 days. The 2nd line is the 9 day moving average of the adj. closing price. 
        When MACD > 9 day M.A. it is considered that there is an uptrend, else, an downtrend. 
        At last, when MACD line crosses the 9 day M.A. from "above", a "Sell" signal is given, 
        while it crosses it from below, a "Buy" signal is given.
        '''
        if self.stock_data.status_getter(ticker) != "Open":
            return "Market Closed"
        else:
            data = self.stock_data.stock_data_getter(ticker)

            low_high_closing_df = pd.DataFrame(data)
            low_high_closing_df = data.iloc[:, 4:5]  # Getting only the "Adj Close" column
            low_high_closing_df = low_high_closing_df.tail(52) # Getting the last 52 days


            # Get the 12-day EMA of the closing price
            low_high_closing_df['EMA_12d'] = low_high_closing_df['Adj Close'].ewm(span=12, adjust=False, min_periods=12).mean()
            # Get the 26-day MA of the closing price
            low_high_closing_df['MA_26d'] = low_high_closing_df['Adj Close'].ewm(span=26, adjust=False, min_periods=26).mean()
            # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
            low_high_closing_df['MACD'] = low_high_closing_df['EMA_12d'] - low_high_closing_df['MA_26d']
            # Making the signal line
            low_high_closing_df['MA_9d'] = low_high_closing_df['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()

            low_high_closing_df['Diff'] = low_high_closing_df['MACD'] - low_high_closing_df['MA_9d'] 

            Diff = low_high_closing_df['Diff'].astype(float)
            
            if self.stock_data.current_price_getter(ticker) is None:
                return "Market Closed"
            else:
                message = ""

                if Diff.iloc[-1] < 0:
                    if Diff.iloc[-2] >= 0:
                        message = "Downtrend and sell signal"
                    else:
                        message = "Downtrend and no signal"
                else:
                    if Diff.iloc[-2] <= 0:
                        message = "Uptrend and buy signal"
                    else:
                        message = "Uptrend and no signal"
                return message
            
    def LSTM_stock_data_getter(self, Ticker):
        '''
        This method will return a dataframe containing Stock data from the Yahoo's "yfinance" 
        library regardrless of whether the market is open or not, and will feed the LSTM model.
        '''
        # data = yf.download(tickers = str(Ticker), period = "2y", interval = "1d")
        
        start_date = "2016-08-26"
        end_date = "2023-08-26"
        price_data = yf.download(tickers = str(Ticker),end=end_date, start=start_date, interval = "1d" ) 
        nyse = yf.download("^NYA", start=start_date, end=end_date, interval = "1d")
        nasdaq = yf.download("^IXIC", start=start_date, end=end_date, interval = "1d")
        sp500 = yf.download("^GSPC", start=start_date, end=end_date, interval = "1d")

        df_price = pd.DataFrame(price_data)
        df_nyse = pd.DataFrame(nyse)
        df_nasdaq = pd.DataFrame(nasdaq)
        df_sp500 = pd.DataFrame(sp500)

        # List of DataFrames to concatenate
        all_data = pd.concat([price_data['Adj Close'], nyse['Adj Close'], nasdaq['Adj Close'], sp500['Adj Close']], axis=1)
        all_data.columns = ['Price', 'NYSE', 'NASDAQ', 'S&P500']
        
        return all_data.dropna()
                        
        # Prediction in the data we evaluate the model
        # If the user wants to run the model with the data that has been evaluated and predicted for , uncomment the 2 lines below 
        # Setting the start = 2022-08-26 and end = 2020-08-26 Yahoo Finance will return data from 25-8-2020 to 25-8-2022 (2 years period).
        # In those data our model has been evaluated.

        #data = yf.download(tickers = str(Ticker),end="2022-08-26", start="2020-08-26") 
        #df = pd.DataFrame(data)
        
        return final_df
            
    def LSTM_7_days_price_predictor(self, ticker):
            '''
            This method predicts the price of a chosen stock for the next 7 days as of today, by using the daily adjusted closing 
            prices for the last 2 years. At first, a 60-day window of historical prices (i-60) is created as our feature data (x_train)
            and the following 60-days window as label data (y_train). For every stock available, we have manually defined different 
            parameters so that they fit as good as it gets to the model. Finally we combute the R2 metric and make the predictions. At 
            last, we proceed with the predictions. The model looks back in our data (60 days back) and predicta for the following 7 days.
            '''

            stock_data = self.LSTM_stock_data_getter(ticker)
            print(stock_data)
            # stock_data=pd.DataFrame(data=stock_data).drop(['Open','High','Low','Close', 'Volume'],axis=1).reset_index()
            stock_data=pd.DataFrame(data=stock_data).reset_index()
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data=stock_data.dropna()

            # Data Preprocessing
            random.seed(1997)
            close_prices = stock_data
            values = close_prices.values
            training_data_len = math.ceil(len(values)* 0.8)

            scaler = MinMaxScaler(feature_range=(0,1))
            stock_data_numeric = stock_data.select_dtypes(include=['float64', 'int'])
            scaled_data = scaler.fit_transform(stock_data_numeric)
            train_data = scaled_data[0: training_data_len, :]

            x_train = []
            y_train = []

            for i in range(60, len(train_data)):
                x_train.append(train_data[i-60:i, :]) # Changed 0 to : to include all columns
                y_train.append(train_data[i, 0])

            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))

            # Preparation of test set
            test_data = scaled_data[training_data_len-60: , : ]
            x_test = []
            y_test = values[training_data_len:]

            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, :])
            x_test = np.array(x_test)
            # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            ##### Setting Up LSTM Network Architecture and the Training of the LSTM Model
            def LSTM_trainer(seed, DROPOUT, LSTM_units,patience,batch_size, epochs):

                tf.random.set_seed(seed)
                DROPOUT = DROPOUT
                global model_lstm
                model_lstm = keras.Sequential()
                model_lstm.add(layers.LSTM(LSTM_units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                model_lstm.add(Dropout(rate=DROPOUT))
                model_lstm.add(layers.LSTM(LSTM_units, return_sequences=False))
                model_lstm.add(Dropout(rate=DROPOUT))
                model_lstm.add(layers.Dense(50))
                model_lstm.add(Dropout(rate=DROPOUT))
                model_lstm.add(layers.Dense(25))
                model_lstm.add(Dropout(rate=DROPOUT))
                model_lstm.add(layers.Dense(4))
                model_lstm.add(Activation('linear'))
            
                print('\n')
                print("Compiling the LSTM Model for the "  + str(ticker) + " stock....\n")
                t0 = time.time()
                model_lstm.compile(optimizer='adam', loss='mean_squared_error',metrics=['mae'])
                callback=EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=patience,
                                verbose=1, mode='auto')
                model_lstm.fit(x_train, 
                            y_train,
                            batch_size= batch_size, 
                            epochs=epochs,
                            verbose=1,
                            validation_split=0.1,# ...holding out 10% of the data for validation 
                            shuffle=True,callbacks=[callback])
                t1 = time.time()
                global ex_time
                ex_time = round(t1-t0, 2)
                print("Compiling took :",ex_time,"seconds")

                predictions = model_lstm.predict(x_test)
                print(pd.DataFrame(predictions))
                predictions = scaler.inverse_transform(predictions)
                #rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
                global r_squared_score 
                global rmse
                # r_squared_score = round(r2_score(y_test, predictions),2)
                # rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
                # print('Rmse Score: ', round(rmse),2)
                # print('R2 Score: ', r_squared_score)

            if ticker == 'AAPL':
                LSTM_trainer(1, 0.2, 128,2, 32, 1)
            elif ticker == 'NVDA':
                LSTM_trainer(2, 0.2, 100,2, 30, 50)    
            elif ticker == 'PYPL':
                LSTM_trainer(6, 0.2, 100,10,25, 30)
            elif ticker == 'MSFT':
                LSTM_trainer(4, 0.1, 80, 2,20, 40)
            elif ticker == 'TSLA':
                LSTM_trainer(5, 0.1, 120, 4,20, 25)
            elif ticker == 'AMZN':
                LSTM_trainer(6, 0.1, 120,2, 20, 25)    
            elif ticker == 'SPOT':
                LSTM_trainer(9, 0.2, 200,5, 20, 40)
            #elif ticker == 'TWTR' :
            #   LSTM_trainer(15, 0.2, 100,4,20, 40)
            elif ticker == 'UBER':
                LSTM_trainer(15, 0.2, 100,7,20, 40)
            elif ticker == 'ABNB':
                LSTM_trainer(15, 0.2, 120,8,20, 40)
            elif ticker == 'GOOG':
                LSTM_trainer(15, 0.2, 100,3,20, 25)

            # Unseen Predictions for the next 7 days 
            close_data = scaled_data
            look_back = 60

            def predict(num_prediction, model):
                prediction_list = close_data[-look_back:]

                for _ in range(num_prediction):
                    x = prediction_list[-look_back:]
                    print(np.array(x).shape)
                    x = x.reshape((1, look_back, x_test.shape[2]))
                    
                    out = model.predict(x)[0]
                    prediction_list = np.concatenate((prediction_list, out.reshape(1, -1)), axis=0)
                prediction_list = prediction_list[look_back-1:, :]

                return prediction_list
        
            def predict_dates(num_prediction):
                last_date = stock_data['Date'].values[-1]
                prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
                return prediction_dates

            num_prediction = 30
            

            forecast = predict(num_prediction, model_lstm)
            print(pd.DataFrame(forecast))
            forecast_dates = predict_dates(num_prediction)
        
            # plt.figure(figsize=(25,10))
            forecast = forecast.reshape(-1, 1)
            forecast_inverse = scaler.inverse_transform(forecast)[:, 0]
            print(pd.DataFrame(forecast_inverse))
        
            # Ploting the Actual Prices and the Predictions of them for the next 7 days
            base = stock_data['Date'].iloc[[-1]] # Here we create our base date (the last existing date with actual prices)
            testdata = pd.DataFrame(forecast)# Here we create a data frame that contains the prediction prices and an empty column for their dates
            testdata['Date'] = ""
            testdata.columns = ["Adj Close","Date"]
            testdata = testdata.iloc[1:,:]
            testdata["Label"] = "" # Let's add a column "Label" that would show if the respective price is a prediction or not
            testdata["Label"] = "Prediction"
            testdata = testdata[["Date", "Adj Close", "Label"]]

            date_list = [base + datetime.timedelta(days=x+1) for x in range(testdata.shape[0]+1)] 
            date_list = pd.DataFrame(date_list)
            date_list.columns = ["Date"]
            date_list.reset_index(inplace = True)
            date_list.drop(["index"], axis = 1, inplace = True)
            date_list.index = date_list.index + 1
            testdata.Date = date_list

            stock_data["Label"] = ""
            stock_data["Label"] = "Actual price"
            finaldf = pd.concat([stock_data,testdata], axis=0) # Here we concatenate the "testdata" and the original data frame "df" into a final one
            finaldf.reset_index(inplace = True)
            finaldf.drop(["index"], axis = 1, inplace = True)
            finaldf['Date'] = pd.to_datetime(finaldf['Date'])
        
            # plt.rcParams["figure.figsize"] = (25,10)
            #We create two different data frames, one that contains the actual prices and one that has only the predictions
            finaldfPredictions = finaldf.iloc[-num_prediction-1:] 
            finaldfActuals = finaldf.iloc[:-num_prediction]

            plot_1 = go.Scatter(
                x = finaldfActuals['Date'],
                y = finaldfActuals['Adj Close'],
                mode = 'lines',
                name = 'Historical Data (2 years)',
                line=dict(width=1,color='#3D9140'))
            plot_2 = go.Scatter(
                x = finaldfPredictions['Date'],
                y = finaldfPredictions['Adj Close'],
                mode = 'lines',
                name = f'{num_prediction}-day Prediction',
                line=dict(width=1,color="#EE3B3B"))
            plot_3 = go.Scatter(
                x = finaldfPredictions['Date'][:1],
                y = finaldfPredictions['Adj Close'][:1],
                mode = 'markers',
                name = 'Latest Actual Closing Price',
                line=dict(width=1))

            layout = go.Layout(
                title = f'Next {num_prediction} days stock price prediction of ' + str(ticker),
                xaxis = {'title' : "Date"},
                yaxis = {'title' : "Price ($)"}
            )
            fig = go.Figure(data=[plot_1, plot_2,plot_3], layout=layout)
            fig.update_layout(template='plotly_dark',autosize=True)
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
                annotations = [dict(x=0.5,
                                    y=0, 
                                    xref='paper',
                                    yref='paper',
                                    text="Current In Sample R- Squared : "  + " % \n", #str(r_squared_score*100)
                                    showarrow = False)],
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False)
                

                            )
            fig.add_annotation(x=0.5, 
                            y=0.05,
                            xref='paper',
                            yref='paper',
                            text="Current In Sample Root Mean Square Error : " + " % ", #  + str(round(rmse,2))
                            showarrow=False)

            return fig


model = AutoModelForSequenceClassification.from_pretrained("models", from_tf=False, config="config.json") 
tokenizer = AutoTokenizer.from_pretrained("models/tokenizer/")

s = stock_data('AAPL',20, model, tokenizer)
# articles_list  = s.article_parser()["headline"].tolist()


# clf = pipeline("text-classification", model=model, tokenizer=tokenizer) 
# outputs_list = clf(articles_list)
# sentiments = []

# for item in outputs_list:
#     sentiments.append(item["label"])

# sentiments_df = pd.DataFrame(sentiments)
# sentiments_df.rename(columns = {0:'sentiment'}, inplace = True)

# sentiments_df["sentiment"] = sentiments_df["sentiment"].apply(lambda x: 100 if x == "positive" else -100 if x=="negative" else 0)            
# sentiments_df["roll_avg"] = round(sentiments_df["sentiment"].rolling(5, min_periods = 1).mean(), 2)
# sentiments_df = sentiments_df.tail(12).reset_index()

# pd.options.plotting.backend = "plotly"

# fig = sentiments_df["roll_avg"].plot(title="Sentiment Analysis of the last 12 www.marketwatch.com articles about " + 'AAPL', 

# template="plotly_dark",
# labels=dict(index="12 most recent article headlines", value="sentiment  score (rolling avg. of window size 5)"))
# fig.update_traces(line=dict(color="#3D9140", width=3))
# fig.update_layout(yaxis_range=[-100,100])
# fig.update_layout(xaxis_range=[0,12])
# fig.update_layout(showlegend=False)
# fig.add_hline(y=0, line_width=1.5, line_color="black")

# current_sentiment = sentiments_df["roll_avg"].tail(1).values[0]
# print(current_sentiment)

figure = s.LSTM_7_days_price_predictor(s.ticket)
figure.show()