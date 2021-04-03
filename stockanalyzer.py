# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import finnhub
import numpy as np
import datetime
import pandas as pd
import json
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import math


#function: converts date (MM/DD/YY) to datetime string to insert into finhub API query
def convertDateToTimestamp(date_string):
    
    date = datetime.datetime.strptime(date_string, "%m/%d/%Y")

    timestamp = datetime.datetime.timestamp(date)

    return int(timestamp)


# Configure API key
configuration = finnhub.Configuration(
    api_key={
        'token': 'bs1pr97rh5rbe4rksjkg' # Replace this
    }
)

#initialize finhub client
finnhub_client = finnhub.DefaultApi(finnhub.ApiClient(configuration))

#set dates to obtain stock_candle data
fromTimeStamp = convertDateToTimestamp("1/1/2012")
toTimeStamp = convertDateToTimestamp("7/12/2020")

#Get Stock candle data from Apple within specified date range
stock_candles = finnhub_client.stock_candles('AAPL', 'D', fromTimeStamp, toTimeStamp).to_dict()
stock_candles = json.dumps(stock_candles, indent=4)


#initialize dataframe to organize data
df = pd.read_json(stock_candles)

#filter array to only contain opening, high, low, and current
df = df.filter(items=['o'])

data = df.values



#normalize data to fit range between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data)

#set training data to 80% of the data collected from api
training_data_len = math.ceil(len(data) *.8)

traindata = scaled_data[0:training_data_len, :]


x_train = []

y_train = []


for i in range(60, len(traindata)):
    x_train.append(traindata[i-60:i,0])
    y_train.append(traindata[i,0])
   

#convert training data to numpy array inorder to reshape data      
x_train = np.array(x_train)
y_train = np.array(y_train)

#reshape data
# number of samples(rows) number of timesteps(columns) and number of features(opening price)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Build the LSTM Model

model = Sequential()

#build lstm model with 50 nodes, it will return the sequences for another layer, and the shape is the number of timesteps(columns) and #number of features(opening price)
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


#compile model
model.compile(optimizer='adam', loss='mean_squared_error')


#train the model total number of training examples within a batch, epochs, number of iterations when a entire data set is passed forward and backwards through a dataset
model.fit(x_train, y_train, batch_size=1,epochs=1 )


#Create the testing dataset
#create a new array containing scaled values from the remaining 20%
test_data = scaled_data[training_data_len - 60:, :]

#Create the data sets x_test and y_test
x_test = []
y_test = data[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

#Get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean squared error
rmse=np.sqrt(np.mean(((predictions-y_test)**2)))

print (rmse)

train = df[:training_data_len]
valid = df[training_data_len:]
valid['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Open Price, (USD)', fontsize=18)

plt.plot(train['o'])
plt.plot(valid[['o','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show