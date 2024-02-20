import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler

from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from pandas_datareader import data as pdr
import yfinance as yf


# Predicts the next day stock prices based on the last 60 days of stock prices.


# Load data
companies = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "JNJ",
    "NVDA",
    "BRK.B",
    "JPM",
    "WMT",
    "V",
    "PG",
    "UNH",
    "MA",
    "PYPL",
    "HD",
    "BAC",
    "CRM",
    "NFLX"
]


start = dt.datetime(2012,1,1)
end = dt.datetime(2024,1,1)


yf.pdr_override()

for company in companies:
    data = pdr.get_data_yahoo(company, start, end) # or y_symbols instead of company. y_symbols = ['SCHAND.NS', 'TATAPOWER.NS', 'ITC.NS']


    # Prepare data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

    prediction_days =  60

    # Training data
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    # number arrays    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


    # Build the model 
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    ''' Test the model accuracy on existing data '''

    # load test data
    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()

    test_data = pdr.get_data_yahoo(company, test_start, test_end) 


    actual_prices = test_data['Close'].values

    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make predictions on test data

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    # Plot the test predictions

    # black is actual, green is prediction
    plt.plot(actual_prices, color='black', label=f"Actual {company} price")
    plt.plot(prediction_prices, color='green', label=f"Predicted {company} price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()


    # Predict next day

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    print(f"Prediction: {prediction}")