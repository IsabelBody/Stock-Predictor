import numpy as np
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from pandas_datareader import data as pdr
import yfinance as yf

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

start = dt.datetime(2012, 1, 1)
end = dt.datetime.now()

yf.pdr_override()

predictions = []

for company in companies:
    data = pdr.get_data_yahoo(company, start, end)

    # Prepare data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    prediction_days = 60

    # Training data
    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x - prediction_days:x, 0])
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
    model.add(Dense(units=1))  # Prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Predict next day
    real_data = scaled_data[-prediction_days:].reshape(1, -1, 1)
    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    predictions.append({'Company': company, 'Prediction': prediction[0][0]})

