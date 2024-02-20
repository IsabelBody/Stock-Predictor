import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
import yfinance as yf

# Function to get top n companies from Yahoo Finance
def get_top_companies(n):
    top_companies = pd.read_html('https://finance.yahoo.com/quote/%5EGSPC/components/')[0]
    return top_companies['Symbol'][:n].tolist()

# Predicts the next day stock prices based on the last 60 days of stock prices.

# Load data for top 20 companies
companies = get_top_companies(20)
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = {}
for company in companies:
    data[company] = yf.download(company, start=start, end=end)['Close']

# Prepare data
scalers = {}
scaled_data = {}
for company, prices in data.items():
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.values.reshape(-1, 1))
    scalers[company] = scaler
    scaled_data[company] = scaled_prices

prediction_days = 60

# Training data
x_train = {}
y_train = {}
for company, scaled_prices in scaled_data.items():
    x_train[company] = []
    y_train[company] = []
    for x in range(prediction_days, len(scaled_prices)):
        x_train[company].append(scaled_prices[x - prediction_days:x, 0])
        y_train[company].append(scaled_prices[x, 0])

# number arrays    
x_train = {company: np.array(values) for company, values in x_train.items()}
y_train = {company: np.array(values) for company, values in y_train.items()}
x_train = {company: np.reshape(values, (values.shape[0], values.shape[1], 1)) for company, values in x_train.items()}

# Build the model 
models = {}
for company in companies:
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train[company].shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Prediction of the next closing value
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train[company], y_train[company], epochs=25, batch_size=32)
    models[company] = model

''' Test the model accuracy on existing data '''

# load test data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

test_data = {}
for company in companies:
    test_data[company] = yf.download(company, start=test_start, end=test_end)['Close']

actual_prices = {company: prices.values for company, prices in test_data.items()}

total_dataset = pd.concat([pd.Series(prices.values.flatten()) for prices in test_data.values()], axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data[companies[0]]) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)

# Make predictions on test data
prediction_prices = {}
for company, model in models.items():
    scaler = scalers[company]
    model_inputs_company = model_inputs.copy()
    model_inputs_company = scaler.transform(model_inputs_company)
    x_test = []
    for x in range(prediction_days, len(model_inputs_company)):
        x_test.append(model_inputs_company[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    prediction_prices[company] = model.predict(x_test)
    prediction_prices[company] = scaler.inverse_transform(prediction_prices[company])

# Plot the test predictions
for company in companies:
    plt.plot(actual_prices[company], color='black', label=f"Actual {company} price")
    plt.plot(prediction_prices[company], color='green', label=f"Predicted {company} price")
    plt.title(f"{company} Share Price")
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()

# Predict next day
real_data = {company: [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]] for company in companies}
real_data = {company: np.array(values) for company, values in real_data.items()}
real_data = {company: np.reshape(values, (values.shape[0], values.shape[1], 1)) for company, values in real_data.items()}

prediction = {}
for company, model in models.items():
    prediction[company] = model.predict(real_data[company])
    prediction[company] = scalers[company].inverse_transform(prediction[company])
    print(f"Prediction for {company}: {prediction[company]}")
