import numpy as np 
import requests
import pandas
from keras.models import load_model
from tensorflow import keras
from keras.layers import Dense,LSTM
from keras import Sequential
from flask import render_template
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data.iloc[i + sequence_length]
        sequences.append((sequence, target))
    return sequences


def preprocessdata(stock):
    url = "https://apistocks.p.rapidapi.com/daily"
    querystring = {"symbol": "AAPL", "dateStart": "2020-01-01", "dateEnd": "2023-11-31"}

    headers = {
        "X-RapidAPI-Key": "d059e17c8emsh19a1196699dd447p158dbfjsnad67a36b667e",
        "X-RapidAPI-Host": "apistocks.p.rapidapi.com"
    }


    try:
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status() 
        data = response.json()
        data = pandas.DataFrame(data['Results'])


        # Extract relevant columns (e.g., 'date' and 'close')
        data = data[['Date', 'Close']]

        # Convert 'date' to datetime
        data['Date'] = pandas.to_datetime(data['Date'])

        # Sort the data by date
        data = data.sort_values(by='Date')

        # Set 'date' as the index
        data.set_index('Date', inplace=True)

        # Normalize the data using Min-Max scaling
        scaler = MinMaxScaler()
        data['Close'] = scaler.fit_transform(data[['Close']])

        # Split the data into training and testing sets
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]
        sequence_length = 10  # You can adjust this to your needs
        train_sequences = create_sequences(train_data, sequence_length)
        test_sequences = create_sequences(test_data, sequence_length)
        X_train, y_train = zip(*train_sequences)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = zip(*test_sequences)
        X_test, y_test = np.array(X_test), np.array(y_test)
        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu', return_sequences=True))
        model.add(LSTM(50, activation='relu'))  # You can add more LSTM layers as needed
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=15, batch_size=32)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        pred = model.predict([[int(stock)]])
        pred = scaler.inverse_transform(pred)
        return pred[0][0]
    except requests.exceptions.RequestException as e:
        print(e)
        return f'An error occurred: {str(e)}'


