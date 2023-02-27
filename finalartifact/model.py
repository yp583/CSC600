import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense

class StockPredictionModel:
    def __init__(self, dim_X, dim_Y):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(dim_X[0], dim_X[1]), activation="relu"))
        self.model.add(Dense(dim_Y))
    def train(self, train_X, train_Y): 
        self.model.compile(loss='mae', optimizer='adam')
        # fit network
        history = self.model.fit(train_X, train_Y, epochs=40, batch_size=72, verbose=2, shuffle=False)
        return history
    def predict(self, test_X):
        forecast = self.model.predict(test_X, verbose=0)
        return forecast
    def error(self, test_Y, prediction = None, test_X = None):
        error = [0] * len(test_Y[0,])
        if (not prediction.any()):
            prediction = self.predict(test_X)
        for i in range(len(test_Y[0,])):
            for j in range(len(test_Y)):
                currerror = prediction[j][i] - test_Y[j][i]
                currerror *= currerror #square the error
                error[i] += currerror
        return np.sqrt(error)






