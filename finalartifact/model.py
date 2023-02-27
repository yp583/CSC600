import pandas as pd
import numpy as np
import datagathering as dg
import dataprep as dp

import matplotlib.pyplot as plt

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





#testing out classes and model
import datagathering as dg
import dataprep as dp

raw = dg.gethist("AMZN", "20y") #get amazon data from the last 20 years
data_X, data_Y = dp.prepdata(raw, 5, 5) #prep the raw data

tdpy = 237 #237 trading days a year

train_X, train_Y = data_X[:tdpy * 18, :], data_Y[:tdpy * 18, :] #train the model on the last 18 years
test_X, test_Y = data_X[tdpy * 19:tdpy * 20, :], data_Y[tdpy * 19:tdpy * 20, :] #train the model on 

spm = StockPredictionModel([train_X.shape[1], train_X.shape[2]], train_Y.shape[1])
spm.train(train_X, train_Y)
prediction = spm.predict(test_X)
error = spm.error(test_Y, prediction)
print(error)


plt.plot(test_Y[:, 0], label="atcual")
plt.plot(prediction[:, 4], label="prediction")

plt.legend()
plt.show()

