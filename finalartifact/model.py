import pandas as pd
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


#testing out classes and model
import datagathering as dg
import dataprep as dp

raw = dg.gethist("AMZN", "20y")
data_X, data_Y = dp.prepdata(raw, 5, 5)

tdpy = 237 #237 trading days a year

train_X, train_Y = data_X[:tdpy * 18, :], data_Y[:tdpy * 18, :]
test_X, test_Y = data_X[tdpy * 19:tdpy * 20, :], data_Y[tdpy * 19:tdpy * 20, :]

spm = StockPredictionModel([train_X.shape[1], train_X.shape[2]], train_Y.shape[1])
spm.train(train_X, train_Y)
yhat = spm.predict(test_X)
print(yhat[:, 0])
plt.plot(test_Y[:, 0], label="atcual")
plt.plot(yhat[:, 4], label="prediction")

plt.legend()
plt.show()

