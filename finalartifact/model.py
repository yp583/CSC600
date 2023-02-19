from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datagathering as dg
import dataprep as dp
import featureengineering as fe

import matplotlib.pyplot as plt

#get data and engineer feature and prep it for model
raw = dg.gethist("AAPL", "20y")
rawengineered = fe.getengineeredfeatures(raw)

scaler = MinMaxScaler()

#scale the data between 0 and 1 which makes the model learn better. Disparate numbers like volume vs RSI will cause weights to be skewed in the model.
data = pd.DataFrame(scaler.fit_transform(rawengineered), columns=rawengineered.columns)

preppeddata = dp.series_to_supervised(data, 7, 1) #train on the last 7 days to predict the next 1 days
preppeddata.drop(preppeddata.columns[range(85, 96)], axis = 1, inplace=True)

#split data into training and testing sets

vals = preppeddata.values

tdpy = 237 # ~237 trading days a year

train = vals[:tdpy*18, :]
test = vals[tdpy*18:(tdpy*19), :]



train_X, train_Y = train[:, :-1], train[:, -1]
test_X, test_Y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

#atcual model building
from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_Y, epochs=50, batch_size=72, verbose=2, shuffle=False)
# plot prediction
yhat = model.predict(test_X, verbose=0)
plt.plot(test_Y, label="atcual")
plt.plot(yhat, label="prediction")

plt.legend()
plt.show()
