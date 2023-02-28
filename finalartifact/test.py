#testing out classes and model
import datagathering as dg
import dataprep as dp
import model as ml

import matplotlib.pyplot as plt

raw = dg.gethist("AMZN", "20y") #get amazon data from the last 20 years
data_X, data_Y = dp.prepdata(raw, 10, 1) #prep the raw data. 1st parameter is data, 2nd is the amount of days to train on, and the 3rd is amount days to predict

tdpy = 237 #237 trading days a year

train_X, train_Y = data_X[:tdpy * 18, :], data_Y[:tdpy * 18, :] #train the model on the last 18 years
test_X, test_Y = data_X[tdpy * 19:tdpy * 20, :], data_Y[tdpy * 19:tdpy * 20, :] #train the model on 

spm = ml.StockPredictionModel([train_X.shape[1], train_X.shape[2]], train_Y.shape[1])
spm.train(train_X, train_Y)
prediction = spm.predict(test_X)
error = spm.error(test_Y, prediction)
print(error)


plt.plot(test_Y[:, 0], label="atcual")
plt.plot(prediction[:, 0], label="prediction 1 day out")
plt.plot(prediction[:, 4], label="prediction 4 days out")

plt.legend()
plt.show()
