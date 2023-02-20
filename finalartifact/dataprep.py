from pandas import DataFrame, concat

#this is not my function, I found it online. This is something I could have coded, but it would have been tedious. The function 
#basically takes my data and splits it into combinations of slices of the past n day increments and future m day increments for
#training an LSTM model
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan: 
        agg.dropna(inplace=True)
    return agg

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import datagathering as dg
import featureengineering as fe

def prepdata(raw, trainon, predictfor):
    rawengineered = fe.getengineeredfeatures(raw)
    featurenum = len(rawengineered.axes[1])

    scaler = MinMaxScaler()

    #scale the data between 0 and 1 which makes the model learn better. Disparate numbers like volume vs RSI will cause weights to be skewed in the model.
    data = pd.DataFrame(scaler.fit_transform(rawengineered), columns=rawengineered.columns)

    preppeddata = series_to_supervised(data, trainon, predictfor) #train on the last 7 days to predict the next 1 days
    dropcols = []
    for i in range(0, predictfor):
        for j in range((featurenum*trainon+1)+(i*featurenum), (featurenum*(trainon+1))+(i*featurenum)):
            dropcols.append(j)
    print(dropcols)
    preppeddata.drop(preppeddata.columns[dropcols], axis = 1, inplace=True)

    
    #split data into training and testing sets

    vals = preppeddata.values

    datax, datay = vals[:, :-predictfor], vals[:, -predictfor:]

    # reshape input to be 3D [samples, timesteps, features]
    datax = datax.reshape((datax.shape[0], 1, datax.shape[1]))

    
    return datax, datay