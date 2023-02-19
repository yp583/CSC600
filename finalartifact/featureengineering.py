import talib
import datagathering as dg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#sp500 = "^GSPC" #the S&P500 index symbol on yahoo finance. This allows me to get the market data for the top 500 companies which are highly infuential to short term price movements
#ticker = "AAPL" #the ticker of the company to be trained on

#market = dg.gethist(ticker, "20y")

#features that are not included with TALib
def rh(data, timeperiod):
    rollinghighs = data.copy()
    for i in range(0, timeperiod):
        rollinghighs[i] = np.NaN
    for i in range(timeperiod, data.size):
        rollinghighs[i] = max(data[i-timeperiod:i])
    return rollinghighs
def rl(data, timeperiod):
    rollinglows = data.copy()
    for i in range(0, timeperiod):
        rollinglows[i] = np.NaN
    for i in range(timeperiod, data.size):
        rollinglows[i] = min(data[i-timeperiod:i])
    return rollinglows
def mktratioprice(data):
    mkt = dg.gethist("^GSPC", "20y") #the S&P500 index symbol on yahoo finance. This allows me to get the market data for the top 500 companies which are highly infuential to short term price movements
    mktratio = data.copy()
    for i in range(0, data.size):
        mktratio[i] = data[i]/mkt["Close"][i]
    return mktratio
def mktratioRSI(data, timeperiod):
    mkt = dg.gethist("^GSPC", "20y") #the S&P500 index symbol on yahoo finance. This allows me to get the market data for the top 500 companies which are highly infuential to short term price movements
    mktRSI = talib.RSI(mkt["Close"], timeperiod=timeperiod)
    mktratio = data.copy()
    for i in range(timeperiod, data.size):
        mktratio[i] = data[i]/mktRSI[i]
    return mktratio

def addengineeredfeatures(data):
    price = data["Close"]
    rh10 = rh(price, timeperiod=10)
    rl10 = rh(price, timeperiod=10)
    sma10 = talib.SMA(price, timeperiod = 10)
    sma20 = talib.SMA(price, timeperiod = 20)
    sma100 = talib.SMA(price, timeperiod = 100)
    tema20 = talib.TEMA(price, timeperiod=20)
    tema100 = talib.TEMA(price, timeperiod=100)
    rsi10 = talib.RSI(price, timeperiod=10)
    mktratprice = mktratioprice(price)
    mktratiorsi10 = mktratioRSI(rsi10, 10)
    d = {'price': price, 'volume': data["Volume"], 'sma10': sma10, 'sma20': sma20, 'sma100': sma100, 'tema20': tema20, 'tema100':tema100, 'rh10':rh10, 'rl10': rl10, 'RSI10':rsi10, 'mktratioprice': mktratprice, 'mktratioRSI10':mktratiorsi10}
    df = pd.DataFrame(data=d)
    df = df.dropna()
    return df
#addengineeredfeatures(market)
#plt.plot(market["Close"])
#plt.show()