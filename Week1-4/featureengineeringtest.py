import talib
import datagatheringtest as dgt

rawdata = dgt.gethist("AAPL", "20y") #get Apple's stock data for the past 20 years using the function we wrote above

price = rawdata["Close"] #each day's stock price can be represented by the close price for that day

sma100 = talib.SMA(price, timeperiod = 100) #SMA for the last 100 days
tema100 = talib.TEMA(price, timeperiod=100) #TEMA for the last 20 days
rsi10 = talib.RSI(price, timeperiod=10) #RSI for the last 10 days
print(rsi10)

mktdata = dgt.gethist("^GSPC", "20y") #get S&P500 prices for the last 20 years

mktprice = mktdata["Close"] #get the close as it is a representative price for the day

stocktomktpriceratio = price / mktprice #create a ratio between price and mktprice. Numpy arrays are easily divisible with a simple operator if they are the same dimensions, which these two should be as they are both 20 years long

print(stocktomktpriceratio)