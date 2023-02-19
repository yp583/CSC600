import yfinance as yf
"""
sp500 = "^GSPC" #the S&P500 index symbol on yahoo finance. This allows me to get the market data for the top 500 companies which are highly infuential to short term price movements
ticker_ = "AAPL" #the ticker of the company to be trained on

info_ = yf.Ticker(ticker_) #initializes the ticker with yfinance library
marketinfo = yf.Ticker(sp500)

hist = info_.history(period="20y") #data going back 20 years. Returned is a pandas dataframe.
markethist = marketinfo.history(period="20y") #data going back 20 years. Returned is a pandas dataframe.
print(hist)
"""
def gethist(ticker, period):
    info = yf.Ticker(ticker) #initializes the ticker with yfinance library
    return info.history(period=period) #data going back to however many years specified by parameter. Returned is a pandas dataframe.