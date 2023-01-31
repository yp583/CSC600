import yfinance as yf

s&p500 = "^GSPC" #the S&P500 index symbol on yahoo finance. This allows me to get the market data for the top 500 companies which are highly infuential to short term price movements
ticker = "AAPL" #the ticker of the company to be trained on

info = yf.Ticker(ticker) #initializes the ticker with yfinance library
marketinfo = yf.Ticker(s&p500)

hist = info.history(period="20y") #data going back 20 years. Returned is a pandas dataframe.
markethist = marketinfo.history(period="20y") #data going back 20 years. Returned is a pandas dataframe.