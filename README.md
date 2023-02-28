# Machine Learning Prediction Model for Stocks
## Background

In today’s world, the stock market is a common outlet for most people’s retirements, savings, and other financials. The common investor is able to retire earlier, depend less on monthly paychecks, and overall be more financially stable due to profits from their stock portfolio. To interact with the markets, a large portion of the population utilizes ETFs for their investments - a way to easily invest in the entire stock market at once. Investors usually hold onto their ETFs for years until they return sizable profits. However, while this practice mostly works, it is not the most efficient. 

Professional stock traders or analysts seek to make higher returns in the market than the average investor. They make their money by finding the best companies, as they bet on their analysis to lead them to higher profits. They do heavy research into companies to find whether the price is opportune to buy or sell, dissecting SEC filings, tracking major news, insider trading, financial ratios, and much more. However, while analysts and professional stock traders are able to do this required research, the majority of people cannot. Time constraints or lack of know-how, block them from taking full advantage of the markets. Without that barrier, more people can retire earlier, depend less on monthly paychecks, and overall be more financially stable. 
This project aims to build a tool, namely a machine learning model to provide those without this technical and fundamental knowledge to succeed in the markets with minimal effort. 

## Overview

In this project, I create an artificial [neural network](https://www.investopedia.com/terms/n/neuralnetwork.asp) to predict stock prices for the future. To start, I collect [stock data](#the-data) from Yahoo Finance using the [yfinance API](https://github.com/ranaroussi/yfinance). Using this data, I also create some helpful indicators that I believe would benefit the prediction capabilities of my model with a technique called [feature engineering](#what-is-feature-engineering). Professional traders do this as well, using indicators like [SMA](https://www.investopedia.com/terms/s/sma.asp) to aid in their trading and investing strategies. I then use this data to train the neural network, which then makes predictions for future stock prices. 

### The Data
---
From Yahoo Finance, or the yfinance API, I am able to get the **daily open, close, high, low, and volume** for any individual stock. From an API call, a pandas dataframe like the table below is returned.

| Date | Open     |  Close  |   Low   |   High  |  Volume |
| ---- | -------- | ------- | ------- | ------- | ------- |
|Feb 17, 2023|	152.35|	153.00|	150.85|	152.55|	152.55	|59,095,900|
|Feb 16, 2023|	153.51|	156.33|	153.35|	153.71	|153.71	|68,167,900|
|Feb 15, 2023|	153.11|	155.50|	152.88|	155.33	|155.33	|65,669,300|
|Feb 14, 2023|	152.12|	153.77|	150.86|	153.20	|153.20	|61,707,600|
|Feb 13, 2023|	150.95|	154.26|	150.92|	153.85	|153.85	|62,199,000|

*The table above shows Apple's data from Feb 13, 2023 - Feb 17, 2023*

### Feature Engineering
---
Feature engineering is a technique where data/computer scientists construct new data from other data to allow machine learning models to make better predictions. For example, we can create a new feature like a 5 day SMA by averageing the price of the last 5 days. This is a new feature that our model can be trained off of and that provides new information or context for our model despite being based off of the same original data.

For this project, the new features that are engineered are as follows:

1. SMA or [Simple Moving Average](https://www.investopedia.com/terms/s/sma.asp) (10 day, 20 day, and 100 day)
2. TEMA or [Triple Exponenetial Moving Average](https://www.investopedia.com/ask/answers/041315/why-triple-exponential-moving-average-tema-important-traders-and-analysts.asp#:~:text=The%20triple%20exponential%20moving%20average%20(TEMA)%20is%20a%20modified%20moving,associated%20with%20traditional%20moving%20averages.) (10 day, 20 day, and 100 day)
3. RSI or [Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp)
4. Rolling High and Rolling Low (10 day)
5. Ratio of [S&P500](https://www.investopedia.com/terms/s/sp500.asp) to Stock Price
6. Ratio of S&P500 RSI to Stock's RSI

### Long Short Term Memory Models (LSTM)
---

For this project specifically, I use a type of [recurrent neural network](https://www.ibm.com/topics/recurrent-neural-networks) called Long Short Term Memory (LSTM) to make prediction for stock prices. A recurrent neural network is a type of neural network that uses the output from its previous prediction (ie. its prediction for the day before) to influence its next prediction (ie. its prediction for the next day). 

Unfortunately, RNNs have trouble allowing days from futher in the past to influence their predictions for the future. This is where LSTMs come in. The LSTM is a variation of RNNs which allows for datapoints much further back in time to influence the model's predictions, hence the reference to Long Term Memory in the name LSTM. It also allows recent days to still influence its prediction, balancing between allowing data from further back in time as well as data from the more recent past to make its predictions. 

For this reason, this project utilized the LSTM model for its predictions as stock prices are generally related to the stock's past, including anything from yesterday to a couple of months ago.
