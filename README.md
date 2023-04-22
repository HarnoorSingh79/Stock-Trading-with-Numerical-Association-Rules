# Predictive Stock Trading using Numerical Association Rules
This project demonstrates how to use numerical association rules for building a predictive stock trading strategy. Association rule learning is a popular technique in data mining that aims to find relationships between variables in large datasets. In this project, we'll use the mlxtend library to generate association rules from discretized technical indicators and then develop a simple long-only trading strategy based on the selected rules.

##  Requirements
Python 3.6+
pandas
numpy
yfinance
scikit-learn
mlxtend
matplotlib
pandas_ta

## Project Overview
1. Import Libraries and Load Data
First, we'll import the necessary libraries and load historical stock price data using the yfinance library.

2. Calculate Technical Indicators
Next, we'll calculate the technical indicators, such as the simple moving average (SMA) and the relative strength index (RSI), using the pandas_ta library.

3. Discretize Technical Indicators
We'll then discretize the calculated technical indicators into bins using the KBinsDiscretizer from scikit-learn. This step transforms the continuous numerical data into discrete categorical data, which is necessary for generating association rules.

4. Generate Association Rules
Using the discretized technical indicators, we'll generate association rules with the fpgrowth function from the mlxtend library. The generated rules will consist of antecedents and consequents, which describe relationships between different levels of the discretized technical indicators.

5. Select Rules for Trading Signals
From the generated association rules, we'll select a subset of rules that will be used to generate trading signals. In this project, we focus on long-only trading signals based on the selected rules.

6. Implement Trading Strategy
Finally, we'll implement a simple long-only trading strategy based on the generated trading signals. We'll track the equity curve of the strategy to evaluate its performance.

## Results
The trading strategy based on the selected association rules achieved a total return of 848.52%. It's important to note that this result is based on historical data and does not account for trading costs, taxes, or other factors that may affect the real-world performance of the strategy.

## Visualization
We've included visualizations of the stock prices, trading signals, and the equity curve of the trading strategy to help better understand the strategy's performance over time.

![Trading Signals](https://github.com/HarnoorSingh79/Stock-Trading-with-Numerical-Association-Rules/blob/main/Figure_01.png)

![Equaity curve](https://github.com/HarnoorSingh79/Stock-Trading-with-Numerical-Association-Rules/blob/main/Figure_02.png)

## Disclaimer
This project is for educational purposes only and should not be considered as investment advice. The performance of the trading strategy is based on historical data and may not accurately predict future results. Trading stocks carries risks, and you should consult a financial advisor before making any investment decisions.

