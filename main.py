import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_ta as ta
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Download historical stock data
symbol = 'AAPL'
start_date = '2010-01-01'
end_date = '2022-12-31'
data = yf.download(symbol, start=start_date, end=end_date)

# Calculate technical indicators
data['SMA_30'] = ta.sma(data['Close'], length=30)
data['SMA_90'] = ta.sma(data['Close'], length=90)
data['RSI'] = ta.rsi(data['Close'], length=14)

# Remove rows with missing values
data.dropna(inplace=True)

# Discretize the technical indicators
n_bins = 5
discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
discretized_data = discretizer.fit_transform(data[['SMA_30', 'SMA_90', 'RSI']])
discretized_data = pd.DataFrame(discretized_data, columns=['SMA_30', 'SMA_90', 'RSI'], index=data.index)

# Convert discretized_data into one-hot encoded format
discretized_data = discretized_data.astype(str)
discretized_data = pd.get_dummies(discretized_data, prefix_sep='=')

# Generate frequent itemsets using the FPGrowth algorithm
min_support = 0.05
frequent_itemsets = fpgrowth(discretized_data, min_support=min_support, use_colnames=True)

# Generate association rules
min_confidence = 0.6
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_confidence)

# Filter the rules based on your criteria
selected_rules = rules[(rules['lift'] > 1) & (rules['conviction'] > 1)]

# Initialize trading signals
trading_signals = pd.Series(index=discretized_data.index, dtype=float)

# Loop through the selected rules
for index, rule in selected_rules.iterrows():
    antecedent = list(rule['antecedents'])
    consequent = list(rule['consequents'])[0]

    # Apply the antecedent conditions to the discretized data
    condition = True
    for item in antecedent:
        condition &= (discretized_data[item] == 1)

    # Generate a buy signal if the consequent condition is met
    if consequent.startswith('SMA_30'):
        buy_signal = (discretized_data[consequent] == 1) & condition
    else:
        # Customize this part for other technical indicators
        buy_signal = pd.Series(False, index=discretized_data.index)

    # Update trading signals
    trading_signals[buy_signal] = 1

# Fill any missing signals with 0
trading_signals.fillna(0, inplace=True)


# Initialize the position, cash, and equity
position = 0
cash = 100000  # Starting cash balance
equity = pd.Series(index=trading_signals.index, dtype=float)

# Loop through the trading signals
for date, signal in trading_signals.iteritems():
    if signal == 1 and position == 0:
        # Buy signal: Buy as many shares as possible
        shares_to_buy = cash // data.loc[date, 'Close']
        position += shares_to_buy
        cash -= shares_to_buy * data.loc[date, 'Close']
    elif signal == 0 and position > 0:
        # Sell signal: Sell all shares
        cash += position * data.loc[date, 'Close']
        position = 0

    # Update equity
    equity[date] = cash + position * data.loc[date, 'Close']

# Calculate the performance metrics of the trading strategy
total_return = (equity[-1] - equity[0]) / equity[0]
print(f"Total return: {total_return * 100:.2f}%")

# Visualize the stock prices and trading signals
plt.figure(figsize=(15, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(trading_signals * data['Close'], 'o', label='Trading Signals', markersize=5)
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.title('Stock Prices and Trading Signals')
plt.show()

# Visualize the equity curve of the trading strategy
plt.figure(figsize=(15, 7))
plt.plot(equity, label='Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.legend(loc='upper left')
plt.title('Equity Curve of the Trading Strategy')
plt.show()
