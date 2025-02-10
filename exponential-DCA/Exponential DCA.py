# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 17:25:24 2025

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


# %%
def prepare_data():
    
    data = pd.read_csv('btc.csv')
    data = data[365*2:-1]

    data['Date'] = pd.to_datetime(data['time'])
    data.set_index('Date', inplace=True)
    
    data['LogPriceUSD'] = np.log(data['PriceUSD'])
    data = data[['PriceUSD', 'LogPriceUSD']]
    data.fillna(0, inplace=True)
    
    return data

data = prepare_data()


# %%
def log_fit(x, a, b, c):
    
    return a * np.log(x + c) + b


def log_trend(X, Y, get_param=False):
    """
    X : array-like, a series of date indices in integer form.
    Y : array-like, a series of float values corresponding to the dependent variable.
        
    get_param : bool, optional
        If `get_param` is False, returns an array of fitted trend values.
        If `get_param` is True, returns a tuple `(a, b, c)`, which are the parameters of the logarithmic trend.
    """
    
    initial_guess = [1, 1, 1]
    fitted_param, _ = curve_fit(log_fit, X, Y, p0=initial_guess)
    a, b, c = fitted_param
    
    trend = log_fit(X, a, b, c)
    if get_param: return a, b, c
    else: return trend


# %%
def transform_col(df, name, plot=False):
        
    # 1. Logarithmic Transform    
    Y = pd.Series(df[f'{name}']) 
    X = (Y.index - Y.index[0]).days

    df[f'{name}_log_trend'] = log_trend(X, Y, get_param=False)
    df[f'{name}_residuals'] = df[f'{name}'] - df[f'{name}_log_trend']
    
    # 2. Residual Scaling    
    rolling_window=30
    volatility = df[f'{name}_residuals'].rolling(rolling_window).std().dropna()

    Y = volatility.values.reshape(-1, 1)
    X = np.arange(len(volatility)).reshape(-1, 1)  # Time as feature

    def get_vol_trend(X, Y):
        
        regressor = LinearRegression()
        regressor.fit(X, Y)        
        trend = regressor.predict(np.arange(len(df)).reshape(-1, 1))
        return trend

    df[f'{name}_vol_trend'] = get_vol_trend(X, Y)
    df[f'{name}_scaled_residuals'] = df[f'{name}_residuals'] / df[f'{name}_vol_trend']

    # 3. Plotting
    def plot_trend_residuals(df):
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        axs[0].set_title("Original Data and Logarithmic Trend")
        axs[1].set_title("Residuals of Logarithmic Trend")
        axs[2].set_title('Scaled Residuals of Logarithmic Trend')

        axs[0].plot(df.index, df[f'{name}'], label='Original Data', color='blue')
        axs[0].plot(df.index, df[f'{name}_log_trend'], label='Logarithmic Trend', color='red', linestyle='--')
        axs[1].plot(df.index, df[f'{name}_residuals'], label='Residuals', color='green')
        axs[2].plot(df.index, df[f'{name}_scaled_residuals'], label='Scaled Residuals')
        
        [ax.grid(True) or ax.legend() for ax in axs]
        
        plt.tight_layout()
        plt.show()

    if plot: plot_trend_residuals(df)
    return df

data = transform_col(data, 'LogPriceUSD', plot=True)


# %%
def generate_signals(df):
    
    df = df.copy()
    df["Signal"] = "Empty"
    df.loc[df["LogPriceUSD_scaled_residuals"] < -5, "Signal"] = "Buy"  # Buy signal
    df.loc[df["LogPriceUSD_scaled_residuals"] > 10, "Signal"] = "Sell"  # Sell signal
    return df

data = generate_signals(data)


# %%
def HODL_strategy(df, initial_capital):
    
    first_price = df.iloc[0]['PriceUSD']
    btc_held = initial_capital / first_price  
    final_value = btc_held * df.iloc[-1]['PriceUSD']  
    
    equity_curve = [btc_held * price for price in df['PriceUSD']]  
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve

    return output, final_value

df_HODL, final_value_HODL = HODL_strategy(df=data[-9*365:], initial_capital=10000)


# %%
def DCA_strategy(df, initial_capital, allocation_per_trade):
    
    capital = 0  
    position =  initial_capital / df.iloc[0]['PriceUSD']
    equity_curve = []  
    trade_history = []
    
    for i in range(len(df)):
        price = df.iloc[i]['PriceUSD']
        signal = df.iloc[i]['Signal']
        equity = capital + (position * price)

        if signal == 'Buy' and capital > 0:
            amount_to_invest = capital * allocation_per_trade
            btc_to_buy = amount_to_invest / price
            
            if btc_to_buy > 0:
                position += btc_to_buy
                capital -= amount_to_invest  
                trade_history.append({'Type': 'Buy', 'Price': price, 'BTC': btc_to_buy, 'Capital': capital})

        elif signal == 'Sell' and position > 0:
            btc_to_sell = position * allocation_per_trade
            
            if btc_to_sell > 0:
                capital += btc_to_sell * price  
                position -= btc_to_sell  
                trade_history.append({'Type': 'Sell', 'Price': price, 'BTC': btc_to_sell, 'Capital': capital})

        equity_curve.append(equity)  
        
    output = df.copy()
    output.loc[:, 'Equity'] = equity_curve
    final_value = capital + (position * df.iloc[-1]['PriceUSD'])
    
    return output, trade_history, final_value

df_DCA, trade_history, final_value_DCA = DCA_strategy(df=data[-9*365:], initial_capital=10000, allocation_per_trade=0.025)


# %%
def optimize_allocation(df):
    
    dct = {}
    for year in range(4, 12):
        all_final_values = []
        for i in tqdm(range(1, 100)):
            _, _, final_value = DCA_strategy(df=data[-year*365:], initial_capital=10000, allocation_per_trade=i/100)
            all_final_values.append(final_value)
        dct[f'{year} years'] = all_final_values.copy()
    return dct

# dct = optimize_allocation(df)


# %%
def plot_results(df, df_HODL, df_DCA, log=False):
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Equity Curves
    if log:
        axs[0].plot(df_HODL.index, np.log(df_HODL['Equity']), label='Returns (Buy & Hold)', color='gray', linestyle='dashed', linewidth=2)
        axs[0].plot(df_DCA.index, np.log(df_DCA['Equity']), label='Returns (DCA)', color='blue', linewidth=2)
        axs[0].set_title('Logarithmic Equity Curve Comparison')
        axs[0].set_ylabel('Logarithmic Portfolio Value')
    else:
        axs[0].plot(df_HODL.index, df_HODL['Equity'], label='Returns (Buy & Hold)', color='gray', linestyle='dashed', linewidth=2)
        axs[0].plot(df_DCA.index, df_DCA['Equity'], label='Returns (DCA)', color='blue', linewidth=2)
        axs[0].set_title('Equity Curve Comparison')
        axs[0].set_ylabel('Portfolio Value ($)')

    # Plot 2: BTC Price with Buy/Sell Signals
    buy_signals = df[df['Signal'] == 'Buy']
    sell_signals = df[df['Signal'] == 'Sell']
    
    if log:
        axs[1].plot(df.index, np.log(df['PriceUSD']), label='BTC Price', color='black', linewidth=2)
        axs[1].scatter(buy_signals.index, np.log(buy_signals['PriceUSD']), label='Buy Signal', color='green', marker='s', s=25)
        axs[1].scatter(sell_signals.index, np.log(sell_signals['PriceUSD']), label='Sell Signal', color='red', marker='s', s=25)
        axs[1].set_title('Logarithmic BTC Price with Buy/Sell Signals')
        axs[1].set_ylabel('Logarithmic BTC Price')
    else:
        axs[1].plot(df.index, df['PriceUSD'], label='BTC Price', color='black', linewidth=2)
        axs[1].scatter(buy_signals.index, buy_signals['PriceUSD'], label='Buy Signal', color='green', marker='s', s=25)
        axs[1].scatter(sell_signals.index, sell_signals['PriceUSD'], label='Sell Signal', color='red', marker='s', s=25)
        axs[1].set_title('BTC Price with Buy/Sell Signals')
        axs[1].set_ylabel('Price')

    [ax.grid(True) or ax.legend() for ax in axs]
    
    plt.tight_layout()
    plt.show()


plot_results(data, df_HODL, df_DCA)
plot_results(data, df_HODL, df_DCA, log=True)

