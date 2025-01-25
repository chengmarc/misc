# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import timedelta
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


# %%
df = pd.read_csv(r"C:\Users\marcc\Desktop\btc.csv")
df['time'] = pd.to_datetime(df['time'])
df['pct_change'] = df['PriceUSD'].pct_change() * 100

plt.figure(figsize=(15, 5))
plt.plot(df['time'].to_list(), df['PriceUSD'].to_list())
plt.grid(color='gray', alpha=0.5)
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.title('Bitcoin Price since September 17th 2014')
plt.xlabel('time')
plt.ylabel('PriceUSD')
plt.show()

INTERVAL_DAYS = 30
FORCAST_INTERVALS = 5
CUTOFF_PERCENTAGE = 30


# %%

def slice_observ(df, interval_days=90):
    
    halving_dates = ['2012-11-28', '2016-07-09', '2020-05-11', '2024-04-19', '2028-03-20']
    halving_dates = list(pd.to_datetime(halving_dates))

    interval_data = {}
    for i in range(len(halving_dates)-1):
        
        halving_start = halving_dates[i]
        halving_end = halving_dates[i+1]
        
        current_date = halving_start    
        while True:
            
            interval_start = current_date
            interval_end = current_date + timedelta(days=interval_days)
            interval_index = str((interval_start - halving_start).days) + "-" + str((interval_end - halving_start).days)
            interval_prices = df[(df['time'] >= interval_start) & (df['time'] < interval_end)]['pct_change']
            
            # Combine data for this interval across all halvings
            if interval_index not in interval_data:
                interval_data[interval_index] = []
            interval_data[interval_index].extend(interval_prices.tolist())
            
            # Check if interval exceeds the next halving event
            if interval_end > halving_end: break          
            current_date = interval_end
            
    return interval_data


observ_distributons = slice_observ(df, INTERVAL_DAYS)


# %%

def fit_dist(rv_list, cutoff):

    distributions = [stats.norm, stats.cauchy, stats.t, stats.f,
                     stats.alpha, stats.beta, stats.gamma, 
                     stats.chi, stats.chi2]

    best_fit = None
    best_params = None
    best_ks_stat = np.inf

    for distribution in distributions:

        params = distribution.fit([x for x in rv_list if abs(x) < cutoff])
        ks_stat, _ = stats.kstest(rv_list, distribution.cdf, args=params)
        # Perform the Kolmogorov-Smirnov test

        if ks_stat < best_ks_stat:
            best_fit = distribution
            best_params = params
            best_ks_stat = ks_stat

    print("Best fit distribution:", best_fit.name)
    print("Best fit parameters:", best_params)
    print("Kolmogorov-Smirnov statistic:", best_ks_stat)

    return best_fit, best_params


fitted_distributions = {}
for interval_index, prices in observ_distributons.items():
    dist, params = fit_dist(prices, CUTOFF_PERCENTAGE)
    fitted_distributions[interval_index] = (dist, params)
    

# %%

def plot_dist(observ, dist, params):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
    axs[0].hist(observ, bins=60, range=(-30, 30), edgecolor='black')
    axs[0].grid(color='gray', alpha=0.5)
    axs[0].set_title('Histogram of Observed Daily Returns')
    axs[0].set_xlabel('Daily Return')
    axs[0].set_ylabel('Frequency')

    axs[1].hist(dist.rvs(*params, size=10000), bins=60, range=(-30, 30), edgecolor='black')
    axs[1].grid(color='gray', alpha=0.5)
    axs[1].set_title('Histogram of Simulated Daily Returns')
    axs[1].set_xlabel('Daily Return')
    axs[1].set_ylabel('Frequency')
    

for interval in fitted_distributions.keys():
    observ = observ_distributons[interval]
    dist, params = fitted_distributions[interval]
    plot_dist(observ, dist, params)
    

# %%

def simulate_market(df, cutoff, forecast_intervals, interval_days=90):
    
    start_date = list(df['time'])[-1]
    end_date = start_date + timedelta(days=interval_days*forecast_intervals-1)   

    simulations = []
    current_date = start_date
    while current_date <= end_date:
        
        interval_start = current_date
        interval_end = current_date + timedelta(days=interval_days)    
        interval_index = str((interval_start - start_date).days) + "-" + str((interval_end - start_date).days)
        
        dist, params = fitted_distributions[interval_index]
        
        for i in range(interval_days):
            change = -200
            while change < -cutoff or change > cutoff:
                change = dist.rvs(*params)
            simulations.append(change)
        
        current_date = interval_end
        

    last_price = list(df['PriceUSD'])[-2]
    price_forecast = [last_price]
    for i in range(len(simulations)):
        price_forecast.append(price_forecast[-1] * (100 + simulations[i]) / 100)
    price_forecast = price_forecast[1:]

    date_series = pd.Series(pd.date_range(start=start_date, end=start_date + timedelta(days=len(simulations)-1), freq='D'))
    data_series = pd.Series(price_forecast)
    
    index = pd.concat([df['time'], date_series], ignore_index=True)
    price = pd.concat([df['PriceUSD'], data_series], ignore_index=True)
    
    output = pd.concat([index, price], axis=1)
    output.columns = ['time', 'PriceUSD']
    output.set_index('time', inplace=True)
    return output


output = simulate_market(df, CUTOFF_PERCENTAGE, FORCAST_INTERVALS, INTERVAL_DAYS)
output[-365-FORCAST_INTERVALS*INTERVAL_DAYS:].plot(figsize=(15, 5), legend=True, title="Forecasted Bitcoin Price")
plt.grid(color='gray', alpha=0.5)
plt.title('Simulated Bitcoin Price from 2020 to 2026')
plt.xlabel('Days Past')
plt.ylabel('Price (USD)')
plt.show()


# %% Simulate 100 times
simulations, targets = [], []
plt.figure(figsize=(15, 5))
for i in tqdm(range(10000)):
    output = simulate_market(df, CUTOFF_PERCENTAGE, FORCAST_INTERVALS, INTERVAL_DAYS)
    simulations.append(output[-365-FORCAST_INTERVALS*INTERVAL_DAYS:])
    targets.append(int(output[-1:]['PriceUSD']))
    
for series in simulations:
    plt.plot(series, color='grey', alpha=0.1)

plt.title('Multiple Series Plot of Bitcoin Price Simulation')
plt.xlabel('Days Past')
plt.ylabel('Price (USD)')
plt.ylim(0, 600000)
plt.show()
   
plt.hist(targets, bins=60, edgecolor='black')
plt.grid(color='gray', alpha=0.5)
plt.title('Distribution of Price at the End of Forecast')
plt.xlabel('Daily Return')
plt.ylabel('Price (USD)')
plt.show()

