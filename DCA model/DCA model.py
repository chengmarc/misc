# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 15:54:43 2022

@author: Admin
"""
import os
import pandas as pd
import numpy as np

from sys import path
parent_folder = r'C:\Users\Admin\Desktop\Crypto Model'
if parent_folder not in path: path.append(parent_folder)

# %% Define Function
def DCA_weekly(df, start_date: str, end_date: str):
    df = df.reindex(index=df.index[::-1]) 
    df['date'] = pd.DatetimeIndex(df['date'])
    df = df[df['date'] >= start_date]
    df = df[df['date'] < end_date]   
        
    USD_invested = 0
    crypto_owned = 0
    day = 0
        
    for row, rows in df.iterrows():
        day += 1
        day_of_week = day % 7        
        if day_of_week == 1:
            USD_invested += 100
            crypto_owned += 100/rows['open']
            
    USD_eqvlnt = crypto_owned * df.iloc[-1]['open']
    yield_perc = USD_eqvlnt / USD_invested
    
    return USD_invested, USD_eqvlnt, crypto_owned, yield_perc

# %% Execution
df = pd.read_csv(os.path.join(parent_folder, 'Binance_BTCUSDT_d.csv'), usecols=['date','open'])

DCA_weekly(df, start_date="2017-12-15", end_date="2022-12-15")
