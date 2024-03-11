# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:21:49 2023

@author: Admin
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import pandas as pd

path = r"D:\Data Extraction\database"

files = os.listdir(path)

files = [os.path.join(path, file) for file in files]

dataframes = []
for file in files:
    df = pd.read_csv(file, usecols = ["snapped_at", "price"])
    df = df.set_index("snapped_at")
    dataframes.append(df)
    print(f"Successfully appended {os.path.basename(file)}")

final = pd.concat(dataframes, axis=1)
final = final.sort_index()
final.to_csv(r"C:\Users\marcc\Desktop\time_series.csv")