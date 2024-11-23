# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:52:25 2024

@author: Admin
"""

# Question: 
# I am taking 200mg of Seroquol every day to treat Bipolar-I disorder,
# I want to visualize the amount of chemicals in my body through out time.

from math import log
import matplotlib.pyplot as plt

half_life = 6 # hours
dose = 200 # mg
days = 7 # simulation period

x = 10**(log(0.5, 10)/half_life)-1 # metabolic rate

data = [dose]
date = [0]
for i in range(days*24):
    residual = data[-1]
    residual = residual*(1+x)
    if i%24 == 22:
        residual += dose
    data.append(residual)
    date.append(i/24)
    
plt.plot(date, data)

# Adding labels
plt.xlabel('Days since treatment')
plt.ylabel('Residual in the body (mg)')
plt.title('Half-Life Graph of Medication')