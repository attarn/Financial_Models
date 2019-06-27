#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:10:36 2019

@author: nicholesattar
"""
import pandas as pd

def options_graph2(Price, Portfolio, title = None, maximum = None, minimum = None):
    import matplotlib.pyplot as plt
    import numpy as np
    length = len(Portfolio)
    price = [i/10 for i in range(int(8*Price),int(12*Price))]
    returns = np.zeros(len(price))
    for i in range(length):
        if Portfolio['Asset'].loc[i] == 'Call':
            for j in range(len(returns)):
                returns[j] = returns[j] + Portfolio['Number'].loc[i]*max(-Portfolio['Price'].loc[i], price[j] - Portfolio['Strike'].loc[i] - Portfolio['Price'].loc[i])
        if Portfolio['Asset'].loc[i] == 'Put':
            for j in range(len(returns)):
                returns[j] = returns[j] + Portfolio['Number'].loc[i]*max(-Portfolio['Price'].loc[i], Portfolio['Strike'].loc[i] - price[j] - Portfolio['Price'].loc[i])
        if Portfolio['Asset'].loc[i] == 'Stock':
            for j in range(len(returns)):
                returns[j] = returns[j] + Portfolio['Number'].loc[i]*(price[j]-Price)
    y_max = .25*Price
    y_min = -.25*Price
    max_gain = ''
    max_loss = ''
    if maximum == 'bounded':
        y_max = max(returns)*2
        max_gain = ' Max Profit: '+str(round(max(returns), 2))
    elif maximum == 'inf':
        max_gain = ' Max Profit: inf'
    else:
        pass
    if minimum == 'bounded':
        y_min = min(returns)*2
        max_loss = ' Max Loss: '+str(round(min(returns), 2))
    elif minimum == 'inf':
        max_loss = ' Max Loss: inf'
    else:
        pass
    plt.plot(price,returns)
    plt.plot(price,pd.Series(price)-Price, ls = '--')
    plt.title(title)
    plt.xlabel('Stock Price'+'\n'+str(max_gain)+str(max_loss))
    plt.ylabel('Return')
    #shift = max(abs(y_min), abs(y_max))
    #plt.xlim(Price - 2*shift, Price + 2*shift)
    plt.ylim(y_min, y_max)
    plt.axhline(0, lw = 3, c = 'k')
    plt.axvline(Price, lw = 1, c = 'r', linestyle = '--')
    plt.tight_layout()
    return plt.show()


'''
Examples 


# Bear Spread
Price = 32
Portfolio = pd.DataFrame([['Call', 30, -1, 3],
                          ['Call', 35, 1, 1]],
                columns = ['Asset','Strike','Number','Price'])
options_graph2(Price, Portfolio,'Bear Spread', maximum = 'bounded', minimum = 'bounded')

# Synthetic Strangle
Price = 37.5
Portfolio = pd.DataFrame([['Stock', 37.5, -1],
                          ['Call', 40, 1, 2],
                          ['Call', 35, 1, 4]],
                columns = ['Asset','Strike','Number','Price'])
options_graph2(Price, Portfolio,'Synthetic Strangle', maximum = 'inf', minimum = 'bounded')
'''








