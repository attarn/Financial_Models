#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:18:13 2019

@author: nicholesattar
"""

import numpy as np
import pandas as pd
import datetime as dt
import time
from datetime import timedelta
import calendar
from alpha_vantage.timeseries import TimeSeries
from pandas.tseries.offsets import BDay
from time import sleep
import scipy.stats as si
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()

### Creating a list of possible options expiry dates. This grabs every friday for the next
# year to account for all possible options regardless of when this program is run.
# It also grabs the third friday of January, June, and September for the next two years
datelist = pd.date_range(pd.datetime.today(), periods=52, freq = 'W-FRI').tolist()
datelist2 = pd.date_range(pd.datetime.today() + timedelta(days=365), periods=104, freq = 'W-FRI').tolist()
options = []
for i in datelist2:
    if (i.month in [1,6,9]) & (15 <= i.day <= 21):
        options.append(i)
for i in options:
    if i not in datelist:
        datelist.append(i)
datelist = ([i.date() for i in datelist])
years = [] # This is every likely expiration date for the next three years in unix time
for i in datelist:
    years.append(calendar.timegm(i.timetuple()))

### Grabbing the options information. This uses the possible expiration dates to grab
# the below list of stocks' options information for every strike price and every expiry date.
# It also tells you how long it takes to load each stock's option data (usually 60-120 seconds)
# and how much of the stock's option data has been loaded.
#stocks  = ['AMZN','SNAP','X','F','M','PG','HON','TSLA','KO','XOM']
stocks = ['SNAP', 'X', 'PG']
all_stocks={}
for stock in stocks:
    t0 = time.time()
    print('Loading',str(stock))
    call2 = {}
    put2 = {}
    for i in years: # only works where there are both calls and puts
        try:
            call2[str(dt.datetime.utcfromtimestamp(int(i)).date())],put2[str(dt.datetime.utcfromtimestamp(int(i)).date())] = pd.read_html(('https://finance.yahoo.com/quote/%s/options?p=%s&date=%s' % (stock,stock,int(i))))                                    
        except:
            pass 
        if years.index(i) in [15,30,45]:
            print(str(stock)+':',str(years.index(i)/0.6)+'%')
    all_stocks[str(stock)] = {'Call':call2,'Put':put2}
    t1 = time.time()
    print('Time To Load', str(stock) + ':', str(t1-t0),'\n')

### Creates one dataframe out of the nested dictionairies. The above dictionary
# 'all_stocks' is a nested dictionary of the options information. It's great for navigating
# and clicking through but it's hard to work with so this converts it to one single dataframe
# This becomes a multiindex dataframe and can be accessed by using 
# idx = pd.IndexSlice and call_df = all_stocks_df.loc[idx[:,'Call'],:] for example
# result.loc['stock','date',strike]['Call/Put','Column Name'] also works
# result.loc['AMZN', '2019-06-21', 1600.0]['Call','Strike']

all_stocks_df = pd.DataFrame()
for stock in all_stocks:
    options_df = pd.DataFrame()
    for i in all_stocks[stock]:
        mid = pd.concat(all_stocks[stock][i], axis = 0)
        mid['Option'] = str(i)
        mid.set_index('Option', append=True, inplace=True)
        mid = mid.reorder_levels([2,0,1], axis=0)
        options_df = options_df.append(mid)
    options_df['Stock'] = str(stock)
    options_df.set_index('Stock', append=True, inplace=True)
    options_df = options_df.reorder_levels([3,0,1,2], axis=0)
    all_stocks_df = all_stocks_df.append(options_df)

### This gets rid of all the low 'Open Interest' Options. There can't be 
# arbitrage if the option is illiquid     
all_stocks_df['Open Interest'] = all_stocks_df['Open Interest'].replace('-', '0')
all_stocks_df['Open Interest'] = all_stocks_df['Open Interest'].astype(int)
all_stocks_df = all_stocks_df[all_stocks_df['Open Interest'] > 200]

### Making a dataframe of synthetic assets
Synthetic = pd.DataFrame([['=','=','=','=','=','='],['Long Call','Short Call','Long Stock','Short Stock','Short Stock','Long Stock'],['+','+','+','+','+','+'],['Short Put','Long Put','Long Put','Short Put','Long Call','Short Call']], columns = ['Long Stock','Short Stock','Long Call','Short Call','Long Put','Short Put']).T                    
Synthetic.columns = ['=','Synthetic','+','Equivalent']


### Grabbing the current risk free rate. 3 Month Treasury Bill i-Rate
rf = pd.read_html('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield', header = 0)[1]
rf = rf.iloc[len(rf)-1,3]

### Adding time to expiry to all_stocks_df. This adds the number of business days 
# until the option expires and the number of actual days until it expires
BDays_to_expiry = pd.Series()
Days_to_expiry = pd.Series()
for i in all_stocks_df.index.get_level_values(2):
    BDays_to_expiry = BDays_to_expiry.append(pd.Series(np.busday_count(str(dt.datetime.today().date()), str(i))))
    Days_to_expiry = Days_to_expiry.append(pd.Series(np.busday_count(str(dt.datetime.today().date()), str(i), weekmask = [1,1,1,1,1,1,1])))

all_stocks_df = all_stocks_df.assign(BDays_to_expiry=BDays_to_expiry.values)
all_stocks_df = all_stocks_df.assign(Days_to_expiry=Days_to_expiry.values)


### All the stock pricing data. This uses Alpha Vantage to grab stock information.
# The key comes with a free account and it's limited to 5 requests
# per minute, which is why the loop has to sleep. It grabs the price every hour
# and the price from everyday. 
ts = TimeSeries(key = 'SSMOWFN5YVIU7J3K', output_format='pandas')
Pricing_intraday = {}
Pricing_daily = {}
for stock in stocks:
    print('Loading Pricing Data For:',str(stock),'\n')
    Pricing_intraday[str(stock)] = ts.get_intraday(symbol=stock.rstrip(), interval = '60min', outputsize='full')[0]
    sleep(15.0)
    Pricing_daily[str(stock)] = ts.get_daily(symbol=stock.rstrip(), outputsize = 'full')[0]
    sleep(15.0)

### Getting volatility (Annualized 30 Day Volatility ). This calculates 2 kinds of volatility
# The first is the volatility from the last 30 business days from the intraday data
# and the second is the volatility from the last 30 business days from the daily data.
# Both volatilities are annualized
Volatility = pd.DataFrame(columns = {'Yrly Vol From Intraday Vol', 'Yrly Vol From Daily Vol'})
start = pd.to_datetime(str((dt.datetime.today() - BDay(30)).date()) + ' 09:30:00')
for i in Pricing_intraday:
    Pricing_intraday[i].index = pd.to_datetime(Pricing_intraday[i].index)
    Volatility.loc[str(i)] = [np.log(Pricing_intraday[i].loc[start:]['4. close']/Pricing_intraday[i].loc[start:]['4. close'].shift(1)).std()*np.sqrt(250*7), np.log(Pricing_daily[i]['4. close'][-30:]/Pricing_daily[i]['4. close'][-30:].shift(1)).std()*np.sqrt(250)]

### Making Series out of the 30 day Vols. This turns the volatilities into a series so that
# they can be added to all_stocks_df
Yrly_Vol_From_Intraday_Vol = pd.Series() 
Yrly_Vol_From_Daily_Vol = pd.Series()
for i in stocks:
    Yrly_Vol_From_Intraday_Vol = Yrly_Vol_From_Intraday_Vol.append(pd.Series(Volatility.loc[i][0]*(np.ones(len(all_stocks_df.loc[i,:,:,:])))))
    Yrly_Vol_From_Daily_Vol = Yrly_Vol_From_Daily_Vol.append(pd.Series(Volatility.loc[i][1]*(np.ones(len(all_stocks_df.loc[i,:,:,:])))))

### Getting vol for time to expiry. This calculates the volatility of the stock over the same
# number of days that the option has left until expiration. If the option expires 200 days
# from now, then this finds the volatility of the stock over the last 200 days.
expiry_vol = pd.Series() # SNAP and other newer stocks don't go back long enough for this to work on 600+ day options
for i in stocks:
    for j in all_stocks_df.loc[str(i),:,:,:]['Days_to_expiry']:
        expiry_vol = expiry_vol.append(pd.Series(np.log(Pricing_daily[i]['4. close'][j:]/Pricing_daily[i]['4. close'][j:].shift(1)).std()*np.sqrt(250), index = [j]))

### Adding All the Volatility Series to the Dataframe    
all_stocks_df = all_stocks_df.assign(expiry_vol=expiry_vol.values)
all_stocks_df = all_stocks_df.assign(Yrly_Vol_From_Intraday_Vol=Yrly_Vol_From_Intraday_Vol.values)
all_stocks_df = all_stocks_df.assign(Yrly_Vol_From_Daily_Vol=Yrly_Vol_From_Daily_Vol.values)

### Checking where options of the same strike price exist
# Sets the Strike as one of the Indices
all_stocks_df.set_index('Strike', append=True, drop = False, inplace=True)
# Dropping the numerical index
all_stocks_df.index = all_stocks_df.index.droplevel(3)
# Creating Call and Put DataFrames
idx = pd.IndexSlice
call_df = all_stocks_df.loc[idx[:,'Call'],:]
put_df = all_stocks_df.loc[idx[:,'Put'],:]
# Drop the number in the index
call_df.index = call_df.index.droplevel(1)
put_df.index = put_df.index.droplevel(1)
# Adding Call/Put to column index
call_df.columns = pd.MultiIndex.from_product([call_df.columns, ['Call']])
call_df.columns = call_df.columns.swaplevel(0, 1)
put_df.columns = pd.MultiIndex.from_product([put_df.columns, ['Put']])
put_df.columns = put_df.columns.swaplevel(0, 1)
# Combining the two
result = pd.concat([call_df, put_df], axis=1, sort=False).dropna()

### 'result' is the final dataframe and all further work will be edits to it ###


### Sorting 'stocks' to be alphabetical in order to match 'result'
stocks.sort()

### Grab the strike price for each pair of options
Strike = pd.Series(result.index.get_level_values(2), index = result.index)

### Grabbing the most recent close price and adding it to 'result'
Stocks_df = pd.DataFrame({"Most Recent Closing Price": [Pricing_daily[str(i)].iloc[-1]['4. close'] for i in stocks]}, index = [i for i in stocks])
Stocks = pd.Series()
for i in stocks:
    Stocks = Stocks.append(pd.Series(float(Stocks_df.loc[i])*np.ones(len(result.loc[i]))))
Stocks.index = result.index
result = result.assign(Current_Stock_Price=Stocks.values)

### Call Bid Price. This assumes we're selling the call and buying the put
CallIV = result['Call','Bid']
# Put Ask Price
PutIV = result['Put','Ask']


### Adding Black Scholes Prices. This is for european options but it's a close approximation
def euro_vanilla(S, K, T, r, s, option = 'call'):
    #S: spot price
    #K: strike price
    #T: time to maturity
    #r: interest rate
    #sigma: volatility of underlying asset
    
    d1 = (np.log(S / K) + (r + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * s ** 2) * T) / (s * np.sqrt(T))   
    if option == 'Call':
        results = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    if option == 'Put':
        results = (K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))  
    return results
# This is BS based off of the minimum of the three volatilities calculated and 
# from the maximum to give the range of possible prices
BS_Call_Price_Min_Vol = pd.Series()
BS_Put_Price_Min_Vol = pd.Series()
BS_Call_Price_Max_Vol = pd.Series()
BS_Put_Price_Max_Vol = pd.Series()
for j in ['Call','Put']:
    for i in Stocks.index:
        S = Stocks[i]
        K = Strike[i]
        T = result.loc[i][str(j),'Days_to_expiry']/365
        r = rf/100
        s = min(result.loc[i][str(j),'Yrly_Vol_From_Intraday_Vol'],result.loc[i][str(j),'Yrly_Vol_From_Daily_Vol'],result.loc[i][str(j),'expiry_vol'])
        if j == 'Call':
            BS_Call_Price_Min_Vol = BS_Call_Price_Min_Vol.append(pd.Series(euro_vanilla(S, K, T, r, s, option = str(j))))
        else:
            BS_Put_Price_Min_Vol = BS_Put_Price_Min_Vol.append(pd.Series(euro_vanilla(S, K, T, r, s, option = str(j))))
        s = max(result.loc[i][str(j),'Yrly_Vol_From_Intraday_Vol'],result.loc[i][str(j),'Yrly_Vol_From_Daily_Vol'],result.loc[i][str(j),'expiry_vol'])
        if j == 'Call':
            BS_Call_Price_Max_Vol = BS_Call_Price_Max_Vol.append(pd.Series(euro_vanilla(S, K, T, r, s, option = str(j))))
        else:
            BS_Put_Price_Max_Vol = BS_Put_Price_Max_Vol.append(pd.Series(euro_vanilla(S, K, T, r, s, option = str(j))))
# This adds the BS prices to the 'result' dataframe
temp_index = result.index
result = result.reset_index(drop = True)
BS_Call_Price_Min_Vol = BS_Call_Price_Min_Vol.reset_index(drop = True)
BS_Put_Price_Min_Vol = BS_Put_Price_Min_Vol.reset_index(drop = True)
BS_Call_Price_Max_Vol = BS_Call_Price_Max_Vol.reset_index(drop = True)
BS_Put_Price_Max_Vol = BS_Put_Price_Max_Vol.reset_index(drop = True)
result['Call','BS_Call_Price_Min_Vol'] = BS_Call_Price_Min_Vol
result['Put','BS_Put_Price_Min_Vol'] = BS_Put_Price_Min_Vol
result['Call','BS_Call_Price_Max_Vol'] = BS_Call_Price_Max_Vol
result['Put','BS_Put_Price_Max_Vol'] = BS_Put_Price_Max_Vol
result.index = temp_index

### Finding differences in price between asset and synthetic asset and adding to 'result'
# This is the Converstion Trading opportunity. It is based on old slow data so it's not
# useful in real life but it shows that there can be arbitrage opportunities
Arbitrage = pd.Series(Strike*np.exp(rf/100*result['Put','Days_to_expiry']/365) - Stocks + pd.to_numeric(CallIV) - pd.to_numeric(PutIV))
result = result.assign(Arbitrage=Arbitrage.values)

### Proving that the Black Scholes Prices satisfy Put-Call Parity. All values are 
# or are nearly 0
PC_Parity = result['Call','BS_Call_Price_Min_Vol'] + Strike*np.exp(-rf/100*result['Call','Days_to_expiry']/365) - Stocks - result['Put','BS_Put_Price_Min_Vol']
result = result.assign(PC_Parity=PC_Parity.values)

### Removing any entries where the call or put ask price is 0
result = result[(result['Call','Bid'] != 0) & (result['Put','Ask'] != 0)]

### Removes options that haven't been traded in the last business day. This is one place where
# the code can fail depending on the time of day you run it. Yahoo is slow to update so if 
# you run this in the middle of trading hours you might lose some options that have traded today
# but haven't been updated on Yahoo.
# This grabs the last business day it could have been traded
Last_Business_Day = 0
Current_Time = pd.to_datetime(dt.datetime.now().strftime('%H:%M:%S'))
if (pd.datetime.today().weekday() in [0,1,2,3,4]) & (Current_Time > pd.to_datetime('11:00:00')):
    Last_Business_Day = pd.to_datetime(dt.datetime.today().strftime("%Y/%m/%d"))
elif (pd.datetime.today().weekday() == 5) | ((pd.datetime.today().weekday() in [0,1,2,3,4]) & (Current_Time < pd.to_datetime('11:00:00'))):
    Last_Business_Day = pd.to_datetime(dt.datetime.today().strftime("%Y/%m/%d")) - timedelta(1)
else:
     Last_Business_Day = pd.to_datetime(dt.datetime.today().strftime("%Y/%m/%d")) - timedelta(2)
# This clears out stocks not traded during the last business day because we need liquidity
result['Call','Last Trade Date'] = pd.to_datetime(result['Call','Last Trade Date'])
result = result[pd.to_datetime(result['Call','Last Trade Date']).dt.normalize() == pd.to_datetime(Last_Business_Day).tz_localize('US/Eastern')]
result['Put','Last Trade Date'] = pd.to_datetime(result['Put','Last Trade Date'])
result = result[pd.to_datetime(result['Put','Last Trade Date']).dt.normalize() == pd.to_datetime(Last_Business_Day).tz_localize('US/Eastern')]

### This is the end of the program. The 'result' dataframe is complete and lists
# all Call and Put option information for all liquid options along with three measures
# volatility, the appropriate BS prices based off the min and max of those volatilities, 
# the stock price, the Conversion Arbitrage opportunity, and proof that there is no BS pricing 
# arbitrage opportunity under the column 'PC_Parity' for Put-Call Parity. Below is graphing
# one of the options to show Converstion Arbitrage opportunity. 
# Something like print(result.loc['AMZN', '2019-06-21', 1600.0]) gives a good look at
# all the information for an option



### Graphing The an Option with of a stock with a closing price close to the strike price 
Close_To_Strike = result[(result['Current_Stock_Price']*1.01 > result.index.get_level_values(2)) & (result.index.get_level_values(2) > result['Current_Stock_Price']*.99)].index[0]
Graph = pd.Series(list(range(int(Strike[Close_To_Strike]*.98), int(Strike[Close_To_Strike]*1.02)))) 
Graph_Call = pd.Series(name = 'Call')
Graph_Put = pd.Series(name = 'Put')
Graph_Stock = pd.Series(Graph-Stocks[Close_To_Strike])
for i in Graph:
    if i < Strike.loc[Close_To_Strike]:
        Graph_Call = Graph_Call.append(pd.Series(-float(result['Call','Bid'][Close_To_Strike])))
        Graph_Put = Graph_Put.append(pd.Series(-i+Strike.loc[Close_To_Strike]-result['Put','Ask'][Close_To_Strike]))
    else:
        Graph_Call = Graph_Call.append(pd.Series(i-Strike.loc[Close_To_Strike]-float(result['Call','Bid'][Close_To_Strike])))
        Graph_Put = Graph_Put.append(pd.Series(-result['Put','Ask'][Close_To_Strike]))
Graph_Call = Graph_Call.reset_index(drop = True)
Graph_Put = Graph_Put.reset_index(drop = True)
Synth_Call = Graph_Put + Graph_Stock
plotting = pd.DataFrame([Graph_Stock, Graph_Call, Graph_Put, Graph, Synth_Call], index = ['Stock', 'Call', 'Put', 'Stock_Returns', 'Synth_Call']).T

# Plotting the Graph
plt.axhline(0, c = 'black', lw = 4)
plt.plot(Graph, plotting['Stock'], lw = 3)
plt.plot(Graph, plotting['Call'], lw = 3)
plt.plot(Graph, plotting['Put'], lw = 3)
plt.plot(Graph, plotting['Synth_Call'], ls = '--', lw = 3)
plt.title('Graph of Returns')
plt.xlabel('Stock Price')
plt.ylabel('Returns')
plt.legend()
plt.show()

# Proof that it works
print('Price of Stock $'+ str(Stocks[Close_To_Strike]))
print('Strike Price $'+ str(Strike.loc[Close_To_Strike]))
print('Price of Call $'+ str(result['Call','Bid'][Close_To_Strike]))
print('Price of Put $'+ str(result['Put','Ask'][Close_To_Strike]))
print('Price of Synthetic Call $'+ str(round(abs(Synth_Call[0]),4)))

print('\n')
print('Difference between Call and Synthetic Call $'+ str(round(float(Synth_Call.head(1) - Graph_Call.head(1)),3)))
print('Calculated Arbitrage from DataFrame $', str(round(result['Arbitrage',''][Close_To_Strike],3)))




