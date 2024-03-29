# Coversion_Arbitrage

This file imports options data from Yahoo and stock price information from Alpha Vantage.
It puts the options data into a dataframe, filters it by Open Interest, last trade date
and only includes where there exists calls and puts with the same strike and expiry.
It adds the last price of the stock, adds three measures of the volatility, and adds the 
Black-Scholes price of the options. It calculates the arbitrage opportunity from
Conversion Arbitrage and shows that there is no arbitrage opportunity with the 
Black-Scholes prices. Lastly, it selects an option and graphs the synthetic equivalent
showing the opportunity for arbitrage and showing that the Conversion Arbitrage
opportunity was calculated correctly. A note: if this is run in the morning, Yahoo
can list all Bid and Ask prices as 0 and this program will not work.

