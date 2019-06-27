# Options_Strategies_Graphing
This function returns a graph showing the possible returns of a portfolio containing calls, puts, and long/short equity based on the stock price at the options' expiration date. 
options_graph2(Price, Portfolio, title = None, maximum = None, minimum = None)

### Parameters:
        Price: float 
        ex. Price = 37.5
        Portfolio: takes a pandas dataframe. Must be formatted in the below style

           ex. Portfolio = pd.DataFrame([['Stock', 37.5, -1],
                                                                 ['Call', 40, 1, 2],
                                                                 ['Call', 42, 1, .7],
                                                                 ['Put', 35, 1, 4]],
                                columns = ['Asset', 'Strike', 'Number', 'Price'])

                              The first column takes "Stock", "Call", or "Put" in any order
                              The second column takes the strike price of the options or the price of the stock
                              The third column takes the amount of each asset. Negative if the asset is sold and positive if bought. -inf to inf
                              The fourth column takes the strike price of the options. Leave blank for stock

        title: str (optional)
        ex. "Synthetic Straddle"
        maximum: "bounded" or "inf" (optional). Finds maximum returns
        minimum: "bounded" or "inf" (optional). Finds maximum losses
