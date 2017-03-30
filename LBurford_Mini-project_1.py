import math
import numpy as np
import datetime as dt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import pandas as pd
from zipline.utils import tradingcalendar
import pytz
from statsmodels.tsa.stattools import coint
 
#initialize variables and schedule functions
def initialize(context): 
        
    context.stocks = [sid(24), sid(5061), sid(4283), sid(5885), sid(20940), sid(2765), sid(33729), sid(4589)]
    
    context.pair_list = [(sid(24), sid(5061)),
                  (sid(4283), sid(5885)),   
                  (sid(20940), sid(2765)), 
                  (sid(33729), sid(4589))
                  ]        
    
    schedule_function(rebalance, date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(hours=1))
     
#Augmented Dickey Fuller stationary test with 10% critical value
def stat_test(x):
    
    x = np.array(x)
    result = ts.adfuller(x, regression='c')
    
    if result[0] >= result[4]['10%']:       
        return True
    else:
        return False
        
#Cointegration test with 10% critical value
def coint_test(x, y):
        
    result = coint(x,y)
    
    if result[0] >= result[2][2]:  
        return True
    else:
        return False
       
    
def rebalance(context, data):
    
    if len(get_open_orders()) > 0:
        return
    
    co = False
    
    for i in range(0, len(context.pair_list)):
        
        sx = context.pair_list[i][0]
        sy = context.pair_list[i][1]
                    
        # Calculate the 60 day mavg & std for each stock pair in our list.
        hist = data.history(context.stocks, 'price', 5, '1d')
        avg = np.mean(hist[sx] - hist[sy])
        std = np.std(hist[sx] - hist[sy])
        
        # Get the difference in current price for each stock in each pair in our list.
        cur_price = data.current(context.stocks, 'price')
        diff = (cur_price[sx] - cur_price[sy])
                
        #if not(stat_test(hist[sx])) and not(stat_test(hist[sy])):
            
        #check for cointegration
        co = coint_test(hist[sx], hist[sy])
            
        #else:
        #    co = False
        
        if co == True:
            if diff >= (avg + (2*std)):
                if context.portfolio.positions[sx].amount > 0:
                    order(sx, (-1 * context.portfolio.positions[sx].amount))  
                order(sy, 1500)
            else:
                if context.portfolio.positions[sy].amount > 0:
                    order(sy, (-1 * context.portfolio.positions[sy].amount))
                order(sx, 1500)