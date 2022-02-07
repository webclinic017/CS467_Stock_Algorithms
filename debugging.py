import pandas as pd
import numpy as np
import datetime

ETFpath = '3x-ETF/'  # Should be updated to eventual folder name
ETFfile = 'TQQQ.csv'  # Should be updated to file name, or autosearch like "glob" functions below
ETF_3x = pd.read_csv(ETFpath + ETFfile)

Quantpath = 'Algorithm_Backtests_All_3xETFs/datasets/' # Should be updated to eventual folder name
Quantfile = 'TQQQ-2010-03-01.csv'

# Three options to fit - Price, Quantity, Value - right now going to fit to just Quantity
# (since that's what we care about - Quantity we'll manipulate based on price)

# Note, for now, columns are off by one, so "Price" here equates to column

Quant_rawdata = pd.read_csv(Quantpath + Quantfile)
Quant_dates = Quant_rawdata.index

ETF_3x['Quantity Moved'] = np.nan

for days in range(len(ETF_3x)):
    flag = 0
    temp = 0
    for dates in Quant_dates:
        if ETF_3x['Date'][days] in dates:
            temp += float(Quant_rawdata['Price'][dates])

    if temp == 0:
        temp = np.nan
    ETF_3x['Quantity Moved'][days] = temp


print(ETF_3x)
ETF_3x = ETF_3x.dropna()
print(ETF_3x)