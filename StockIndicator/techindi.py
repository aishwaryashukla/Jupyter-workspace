#!/usr/bin/env python
# coding: utf-8

# In[67]:


# importing variables
# !pip install pandas_datareader
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr


from SendMail import *

# https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for-quantitative-trading-8d98751b5fb

# extracting data from Yahoo Finance API
# tickers = ['NIFTYBEES.NS','LALPATHLAB.NS']
tickers = [
    "ASIANPAINT.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "DABUR.NS",
    "GODREJCP.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HINDUNILVR.NS",
    "INDUSINDBK.NS",
    "ITC.NS",
    "JUBLFOOD.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "MARUTI.NS",
    "PIDILITIND.NS",
    "PIIND.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "WIPRO.NS",
    "ABB.NS",
    "AXISBANK.NS",
    "BAJAJFINSV.NS",
    "BIOCON.NS",
    "CONCOR.NS",
    "HAVELLS.NS",
    "HDFC.NS",
    "INDHOTEL.NS",
    "INFY.NS",
    "KAJARIACER.NS",
    "LALPATHLAB.NS",
    "PCJEWELLER.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "TATASTEEL.NS",
    "TCIEXP.NS",
    "TRENT.NS",
]
all_data = pd.DataFrame()
test_data = pd.DataFrame()
no_data = []

for i in tickers:
    try:
        print("Working on : ", i)
        test_data = pdr.get_data_yahoo(
            i, start=dt.datetime(2021, 1, 1), end=dt.date.today()
        )
        test_data["symbol"] = i
        all_data = all_data.append(test_data)
    except:
        no_data.append(i)

# Creating Return column
all_data["return"] = all_data.groupby("symbol")["Close"].pct_change()

# In[ ]:


# In[68]:


# Simple Moving Average
all_data["SMA_5"] = all_data.groupby("symbol")["Close"].transform(
    lambda x: x.rolling(window=5).mean()
)
all_data["SMA_15"] = all_data.groupby("symbol")["Close"].transform(
    lambda x: x.rolling(window=15).mean()
)
all_data["SMA_ratio"] = all_data["SMA_5"] / all_data["SMA_15"]
# all_data[all_data['symbol']=='SBIN.NS']


# In[69]:


date_index = all_data.sort_index().index.unique()
second_last_day = date_index[len(date_index) - 2]
last_day = date_index[len(date_index) - 1]



# Plotting
for i in tickers:
    print("working for " + i)
    start = dt.datetime.strptime("2021-12-01", "%Y-%m-%d")
    #     end = dt.datetime.strptime('2021-10-09', '%Y-%m-%d')
    end = dt.date.today()


    second_last_day_sam_ratio = all_data.loc[second_last_day][
        all_data.loc[second_last_day]["symbol"] == i
    ].iloc[0]["SMA_ratio"]
    last_day_sam_ratio = all_data.loc[last_day][
        all_data.loc[last_day]["symbol"] == i
    ].iloc[0]["SMA_ratio"]

    print(
        "Sma Product : {} {} {}".format(
            second_last_day_sam_ratio,
            last_day_sam_ratio,
            second_last_day_sam_ratio * last_day_sam_ratio,
        )
    )
    if second_last_day_sam_ratio < 1 and last_day_sam_ratio > 1:
        subject = "BUY Opportunity in " + i
        body = " Buy opportunity spotted in stock " + i
        sendmail(subject, body, "aishwarya.shukla@gmail.com")
        print(" BUY opportunity, stock is trending upwards. ")
    if second_last_day_sam_ratio > 1 and last_day_sam_ratio < 1:
        subject = "SELL Opportunity in " + i
        body = " SELL opportunity spotted in stock " + i
        sendmail(subject, body, "aishwarya.shukla@gmail.com")
        print(" SELL opportunity, stock is trending upwards. ")
