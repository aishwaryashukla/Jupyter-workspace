#!/usr/bin/env python
# coding: utf-8

# In[67]:


#importing variables
# !pip install pandas_datareader
import pandas as pd
import numpy as np
import datetime as dt
import pandas_datareader as pdr
import seaborn as sns
import matplotlib.pyplot as plt
from SendEmail import *
# https://towardsdatascience.com/building-a-comprehensive-set-of-technical-indicators-in-python-for-quantitative-trading-8d98751b5fb

#extracting data from Yahoo Finance API
# tickers = ['NIFTYBEES.NS','LALPATHLAB.NS']
tickers = ['ASIANPAINT.NS', 'BEL.NS', 'BHARTIARTL.NS', 'DABUR.NS', 'GODREJCP.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS', 'INDUSINDBK.NS', 'ITC.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LT.NS', 'MARUTI.NS', 'PIDILITIND.NS', 'PIIND.NS', 'RELIANCE.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'WIPRO.NS', 'ABB.NS', 'AXISBANK.NS', 'BAJAJFINSV.NS', 'BIOCON.NS', 'CONCOR.NS', 'HAVELLS.NS', 'HDFC.NS', 'INDHOTEL.NS', 'INFY.NS', 'KAJARIACER.NS', 'LALPATHLAB.NS', 'PCJEWELLER.NS', 'SBILIFE.NS', 'SBIN.NS', 'TATASTEEL.NS', 'TCIEXP.NS', 'TRENT.NS']
all_data = pd.DataFrame()
test_data = pd.DataFrame()
no_data = []

for i in tickers:
    try:
        print("Working on : ",i)
        test_data = pdr.get_data_yahoo(i, start = dt.datetime(2021,1,1), end = dt.date.today())
        test_data['symbol'] = i
        all_data = all_data.append(test_data)
    except:
        no_data.append(i)

#Creating Return column
all_data['return'] = all_data.groupby('symbol')['Close'].pct_change() 


# In[ ]:





# In[68]:


#Simple Moving Average
all_data['SMA_5'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 5).mean())
all_data['SMA_15'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window = 15).mean())
all_data['SMA_ratio'] = all_data['SMA_5'] / all_data['SMA_15']
# all_data[all_data['symbol']=='SBIN.NS']


# In[69]:


date_index = all_data.sort_index().index.unique()
second_last_day = date_index[len(date_index)-2]
last_day = date_index[len(date_index)-1]


# In[76]:


second_last_day = date_index[28]
last_day = date_index[29]
second_last_day


# last_day

# In[78]:


#Plotting
for i in tickers:
    print("working for "+i)
    start = dt.datetime.strptime('2021-12-01', '%Y-%m-%d')
#     end = dt.datetime.strptime('2021-10-09', '%Y-%m-%d')
    end = dt.date.today()
    sns.set()
    
    
    second_last_day_sam_ratio = all_data.loc[second_last_day][all_data.loc[second_last_day]['symbol']==i].iloc[0]['SMA_ratio']
    last_day_sam_ratio = all_data.loc[last_day][all_data.loc[last_day]['symbol']==i].iloc[0]['SMA_ratio']
    
    print("Sma Product : {} {} {}".format(second_last_day_sam_ratio ,last_day_sam_ratio, second_last_day_sam_ratio*last_day_sam_ratio))
    if(second_last_day_sam_ratio <1 and last_day_sam_ratio > 1):
        print(" BUY opportunity, stock is trending upwards. ")
    if(second_last_day_sam_ratio >1 and last_day_sam_ratio < 1):
        print(" SELL opportunity, stock is trending upwards. ")
    
    fig = plt.figure(facecolor = 'white', figsize = (20,10))

    ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
    ax0.plot(all_data[all_data.symbol==i].loc[start:end,['Close','SMA_5','SMA_15']])
    ax0.set_facecolor('ghostwhite')
    ax0.legend(['Close','SMA_5','SMA_15'],ncol=3, loc = 'upper left', fontsize = 15)
    plt.title(i+" Stock Price, Slow and Fast Moving Average", fontsize = 20)

    ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
    ax1.plot(all_data[all_data.symbol==i].loc[start:end,['SMA_ratio']], color = 'blue')
    ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
    ax1.set_facecolor('silver')
    plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
    plt.show()


# In[5]:


start = dt.datetime.strptime('2021-8-01', '%Y-%m-%d')
end = dt.datetime.strptime('2021-10-09', '%Y-%m-%d')
sns.set()

fig = plt.figure(facecolor = 'white', figsize = (20,10))

ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
ax0.plot(all_data[all_data.symbol=='SBIN.NS'].loc[start:end,['Close','SMA_5','SMA_15']])
ax0.set_facecolor('ghostwhite')
ax0.legend(['Close','SMA_5','SMA_15'],ncol=3, loc = 'upper left', fontsize = 15)
plt.title(i+" Stock Price, Slow and Fast Moving Average", fontsize = 20)

ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
ax1.plot(all_data[all_data.symbol=='SBIN.NS'].loc[start:end,['SMA_ratio']], color = 'blue')
ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
ax1.set_facecolor('silver')
plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
plt.show()


# In[6]:


#Simple Moving average volume
all_data['SMA5_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 5).mean())
all_data['SMA15_Volume'] = all_data.groupby('symbol')['Volume'].transform(lambda x: x.rolling(window = 15).mean())
all_data['SMA_Volume_Ratio'] = all_data['SMA5_Volume']/all_data['SMA15_Volume']


# In[7]:


all_data


# In[8]:


def Wilder(data, periods):
    start = np.where(~np.isnan(data))[0][0] #Check if nans present in beginning
    Wilder = np.array([np.nan]*len(data))
    Wilder[start+periods-1] = data[start:(start+periods)].mean() #Simple Moving Average
    for i in range(start+periods,len(data)):
        Wilder[i] = (Wilder[i-1]*(periods-1) + data[i])/periods #Wilder Smoothing
    return(Wilder)

all_data['prev_close'] = all_data.groupby('symbol')['Close'].shift(1)
all_data['TR'] = np.maximum((all_data['High'] - all_data['Low']), 
                     np.maximum(abs(all_data['High'] - all_data['prev_close']), 
                     abs(all_data['prev_close'] - all_data['Low'])))
for i in all_data['symbol'].unique():
    TR_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'ATR_5'] = Wilder(TR_data['TR'], 5)
    all_data.loc[all_data.symbol==i,'ATR_15'] = Wilder(TR_data['TR'], 15)

all_data['ATR_Ratio'] = all_data['ATR_5'] / all_data['ATR_15']


all_data['prev_high'] = all_data.groupby('symbol')['High'].shift(1)
all_data['prev_low'] = all_data.groupby('symbol')['Low'].shift(1)

all_data['+DM'] = np.where(~np.isnan(all_data.prev_high),
                           np.where((all_data['High'] > all_data['prev_high']) & 
         (((all_data['High'] - all_data['prev_high']) > (all_data['prev_low'] - all_data['Low']))), 
                                                                  all_data['High'] - all_data['prev_high'], 
                                                                  0),np.nan)

all_data['-DM'] = np.where(~np.isnan(all_data.prev_low),
                           np.where((all_data['prev_low'] > all_data['Low']) & 
         (((all_data['prev_low'] - all_data['Low']) > (all_data['High'] - all_data['prev_high']))), 
                                    all_data['prev_low'] - all_data['Low'], 
                                    0),np.nan)

for i in all_data['symbol'].unique():
    ADX_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'+DM_5'] = Wilder(ADX_data['+DM'], 5)
    all_data.loc[all_data.symbol==i,'-DM_5'] = Wilder(ADX_data['-DM'], 5)
    all_data.loc[all_data.symbol==i,'+DM_15'] = Wilder(ADX_data['+DM'], 15)
    all_data.loc[all_data.symbol==i,'-DM_15'] = Wilder(ADX_data['-DM'], 15)

all_data['+DI_5'] = (all_data['+DM_5']/all_data['ATR_5'])*100
all_data['-DI_5'] = (all_data['-DM_5']/all_data['ATR_5'])*100
all_data['+DI_15'] = (all_data['+DM_15']/all_data['ATR_15'])*100
all_data['-DI_15'] = (all_data['-DM_15']/all_data['ATR_15'])*100

all_data['DX_5'] = (np.round(abs(all_data['+DI_5'] - all_data['-DI_5'])/(all_data['+DI_5'] + all_data['-DI_5']) * 100))

all_data['DX_15'] = (np.round(abs(all_data['+DI_15'] - all_data['-DI_15'])/(all_data['+DI_15'] + all_data['-DI_15']) * 100))

for i in all_data['symbol'].unique():
    ADX_data = all_data[all_data.symbol == i].copy()
    all_data.loc[all_data.symbol==i,'ADX_5'] = Wilder(ADX_data['DX_5'], 5)
    all_data.loc[all_data.symbol==i,'ADX_15'] = Wilder(ADX_data['DX_15'], 15)
all_data


# In[9]:


all_data['Lowest_5D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 5).min())
all_data['High_5D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 5).max())
all_data['Lowest_15D'] = all_data.groupby('symbol')['Low'].transform(lambda x: x.rolling(window = 15).min())
all_data['High_15D'] = all_data.groupby('symbol')['High'].transform(lambda x: x.rolling(window = 15).max())

all_data['Stochastic_5'] = ((all_data['Close'] - all_data['Lowest_5D'])/(all_data['High_5D'] - all_data['Lowest_5D']))*100
all_data['Stochastic_15'] = ((all_data['Close'] - all_data['Lowest_15D'])/(all_data['High_15D'] - all_data['Lowest_15D']))*100

all_data['Stochastic_%D_5'] = all_data['Stochastic_5'].rolling(window = 5).mean()
all_data['Stochastic_%D_15'] = all_data['Stochastic_5'].rolling(window = 15).mean()

all_data['Stochastic_Ratio'] = all_data['Stochastic_%D_5']/all_data['Stochastic_%D_15']
all_data


# In[10]:


all_data['Diff'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.diff())
all_data['Up'] = all_data['Diff']
all_data.loc[(all_data['Up']<0), 'Up'] = 0

all_data['Down'] = all_data['Diff']
all_data.loc[(all_data['Down']>0), 'Down'] = 0 
all_data['Down'] = abs(all_data['Down'])

all_data['avg_5up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=5).mean())
all_data['avg_5down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=5).mean())

all_data['avg_15up'] = all_data.groupby('symbol')['Up'].transform(lambda x: x.rolling(window=15).mean())
all_data['avg_15down'] = all_data.groupby('symbol')['Down'].transform(lambda x: x.rolling(window=15).mean())

all_data['RS_5'] = all_data['avg_5up'] / all_data['avg_5down']
all_data['RS_15'] = all_data['avg_15up'] / all_data['avg_15down']

all_data['RSI_5'] = 100 - (100/(1+all_data['RS_5']))
all_data['RSI_15'] = 100 - (100/(1+all_data['RS_15']))

all_data['RSI_ratio'] = all_data['RSI_5']/all_data['RSI_15']
all_data


# In[11]:


all_data['5Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
all_data['15Ewm'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.ewm(span=15, adjust=False).mean())
all_data['MACD'] = all_data['15Ewm'] - all_data['5Ewm']


# In[12]:


all_data['15MA'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=15).mean())
all_data['SD'] = all_data.groupby('symbol')['Close'].transform(lambda x: x.rolling(window=15).std())
all_data['upperband'] = all_data['15MA'] + 2*all_data['SD']
all_data['lowerband'] = all_data['15MA'] - 2*all_data['SD']


# In[13]:


# all_data[all_data['symbol']=='ITC.NS']
feb_data = all_data.loc['2022-02-21']
feb_data[feb_data['SMA_5'] > feb_data['SMA_15']]
feb_data.to_csv('feb.csv')


# In[14]:


print(feb_data[feb_data['SMA_5'] < feb_data['SMA_15']])


# In[15]:



    start = dt.datetime.strptime('2021-8-01', '%Y-%m-%d')
    end = dt.date.today()
    
    sns.set()
    from scipy.signal import argrelextrema
    tmp = all_data[all_data.symbol =='SBIN.NS'].loc[start:end,['Close']]
    tmp_nparray = tmp.to_numpy()
# tmp_nparray[0]
    local_max = argrelextrema(tmp_nparray, np.greater)

    fig = plt.figure(facecolor = 'white', figsize = (20,10))

    ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
    ax0.plot(all_data[all_data.symbol=='SBIN.NS'].loc[start:end,['SMA_5','SMA_15']])
    ax0.set_facecolor('ghostwhite')
  
    ax0.legend(['Close','SMA_5','SMA_15'],ncol=3, loc = 'upper left', fontsize = 15)
    ax0.plot(all_data[all_data.symbol=='SBIN.NS'].loc[start:end,['Close']],)
#     ax0.plot(tmp_nparray)
    plt.title("SBIN.NS"+" Stock Price, Slow and Fast Moving Average", fontsize = 20)

#     ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
#     ax1.plot(all_data[all_data.symbol=='SBIN.NS'].loc[start:end,['SMA_ratio']], color = 'blue')
#     ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
#     ax1.set_facecolor('silver')
    plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
    plt.show()


# In[16]:



# from scipy.signal import argrelextrema
# tmp = all_data[all_data.symbol =='SBIN.NS'].loc[start:end,['Close']]
# tmp_nparray = tmp.to_numpy()
# tmp_nparray
# argrelextrema(df.data.values, np.less_equal,
#                     order=n)[0]]['data']

single_stock_df = all_data[all_data.symbol =='TECHM.NS']

single_stock_df['max_local'] = single_stock_df.iloc[argrelextrema(single_stock_df['15Ewm'].values,np.greater)[0]]['15Ewm']
# all_data['min_local']=all_data.iloc[argrelextrema(all_data.Close.values,np.less)[0]]['Close']
single_stock_df['min_local'] = single_stock_df.iloc[argrelextrema(single_stock_df['15Ewm'].values,np.less)[0]]['15Ewm']
single_stock_df.to_csv('singlestock.csv')


# In[17]:



    start = dt.datetime.strptime('2021-1-01', '%Y-%m-%d')
    end = dt.date.today()
    
    sns.set()
    
    fig = plt.figure(facecolor = 'white', figsize = (20,10))

    ax0 = plt.subplot2grid((6,4), (1,0), rowspan=4, colspan=4)
#     ax0.plot(single_stock_df.loc[start:end,['SMA_5','SMA_15']])
    ax0.set_facecolor('ghostwhite')
  
    ax0.legend(['Close','SMA_5','SMA_15'],ncol=3, loc = 'upper left', fontsize = 15)
    ax0.plot(single_stock_df.loc[start:end,['Close']],)
#     ax0.plot(tmp_nparray)
    plt.title("TECH.NS"+" Stock Price, Slow and Fast Moving Average", fontsize = 20)
    plt.scatter(single_stock_df.index,single_stock_df['min_local'],c='g')
    plt.scatter(single_stock_df.index,single_stock_df['max_local'],c='r')
#     ax1 = plt.subplot2grid((6,4), (5,0), rowspan=1, colspan=4, sharex = ax0)
#     ax1.plot(all_data[all_data.symbol=='SBIN.NS'].loc[start:end,['SMA_ratio']], color = 'blue')
#     ax1.legend(['SMA_Ratio'],ncol=3, loc = 'upper left', fontsize = 12)
#     ax1.set_facecolor('silver')
    plt.subplots_adjust(left=.09, bottom=.09, right=1, top=.95, wspace=.20, hspace=0)
    plt.show()


# In[18]:


# all_data.columns
cols = ['symbol','High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 
       'return', 'SMA_5', 'SMA_15', 'SMA_ratio', 'SMA5_Volume', 'SMA15_Volume',
       'SMA_Volume_Ratio', 'prev_close', 'TR', 'ATR_5', 'ATR_15', 'ATR_Ratio',
       'prev_high', 'prev_low', '+DM', '-DM', '+DM_5', '-DM_5', '+DM_15',
       '-DM_15', '+DI_5', '-DI_5', '+DI_15', '-DI_15', 'DX_5', 'DX_15',
       'ADX_5', 'ADX_15', 'Lowest_5D', 'High_5D', 'Lowest_15D', 'High_15D',
       'Stochastic_5', 'Stochastic_15', 'Stochastic_%D_5', 'Stochastic_%D_15',
       'Stochastic_Ratio', 'Diff', 'Up', 'Down', 'avg_5up', 'avg_5down',
       'avg_15up', 'avg_15down', 'RS_5', 'RS_15', 'RSI_5', 'RSI_15',
       'RSI_ratio', '5Ewm', '15Ewm', 'MACD', '15MA', 'SD', 'upperband',
       'lowerband']
all_data = all_data[cols]
all_data

