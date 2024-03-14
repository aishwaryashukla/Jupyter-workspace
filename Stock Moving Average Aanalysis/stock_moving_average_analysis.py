#!/usr/bin/env python
# coding: utf-8

# In[1]:



# get_ipython().system('pip install yfinance')
import unicodedata
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


# In[2]:


sharekhan_portfolio_df = pd.read_csv("sharekhan.csv",skiprows=4)
sharekhan_stocks_list=sharekhan_portfolio_df['Stock Name'].to_list()
    


# In[3]:


for c in sharekhan_stocks_list :
    print(c) if isinstance(c,str)  else sharekhan_stocks_list.remove(c)
    


# In[4]:


stocks_isin=['INE585B01010',
'INE238A01034',
'INE918I01018',
'INE040A01034',
'INE001A01036',
'INE123W01016',
'INE062A01020',
'INE217B01036',
'INE176B01034',
'INE117A01022',
'INE154A01025',
'INE785M01013',
'INE053A01029',
'INF109K012R6',
'INE009A01021',
'INE075A01022',
'INE081A01012',
'INE376G01013',
'INE600L01024',
'INE002A01018',
'INE849A01020',
'INE111A01025',
'INE586V01016']


# In[5]:


df = pd.read_csv('EQUITY_L.csv')
df=df[df[' ISIN NUMBER'].isin(stocks_isin)]
stocks_list = df['SYMBOL'].tolist()
stocks_list_NSE = list(map(lambda x: x+".NS",stocks_list))
stocks_list_Sharekhan = list(map(lambda x: x+".NS",sharekhan_stocks_list))
final_list = [ls.replace(u'\xa0', u'')  for ls in stocks_list_Sharekhan]
final_list = final_list + stocks_list_NSE
final_list = list(dict.fromkeys(final_list))
print(f" Final List = {final_list}")
data1 = yf.download(final_list,start="2019-01-01",end="2021-03-05")


# In[ ]:





# In[6]:


type(data1['Adj Close'])
close_price_df = data1['Adj Close']
close_price_df.head()


# In[7]:


# close_price_df['ABB.NS.MA10']=close_price_df['ABB.NS'].rolling(10).mean()
# type(close_price_df['ABB.NS'])
# close_price_df['ABB.NS.MA10']=close_price_df.loc[:,('ABB.NS')].rolling(10).mean()
# test = close_price_df.loc[:,('ABB.NS')].rolling(10).mean()
# close_price_df
for cusip in final_list:
    print(" cusip = {cusip}".format(cusip=cusip))
    ma10 = cusip+".MA10"
    ma50 = cusip+".MA50"
    close_price_df[ma10]=close_price_df.loc[:,(cusip)].rolling(10).mean()
    close_price_df[ma50]=close_price_df.loc[:,(cusip)].rolling(50).mean()


# In[8]:


close_price_df.columns


# In[11]:




for ticker in final_list:
    close_price_df[ticker+".MA10"].plot(label=ticker+'.MA10',title=ticker,color='green', y='MA10' )
    close_price_df[ticker+".MA50"].plot(label=ticker+'.MA50',title=ticker, color='red', y='MA50')
    plt.show()


