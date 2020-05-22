#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 01:44:10 2020

@author: anagh
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def dftest(series):
    res=adfuller(series)
    p=res[1]
    return p

df=pd.read_csv('milk.csv')
df.columns=['Month','Qty']
df.dropna(inplace=True)

df.set_index('Month',inplace=True)
df.index=pd.to_datetime(df.index)

from statsmodels.tsa.seasonal import seasonal_decompose
df_decomposed=seasonal_decompose(df['Qty'],model='multiplicative')
f=df_decomposed.plot()

df['First Difference']=df['Qty']-df['Qty'].shift(1)
df.dropna(inplace=True)

df['Seasonal FD']=df['First Difference']-df['First Difference'].shift(12)
df.dropna(inplace=True)
plt.plot(df['Seasonal FD'])
print(dftest(df['Seasonal FD']))

from statsmodels.tsa.stattools import acf,pacf
acfgraph=acf(df['Seasonal FD'],nlags=5)
pacfgraph=pacf(df['Seasonal FD'],nlags=5)

plt.plot(acfgraph)
y=[0 for i in range(6)]
plt.plot(y)
plt.show()

plt.plot(pacfgraph)
plt.plot(y)
plt.show()

p=0
q=0
d=1

acfplotsarima=acf(df['Seasonal FD'],nlags=25)
plt.plot(acfplotsarima)
plt.show()

P=0
D=1
Q=1
s=12

model=sm.tsa.SARIMAX(df['Qty'],order=(0,1,0),seasonal_order=(0,1,1,12))
result=model.fit()

df['Qty'].plot()
df['Pred']=result.predict(start=130,end=150)
df[['Pred','Qty']].plot()


from pandas.tseries.offsets import DateOffset
newdates=[df.index[-1]+DateOffset(months=i) for i in range(30)]



dfnew=pd.DataFrame(index=newdates,columns=df.columns)
dffinal=pd.concat([df,dfnew])

dffinal['Pred']=result.predict(start=156,end=185)
dffinal[['Qty','Pred']].plot()

























