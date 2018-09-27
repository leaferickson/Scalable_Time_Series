# -*- coding: utf-8 -*-


#Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

#Data
data = pd.read_csv("train.csv")
mask1 = (data["store"] == 1)
mask2 = data["item"] == 1
dat1 = data[mask1][mask2]


#Plot Autocorrelations
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dat1['sales'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dat1['sales'], lags=40, ax=ax2)
#plt.savefig("Fig1.png")
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dat1['sales'].diff()[1:], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dat1['sales'].diff()[1:], lags=40, ax=ax2)
plt.show()

test = dat1['sales'].diff()


#Make ARIMA
print(sm.tsa.stattools.adfuller(dat1["sales"]))

model=ARIMA(endog=dat1['sales'],order=(0,1,2))
results=model.fit()
print(results.summary())
#model.predict(results)



results.resid.plot(figsize=(12,8))
plt.show()
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(results.resid, lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(results.resid, lags=40, ax=ax2)
plt.show()



 
X = dat1["sales"]
size = int(len(X) * 0.85)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(7,1,2))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t + size]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
plt.plot(test.reset_index()["sales"])
plt.plot(predictions, color='red')
plt.show()







#Automated Method for auto choosing the optimal ARIMA model


def choose_differences(y):
    """Choose the I term for the ARIMA model"""
    differences = {0 : y.std()}
    differences.update({1 : y.diff().std()})
    differences.update({2 : y.diff().diff().std()})
    return min(differences, key=differences.get)

def 


choose_differences(train)
