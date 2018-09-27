# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

#Data
data = pd.read_csv("train.csv")
mask1 = (data["store"] == 1)
mask2 = data["item"] == 1
dat1 = data[mask1][mask2]


X = dat1[["date", "sales"]]
X.rename(columns = {"date":"ds", "sales":"y"}, inplace = True)
size = int(len(X) * 0.8005)
train, test = X[0:size], X[size:len(X)]


#Prophet
m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=365)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


fig1 = m.plot(forecast)
fig2 = m.plot_components(forecast)

#Error
mean_squared_error(test["y"], forecast.iloc[-365:,-1:])




#Make Model
#def make_model(forecast_days):
predictions = []
for item in data["item"].unique():
    for store in data["store"].unique():
        #Get Data
        mask1 = data["store"] == store
        mask2 = data["item"] == item
        dat1 = data[mask1][mask2]
        X = dat1[["date", "sales"]]
        X.rename(columns = {"date":"ds", "sales":"y"}, inplace = True)
        #Make Model
        m = Prophet()
        m.fit(X)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        for prediction in np.array(forecast.iloc[-90:,-1:]):
            predictions.append(float(prediction))
id_count = np.array(range(len(predictions)))
predictions

submission = pd.DataFrame({'id':id_count, 'sales':predictions})
submission.to_csv('submission.csv')

