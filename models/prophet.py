# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

class fb_prophet_model():
        
    def fit(self, train, test_size):
        shaped_data = self.shape_data(train)
        m = Prophet()
        m.fit(shaped_data) #daily_seasonality = False
        future = m.make_future_dataframe(periods = test_size)
        future.tail()
        forecast = m.predict(future)
        predictions = forecast[['ds', 'yhat']][-test_size:]
        return predictions
        
    def shape_data(self, train):
        new_data = train.reset_index()
        new_data.rename(columns = {new_data.columns[0]:"ds", new_data.columns[1]:"y"}, inplace = True)
        return new_data
        
    

