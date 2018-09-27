# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf


class arima_model():
    
    def __init__(self, data):
        """Arima uses only a numpy array (no col needed)"""
        self.data = data
    
    def choose_differences(self):
        """Choose the I term for the ARIMA model"""
        differences = {0 : self.data.std()}
        differences.update({1 : self.data.diff().std()})
        differences.update({2 : self.data.diff().diff().std()})
        return min(differences, key=differences.get)
    
    def choose_ar_ma_terms(self, differenced_data):
        """Choose the number or AR and MA terms. Based on acf and pacf functions."""
        autocorr, acf_conf_int = acf(differenced_data, nlags = 40, alpha = 0.05)
        partialautocorr, pacf_conf_int = pacf(differenced_data, nlags = 40, alpha = 0.05)
        for i in range(2,4):
            my_dict = {}
            acf_diffs = autocorr[i] - autocorr[i - 1]
            pacf_diffs = partialautocorr[i] - partialautocorr[i - 1]
            my_dict.update({"acf " + str(i - 1) : acf_diffs})
            my_dict.update({"pacf " + str(i -1) : pacf_diffs})
        chosen_max = max(my_dict, key=my_dict.get)
        which_graph, number = chosen_max.split(" ")
        ar = 0
        ma = 0
        if which_graph == "acf":
            ma = int(number)
        else:
            ar = int(number)
        return ar, ma
        """Must come back and refine how you get conf_int cutoff point"""
        
    def run_model(self, differences, ar, ma, forecast_time):
        """Runs the model"""
        model = ARIMA(endog = self.data, order = (ar, differences, ma))
        results = model.fit() #Choose if you want a constant or not
        forecast = results.forecast(steps = forecast_time)[0]
        seasonality, sar, sma = self.test_seasonality(results.resid)
        if seasonality:
            model = sm.tsa.statespace.SARIMAX(endog = self.data, order=(ar,differences,ma), seasonal_order = (sar,1,sma,7), trend='c', enforce_invertibility=False)
            results=model.fit()
            forecast = results.forecast(steps = forecast_time)
        return pd.DataFrame(forecast)
        
    def test_seasonality(self, resids):
        """Test for seasonality in the model.
        If needed, determine if an ar or ma term is needed."""
        ar = 0
        ma = 0
        autocorr = acf(resids[1:], nlags = 40)
        partialautocorr = pacf(resids[1:], nlags = 40)
        #Test for weely seasonality only. SM's arima can't handle yearly on daily data (too much lag).
        weekly_correlation_ma = abs(np.corrcoef(autocorr[1:8], autocorr[8:15])[0][1])
        weekly_correlation_ar = abs(np.corrcoef(partialautocorr[1:8], partialautocorr[8:15])[0][1])
        if weekly_correlation_ma >= 0.9:
            ma += 1
            return True, ar, ma
        elif weekly_correlation_ar >= 0.9:
            ar += 1
            return True, ar, ma
        else:
            return False, ar, ma
        
        
