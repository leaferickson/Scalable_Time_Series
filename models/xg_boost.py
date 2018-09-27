# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


class xgboost_model():

    def __init__(self, data, ar, ma, differences):
        """Pass in a numpy array, convert it to Pandas to add on columns"""
        self.original_data = data
        self.ar = ar
        self.ma = ma
        self.differences = differences
        self.data = pd.DataFrame(data)
        self.create_all_new_terms()
        self.final_difference = 0
        self.model = 0
        
    def create_all_new_terms(self):
        for difference in range(self.differences):
            self.make_difference(self.data, self.data.columns[difference], difference + 1)       
        self.final_difference = self.data.columns[-1]
        self.data = self.createARterms(self.data, self.final_difference, self.ar)
        self.data = self.createMAterms(self.data, self.final_difference, self.ma)
        self.data = self.data.dropna()
        
    def make_difference(self, data, col_select, difference):
          data["diff" + str(difference)] = data[col_select].diff() #differences = 
#          data["diff" + str(difference)] = pd.concat([pd.Series(np.NAN), differences[:-1]])
    
    def createARterms(self, data, col_select, lags):
        if lags == 0:
            return data
        for lag in range(lags):
            if lag == 0:
                ar = list(data[col_select])
            else:
                ar = list(data["AR" + str(lag)])
            ar = [None] + ar
            del ar[-1]
            data["AR" + str(lag + 1)] = ar
        return data
    
    def createMAterms(self, data, col_select, lags):
        if lags == 0:
            return data
        for lag in range(2,lags + 1):
            data["MA" + str(lag)] = data[col_select].rolling(lag).mean()
        return data
    
#    def create_seasonal_terms(self, data, col_select, lag, ar_ma):
#        if ar_ma == "ar":
#            ar = data[col_select][:-lag]
#            ar = [None] * lag + ar
#            data["AR" + str(lag)] = ar
#        else:
#            data["MA" + str(lag)] = data[col_select].rolling(lag).mean()
#        return data
    
    def get_modified_data(self):
        return self.data
    
    def fit(self):
        """Remove the Nans?"""
        xgb = XGBRegressor(n_estimators = 300)
        xgb.fit(self.data.iloc[:,self.differences + 1:], self.data.iloc[:,self.differences])
        self.model = xgb
    
    def forecast(self, test):
        for day in range(test):
            self.forecast_one_day()
        return self.data
    
    def forecast_one_day(self):
        self.original_data = self.original_data.append(pd.Series(-5))
        self.data = pd.DataFrame(self.original_data)
        self.create_all_new_terms()
        data = self.data
#        return self.model.predict(self.data.iloc[-1:,:])
        self.data.iloc[-1:,self.differences] = int(self.model.predict(self.data.iloc[-1:,self.differences + 1:]))
        for diff in range(self.differences):
            self.data.iloc[-1:,self.differences - 1 - diff] = int(self.data.iloc[-1:,self.differences - diff]) + int(self.data.iloc[-2:-1,self.differences - 1 - diff])
#            print(type(self.data.iloc[-1:,self.differences - 1 - diff]))
#            print(self.data.iloc[-1:,self.differences - 1 - diff])
            if self.data.iloc[-1:,self.differences - 1 - diff][0] < 0:
#                self.data.iloc[-1:,self.differences - 1 - diff] = abs(-self.data.iloc[-1:,self.differences - 1 - diff] - int(self.data.iloc[-2:-1,self.differences - 1 - diff]))
                self.data.iloc[-1:,self.differences - 1 - diff] = -self.data.iloc[-1:,self.differences - 1 - diff]
        data = self.data
        self.original_data[-1] = self.data.iloc[-1:,0]
        data = self.data
        return self.data
#diff = test3["diff1"]
#pd.concat([pd.Series(np.NAN), diff[:-1]])
#        test3.dropna()
#xgb = XGBRegressor(n_estimators = 300)
#xgb.fit(test3.dropna().iloc[:,2:], test3.dropna().iloc[:,1])
#xgb.predict(test3.iloc[-4:,2:])
        
#test3 = data.iloc[:,:1]
#test3.rolling(10).mean()
#ar = [None] * 7
#ar2 = test3.iloc[:-7,:1]
#ar2 = [e for e in ar2]
#test3.take()
#ar + ar2
#ar3 = ar.extend(ar2)
#for i in range(1):
#    print(i)