# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class random_forest_model():

    def __init__(self):   
        pass
        
    def fit(self, original_data, modified_data):
        """Remove the Nans?"""
        modified_no_nans = modified_data.dropna()
        original_no_nans = original_data.iloc[modified_no_nans.index]
        rf = RandomForestRegressor(n_estimators = 300)
        rf.fit(modified_no_nans, pd.DataFrame(original_no_nans))
    
    def forecast(self, test):
        ok = 0
        
