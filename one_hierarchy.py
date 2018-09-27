# -*- coding: utf-8 -*-

#Imports
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.insert(0, './models/')
sys.path
from ARIMA import arima_model
from xg_boost import xgboost_model
from random_forest import random_forest_model
from prophet import fb_prophet_model
from sklearn.metrics import mean_squared_error
#from hierarchical import hierarchical

class master():
    
    def __init__(self, data, test_size):
        """Arima uses only a numpy array (no col needed)"""
        self.data = data
        self.differences = 0
        self.ar = 0
        self.ma = 0
        self.test_size = test_size
        self.example = 0
        self.data_to_pass = 0
        self.one_page_data = 0
        self.hierarchical_data = 0


    def main(self):
        my_dict = {}
        for col in self.data:
            my_dict.update({col : max(self.data[col])})
        largest_traffic_page = max(my_dict, key=my_dict.get)
        non_hierarchical_preds = "initialize"
        count = 1
        for col in self.data:
            self.one_page_data = self.data[col]
        
            #Train-test split
            """Will have to repeat this process for each "example" in data """
            train, test = self.train_test_split(self.one_page_data, self.test_size)
            
            #Set data to pass based on differencing
            my_arima = arima_model(train)
            self.differences = my_arima.choose_differences()
            self.data_to_pass = train
            for diff in range(self.differences):
                self.data_to_pass = self.data_to_pass.diff()[1:]
            
            self.ar, self.ma = my_arima.choose_ar_ma_terms(self.data_to_pass)
            
            #Run ARIMA Model
            arima_forecast = my_arima.run_model(self.differences, self.ar, self.ma, self.test_size)
            arima_forecast.rename(columns = {arima_forecast.columns[0]:"preds"}, inplace = True)
            arima_forecast["dates"] = test.index
            arima_forecast["page"] = col
            arima_forecast["model"] = "arima"
            
            #Tree Models
            xg_model = xgboost_model(train, self.ar, self.ma, self.differences)
            xg_model.fit()
            data_with_lag_terms = xg_model.get_modified_data()
#            return data_with_lag_terms
#            return xg_model.forecast(90), data_with_lag_terms
            xg_forecast = xg_model.forecast(self.test_size)
            xg_forecast = xg_forecast.iloc[-self.test_size:,0:1]
            xg_forecast.rename(columns = {xg_forecast.columns[0]:"preds"}, inplace = True)
            xg_forecast["dates"] = test.index
            xg_forecast["page"] = col
            xg_forecast["model"] = "xg_boost"
    #        rf_model = random_forest_model()
    #        rf_model.fit(train, data_with_lag_terms)
            
            #Prophet Model
            prophet_model = fb_prophet_model()
            prophet_forecast = prophet_model.fit(train, self.test_size)
            cols = prophet_forecast.columns.tolist()
            cols = cols[1:2] + cols[0:1]
            prophet_forecast = prophet_forecast[cols]
            prophet_forecast.rename(columns = {prophet_forecast.columns[0]:"preds", prophet_forecast.columns[1]:"dates"}, inplace = True)
            prophet_forecast["page"] = col
            prophet_forecast["model"] = "prophet"
            
            
            forecasts = [arima_forecast, xg_forecast, prophet_forecast]
            error_dict = {}
            for forecast in forecasts:
                error_dict.update({forecast["model"].unique()[0]: mean_squared_error(test, forecast["preds"])})
            min_error_model = min(error_dict, key=error_dict.get)
            all_forecasts = arima_forecast.append(xg_forecast.append(prophet_forecast))
            min_error_model_forecast = all_forecasts[all_forecasts["model"] == min_error_model]
            
            #Test forecast to use against hierarchical
            if count > 1:
                all_page_forecasts = all_page_forecasts.append(min_error_model_forecast)
            else:
                all_page_forecasts = all_forecasts[all_forecasts["model"] == min_error_model]
                count += 1
            
            #Real Forecasts
            if min_error_model_forecast["model"].unique()[0] == "arima":
                my_arima = arima_model(self.one_page_data)
                arima_forecast = my_arima.run_model(self.differences, self.ar, self.ma, self.test_size)
                arima_forecast.rename(columns = {arima_forecast.columns[0]:"preds"}, inplace = True)
                arima_forecast["dates"] = test.index
                arima_forecast["page"] = col
                arima_forecast["model"] = "arima"
                forecast = arima_forecast
            elif min_error_model_forecast["model"].unique()[0] == "xg_boost":
                xg_model = xgboost_model(self.one_page_data, self.ar, self.ma, self.differences)
                xg_model.fit()
                forecast = xg_model.forecast(self.test_size)
                forecast = forecast.iloc[-self.test_size:,0:1]
                forecast.rename(columns = {forecast.columns[0]:"preds"}, inplace = True)
                forecast["dates"] = test.index
                forecast["page"] = col
                forecast["model"] = "xg_boost"
            else:
                prophet_model = fb_prophet_model()
                prophet_forecast = prophet_model.fit(self.one_page_data, self.test_size)
                cols = prophet_forecast.columns.tolist()
                cols = cols[1:2] + cols[0:1]
                prophet_forecast = prophet_forecast[cols]
                prophet_forecast.rename(columns = {prophet_forecast.columns[0]:"preds", prophet_forecast.columns[1]:"dates"}, inplace = True)
                prophet_forecast["page"] = col
                prophet_forecast["model"] = "prophet"
                forecast = prophet_forecast
            
            if count > 2:
                real_forecasts = real_forecasts.append(forecast)
            else:
                real_forecasts = forecast
                count += 1
            
            
#            if largest_traffic_page == col:
#                hierarchical_model = hierarchical(data, 90, self.ar, self.ma, self.differences)
#                hierarchical_data = hierarchical_model.transform_data()
#            
#        #Hierarchical Model
#        hierarchical_model = hierarchical(data, 90, self.ar, self.ma)
#        hierarchical_data = hierarchical_model.transform_data()
        
        #If not hierarchical
        
        
        return all_page_forecasts
#    
    def train_test_split(self, data, test_number):
        return data[:-test_number], data[-test_number:]
        
    
#    
data = pickle.load(open("./data/filled_data2.pkl", "rb"))
truths = ['House_of_Cards_(U.S._TV_series)_en' in page for page in data["Page"]]
data = data[(truths)]
#
##Convert from wide to long
data = data.set_index("Page")
data = data.transpose()
#
##for col in data:
##    print(max(data[col]))
##    print(col.strip().split("_")[-2:] == ['all-access', 'all-agents'])
#example = data.iloc[:,2]
test = master(data, 90)
test2 = test.main()
#
##test4 = test3.append(test2.append(test3))
#test4[test4["model"] == "xg_boost"]
#if test4:
#    test4.append(test4)
#test2.columns[0]
#test2["Page"] = test2.columns[0]
#test2.reset_index(inplace = True)
#test2.rename(columns = {test2.columns[0]:"dates", test2.columns[1]:"views"}, inplace = True)
#test2.iloc[:,1:-1]