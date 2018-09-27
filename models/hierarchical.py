# -*- coding: utf-8 -*-


#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano
from xg_boost import xgboost_model

class hierarchical():
    
    def __init__(self, data, test_length, ar, ma, difference):
        self.ar = ar
        self.ma = ma
        self.data = data
        self.test_length = test_length
        self.differences = difference
        self.model_output = 0
        self.page_idx = 0
        self.model_elements_dict = {}
        
    def transform_data(self):
        train_set = self.data.iloc[:-1 * self.test_length,:]
        test_set = self.data.iloc[-1 * self.test_length:,:]
        
        tester = 0
        for col in self.data:
            one_page_data = self.data[col]  
            xg_model = xgboost_model(one_page_data, self.ar, self.ma, self.differences)
            modified_data1 = xg_model.get_modified_data()
            modified_data1["page"] = modified_data1.columns[0]
            modified_data1.reset_index(inplace = True)
            modified_data1.rename(columns = {modified_data1.columns[0]:"dates", modified_data1.columns[1]:"views"}, inplace = True)
            if tester != 0:
                modified_data2 = pd.concat([modified_data2, modified_data1], axis = 0)
            else:
                modified_data2 = xg_model.get_modified_data()
                tester += 1
#        modified_data1.iloc[:,1:-1] #How to access the meat of the data
        modified_data2.drop_index(drop = True, inplace = True)
        return modified_data2


    def create_tensors(self, data):
        data = data.dropna()
        self.model_output = theano.shared(np.array(data["views"]).astype(int))
        self.page_idx = theano.shared(np.array(data['page'].values))
        for feature in data.iloc[:,1:-1]:
            self.model_elements_dict["{0}".format(feature)] = theano.shared(np.array(data[feature]))
        
    def fit(data):
        with pm.Model() as hierarchical_model:
            # Hyperpriors
            mu_a = pm.Normal('mu_alpha', mu=0., sd=10e2)
            sigma_a = pm.HalfCauchy('sigma_alpha', beta=2)
#            mu_b = pm.Normal('mu_beta',mu=0., sd = 10e2)
#            sigma_b = pm.HalfCauchy('sigma_beta', beta=2)
            
            
            # Intercept for each county, distributed around group mean mu_a
            a = pm.Normal('alpha', mu=mu_a, sd=sigma_a, shape=len(data['page'].unique()))
            
            to_eval
            for item in self.model_elements_dict.items():
                "mu_{0}".format(item[0]) =  pm.Normal('mu_beta_{0}'.format(item[0]),mu=0., sd = 10e2)
                "sigma_{0}".format(item[0]) = pm.HalfCauchy('sigma_beta_{0}'.format(item[0]), beta=2)
                "{0}".format(item[0]) = pm.Normal('beta_{0}'.format(item[0]), mu=eval("mu_{0}".format(item[0])), sd=eval("sigma_{0}".format(item[0])), shape=len(data['page'].unique()))
                "{0}".format(item[0])[page_idx]
            # Intercept for each county, distributed around group mean mu_a
#            b = pm.Normal('beta', mu=mu_b, sd=sigma_b, shape=len(data['page'].unique()))
            
            # Model error
            eps = pm.Uniform('eps', .01, 10, shape=1)
            
            # Expected value
            est = a[self.page_idx] + eval(to_eval)
            #   + b[page_idx] * ar2 + b[page_idx] * ar3 + b[page_idx] * ar4 + b[page_idx] * ar5 + b[page_idx] * ar6 + b[page_idx] * ar7
            
            # Data likelihood
            y = pm.Normal('y', mu=est, sd=eps, observed=model_output)
        
        with hierarchical_model:
            hierarchical_trace = pm.sample(1000)

count = 0
model_elements_dict = {}
for col in test2.iloc[:,1:-1]:
    model_elements_dict["{0}".format(col)] = theano.shared(np.array(test2[col]))
    test3 = np.array(test2[col])
    count += 1

chosen_max = max(my_dict, key=my_dict.get)
for e in model_elements_dict.items():
    print(e)
    print("mu_{0}".format(e[0]))
hoo456 = 456
eval("hoo456")
