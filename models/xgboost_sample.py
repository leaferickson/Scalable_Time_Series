# -*- coding: utf-8 -*-

#Imports
import pandas as pd
import numpy as np
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Data
data = pd.read_csv("train.csv")
mask1 = (data["store"] == 1)
mask2 = data["item"] == 1
dat1 = data[mask1][mask2]


#Build Model
#ar =list(dat1["sales"])
#ar = [None] + ar
#del ar[-1]
#
#test = "AR" + str(6 + 1)

def createARterms(data, col_select, lags):
    for lag in range(lags):
        if lag == 0:
            ar = list(dat1[col_select])
        else:
            ar = list(dat1["AR" + str(lag)])
        ar = [None] + ar
        del ar[-1]
        data["AR" + str(lag + 1)] = ar

createARterms(dat1, "sales", 7)


#Add day of year and day of week
day_of_year = []
day_of_week = []
for str_date in dat1["date"]:
    date = dt.datetime.strptime(str_date, '%Y-%m-%d')
    day_of_year.append(date.timetuple().tm_yday)
    day_of_week.append(date.weekday())
dat1["day_pf_year"] = day_of_year
dat1["day_of_week"] = day_of_week

#Build XGBoost Model
#XGBClassifier(n_estimators = 400, min_samples_split = 10, min_samples_leaf = 1, min_impurity_decrease = 0, max_depth = 110, random_state = 31)
X = np.array(dat1)[7:,4:]
y = np.array(dat1)[7:,3:4]
X_test = X[-365:]
y_test = y[-365:]
X_train = X[:-365]
y_train = y[:-365]

xgb = XGBRegressor(n_estimators = 300)
xgb.fit(X_train, y_train)

preds = xgb.predict(X_test)
diffs = pd.DataFrame(preds.reshape(len(preds), 1) - y_test)
diffs["actual"] = y_test
diffs["Predicted"] = preds
print(mean_squared_error(y_test, preds))



#Graphing Results
plt.scatter(range(len(diffs)), diffs[0])

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(diffs[0], lags=5, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(diffs[0], lags=5, ax=ax2)
plt.show()