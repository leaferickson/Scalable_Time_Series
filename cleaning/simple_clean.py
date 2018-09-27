# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
from datetime import datetime

data = pd.read_csv("../data/train_2.csv")

data.fillna(0, inplace = True)

data

pickle.dump(data, open("../data/filled_data2.pkl","wb"))