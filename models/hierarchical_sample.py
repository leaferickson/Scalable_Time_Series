# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import pymc3 as pm
import theano


data = pickle.load(open("../data/filled_data2.pkl", "rb"))

test = data["Page"].unique()


#[page.split(".") in page for page in data["Page"]]
#data["Page"].split(".")


truths = ['EGOIST_ja' in page for page in data["Page"]]
# for page in data["Page"]:
#     page in 'Francisco_el_matemático'
np.array(truths).sum()
for page in data[(truths)]["Page"]:
    print(page)
example = data[(truths)]





