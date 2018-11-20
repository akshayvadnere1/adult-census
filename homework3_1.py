#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:40:03 2018

@author: akshayvadnere
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

pd.set_option('display.max_columns', 14)
column = ["age","workclass", "fnlwgt","education", "education_num", "marital_status", "occupation", "relationship","race", "sex", "capital_gain","captial_loss", "hours_per_week", "native_country"]
data = pd.read_csv("/Users/akshayvadnere/Desktop/monaaaaaaaaaaaaaaaaaaaaaaaaa/adult_data.csv", names = column, index_col = False)

data.workclass.replace("\?", "Private", inplace = True, regex = True)

data.replace('\?', np.nan, inplace = True, regex = True)

mode_occupation = data.occupation.dropna().mode().to_string()
#print(type(mode_occupation))
data.occupation = data.occupation.fillna(mode_occupation)

mode_country = data.native_country.dropna().mode().to_string()
print(mode_country)
data.native_country = data.native_country.fillna(mode_country)


#print(data.head(200))