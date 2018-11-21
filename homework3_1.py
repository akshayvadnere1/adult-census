#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 20:40:03 2018

@author: akshayvadnere
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

pd.set_option('display.max_columns', 14)
column = ["age","workclass", "fnlwgt","education", "education_num", "marital_status", "occupation", "relationship","race", "sex", "capital_gain","captial_loss", "hours_per_week", "native_country","income"]
data = pd.read_csv("/Users/akshayvadnere/Desktop/monaaaaaaaaaaaaaaaaaaaaaaaaa/adult_data.csv", names = column, index_col = False)

data.workclass.replace("\?", "Private", inplace = True, regex = True)

data.replace('\?', np.nan, inplace = True, regex = True)

mode_occupation = data.occupation.dropna().mode().to_string()
#print(type(mode_occupation))
data.occupation = data.occupation.fillna(mode_occupation)

mode_country = data.native_country.dropna().mode().to_string()
#print(mode_country)
data.native_country = data.native_country.fillna(mode_country)

#print(data.head(200))
encoder = LabelEncoder()
data["workclass"] = encoder.fit_transform(data["workclass"])
data["education"] = encoder.fit_transform(data["education"])
data["marital_status"] = encoder.fit_transform(data["marital_status"])
data["occupation"] = encoder.fit_transform(data["occupation"])
data["relationship"] = encoder.fit_transform(data["relationship"])
data["race"] = encoder.fit_transform(data["race"])
data["sex"] = encoder.fit_transform(data["sex"])
data["native_country"] = encoder.fit_transform(data["native_country"])
data["income"] = encoder.fit_transform(data["income"])
'''
features = data[data.columns[:-1]]
scalar = StandardScaler()
features_scaled = scalar.fit_transform(features)
'''


#Test Dataset preprocessing Steps

#Remove first row which does not have values

data_test = pd.read_csv("/Users/akshayvadnere/Desktop/monaaaaaaaaaaaaaaaaaaaaaaaaa/adult_test.csv", names = column, index_col = False)
data_test.drop(data_test.index[0], inplace = True)

#replace "?" in attributes like "workclass", "occupation", "native_country" by appropriate values
data_test.workclass.replace("\?", "Private", inplace = True, regex = True)

data_test.replace('\?', np.nan, inplace = True, regex = True)

mode_occupation1 = data_test.occupation.dropna().mode().to_string()
#print(type(mode_occupation))
data_test.occupation = data_test.occupation.fillna(mode_occupation1)

mode_country1 = data_test.native_country.dropna().mode().to_string()
#print(mode_country1)
data_test.native_country = data_test.native_country.fillna(mode_country1)

data_test["workclass"] = encoder.fit_transform(data_test["workclass"])
data_test["education"] = encoder.fit_transform(data_test["education"])
data_test["marital_status"] = encoder.fit_transform(data_test["marital_status"])
data_test["occupation"] = encoder.fit_transform(data_test["occupation"])
data_test["relationship"] = encoder.fit_transform(data_test["relationship"])
data_test["race"] = encoder.fit_transform(data_test["race"])
data_test["sex"] = encoder.fit_transform(data_test["sex"])
data_test["native_country"] = encoder.fit_transform(data_test["native_country"])
data_test["income"] = encoder.fit_transform(data_test["income"])

#print(data_test.head(200))


#Build Predictive Models

# 1. Logistic Regression

X_train, y_train, X_test, y_test = data[data.columns[:-1]], data[data.columns[-1]], data_test[data_test.columns[:-1]], data_test[data_test.columns[-1]]

'''
logreg = LogisticRegression().fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Training set score: {:.3f}".format(logreg.score(X_test, y_test)))

linsvc = LinearSVC().fit(X_train,y_train)
print("Training set score: {:.3f}".format(linsvc.score(X_train, y_train)))
print("Training set score: {:.3f}".format(linsvc.score(X_test, y_test)))


dec_tree = DecisionTreeClassifier().fit(X_train,y_train)
print("Training set score: {:.3f}".format(dec_tree.score(X_train, y_train)))
print("Training set score: {:.3f}".format(dec_tree.score(X_test, y_test)))

ran_forest = RandomForestClassifier().fit(X_train,y_train)
print("Training set score: {:.3f}".format(ran_forest.score(X_train, y_train)))
print("Training set score: {:.3f}".format(ran_forest.score(X_test, y_test)))


mlp_class = MLPClassifier().fit(X_train,y_train)
print("Training set score: {:.3f}".format(mlp_class.score(X_train, y_train)))
print("Training set score: {:.3f}".format(mlp_class.score(X_test, y_test)))

'''







#print(data.head(200))