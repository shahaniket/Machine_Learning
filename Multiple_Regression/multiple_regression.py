# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:41:41 2018

@author: aniketsha
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

from sklearn.cross_validation import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

y_pred = regressor.predict(X_Test)
