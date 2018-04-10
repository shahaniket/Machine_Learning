# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 15:15:19 2018

@author: aniketsha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Loading
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values

#Data Pre-Processing 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])

labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#ANN
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialising the ANN
classifier = Sequential()

#input layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim = 11))

#output from input layer, input to hidden layer
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

#output layer
classifier.add(Dense(output_dim=1,init='uniform', activation='sigmoid'))

#compiling ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

#fitting ANN
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)

#prediction
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)