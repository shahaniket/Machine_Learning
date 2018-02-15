# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:11:45 2018

@author: aniketsha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,0:-1].values
Y = dataset.iloc[:,4:5].values

#Encoding categrical data
#Encoding the dependent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features=[3])
X[:,3] = labelencoder.fit_transform(X[:,3])
X = onehotencoder.fit_transform(X).toarray()

#removing dummy variable trap
X = X[:,1:]

#Splitting dataset into training and testing set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

#multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the test results
y_predict = regressor.predict(X_test)

# Backward Elimination to build optimal model
import statsmodels.formula.api as sm
X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis = 1)

"""
#Step 1: all columns
X_opt = X[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================

Removing x2 as it has the highest P-value
"""

'''
#Step 2: remove x2
X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

 ==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.011e+04   6647.870      7.537      0.000    3.67e+04    6.35e+04
x1           220.1585   2900.536      0.076      0.940   -5621.821    6062.138
x2             0.8060      0.046     17.606      0.000       0.714       0.898
x3            -0.0270      0.052     -0.523      0.604      -0.131       0.077
x4             0.0270      0.017      1.592      0.118      -0.007       0.061
==============================================================================

Removing x1 as it has the highest P-value
'''

'''
#Step 3: remove x1
X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.012e+04   6572.353      7.626      0.000    3.69e+04    6.34e+04
x1             0.8057      0.045     17.846      0.000       0.715       0.897
x2            -0.0268      0.051     -0.526      0.602      -0.130       0.076
x3             0.0272      0.016      1.655      0.105      -0.006       0.060
==============================================================================

'''

'''
#Step 4: remove x2 (4)
X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()

==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       4.698e+04   2689.933     17.464      0.000    4.16e+04    5.24e+04
x1             0.7966      0.041     19.266      0.000       0.713       0.880
x2             0.0299      0.016      1.927      0.060      -0.001       0.061
==============================================================================
'''

#Step 5: remove x2 (5)
X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog=Y, exog=X_opt).fit()
regressor_ols.summary()