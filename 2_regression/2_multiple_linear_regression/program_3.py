#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:02:23 2020

@author: Shamsur
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:26:26 2020

@author: Shamsur
"""

# Part-2/ Section-6/ Simple Linear Regression

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm

#Importing dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encode Country Column
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]


# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fitting Multiple Regression on training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the test size predict
y_pred = regressor.predict(X_test)


# Backwork Elemination Preparation
import statsmodels.formula.api as st 
import statsmodels.api as sa
X = np.append(arr = np.ones((50,1)).astype(int),values=X,axis=1)
# Backward Elimination Implimentation
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sa.OLS(endog = y, exog = X_opt)
multi_reg = regressor_OLS.fit()
multi_reg.summary()
# Backward Elimination Implimentation
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = np.array(X[:, [0, 1, 2, 3]], dtype=float)
regressor_OLS = sa.OLS(endog = y, exog = X_opt)
multi_reg = regressor_OLS.fit()
multi_reg.summary()





























































































