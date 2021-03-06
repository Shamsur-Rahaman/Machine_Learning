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

#Importing dataset
dataset = pd.read_csv('/home/user/Documents/documents/Programming_codes/Python/mechine_learning_A-Z/part_2_regression/section_6_simple_linear_regression/Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# Taking care of missing data
"""
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
"""

# Categorical data
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Encode Country Column
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
# Encode purchase column
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)
"""

# Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
"""

# Simple Linear Regression
from sklearn.linear_model import LinearRegression
regresser = LinearRegression()

# Fit data into model
regresser.fit(X_train, y_train)

# Predict data
y_pred = regresser.predict(X_test)

# Making plot for training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regresser.predict(X_train), color='blue')
plt.title("Salary vs Experience(For training set)")
plt.xlabel("Experience")
plt.ylable("Salary")
plt.show()

# Making plot for test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regresser.predict(X_train), color='blue')
plt.title("Salary vs Experience(For training set)")
plt.xlabel("Experience")
plt.ylable("Salary")
plt.show()

















