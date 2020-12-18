#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:02:23 2020

@author: Shamsur
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:26:26 2020

@author: Shamsur
"""

# Part-2/ Section-6/ Polynomial Linear Regression

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import statsmodels.api as sm

#Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

"""
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
"""


# Impliment Polynomial Regression
# first, make a linear regression for comparison
# for polynomial first, make a polynomial coordinate (use fit_transform on independent variable)
# secondly call linear model and put polynomial coordinates in linear model (use only 
#                                                       fit on polynomial independent variable)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


# Visualizing Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title("Truth or Bluff(for linear model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
# Visualizing Polynomial Regression
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title("Truth or Bluff(for Polynomial model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


# For linear Regression Prediction
lin_reg.predict(6.5)
# For Polynomial Regression Prediction
lin_reg_2.predict(poly_reg.fit_transform(6.5))
















