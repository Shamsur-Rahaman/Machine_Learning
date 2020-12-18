#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 19:10:03 2020

@author: Shamsur
"""
# SVR non-regression model implimentation
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import Dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Split dataset
# ---- No need in this case -----


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X    = sc_X.fit_transform(X)
y    = sc_y.fit_transform(y)


# Fitting SVR into dataset
from sklearn.svm import SVR
regression = SVR(kernel='rbf')
regression.fit(X,y)


# Predicting a new result
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))


# Visualising the SVR result
plt.scatter(X, y, color='red')
plt.plot(X, regression.predict(X), color='blue')
plt.title("Truth or bluff(SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()


# Visualising SVR result with higher resulation
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color='blue')
plt.title("Truth and Bluff(SVR)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()







































