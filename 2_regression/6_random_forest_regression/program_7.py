#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 20:29:09 2020

@author: Shamsur

Making Random Forest Regression 
"""
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Building Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)


# Making prediction
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))


# Visualising with high resulation
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff(Random Forest Regressor)")
plt.xlabel("Position Level")
plt.ylabel("Salaries")
plt.show()
