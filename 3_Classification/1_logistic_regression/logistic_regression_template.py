#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 12:15:34 2020

@author: Shamsur
"""

# ------ Logistic Regression Template ---------


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Handling Missing data
# ----- No missing Data here -----

# Spliting Dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)

# Scaling Independent datasets both (train and test) 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)

# Fitting Logistic Regression on both Training sets (DV and IV)
# Create classifier here 

# Predicting Model on Independent Test set  
y_pred = classifier.predict(X_test)

# Making Confusion matrix for test sets (predicted_IV, DV) 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

# Visualising results on Training sets both DV and IV
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1,
                               stop  = X_set[:, 0].max() +1,
                               step  = 0.01),
                     np.arange(start = X_set[:, 0].min() -1,
                               stop  = X_set[:, 0].max() +1,
                               step  = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                  X2.ravel()]).T).reshape(X1.shape),
                                        alpha = 0.75,
                                        cmap  = ListedColormap(("red","green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(("red","green"))(i),
                label = j)
plt.title("Logistic Regression on Training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Visualising results on Test sets both DV and IV
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1,
                               stop  = X_set[:, 0].max() +1,
                               step  = 0.01),
                     np.arange(start = X_set[:, 0].min() -1,
                               stop  = X_set[:, 0].max() +1,
                               step  = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                  X2.ravel()]).T).reshape(X1.shape),
                                        alpha = 0.75,
                                        cmap  = ListedColormap(("red","green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0],
                X_set[y_set == j, 1],
                c = ListedColormap(("red","green"))(i),
                label = j)
plt.title("Logistic Regression on Test set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()




























