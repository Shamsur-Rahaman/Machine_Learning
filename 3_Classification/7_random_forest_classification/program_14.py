#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:49:12 2020

@author: Shamsur Rahaman

Implimenting Random Forest Classifier

"""

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting Dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,
                                                    random_state = 0)

# Scaling I.V.
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting Decision Tree classifier to training set
from  sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Predicting with classifier on X_test
y_pred = classifier.predict(X_test)

# Applying confusion_matrix on y_test and y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising training set result
from matplotlib.colors import ListedColormap
xset, yset = X_train, y_train
x1, x2 = np.meshgrid(np.arange(start = xset[:, 0].min()-1,
                               stop  = xset[:, 0].max()+1,
                               step  = 0.01),
                     np.arange(start = xset[:, 1].min()-1,
                               stop  = xset[:, 1].max()+1,
                               step  = 0.01)
)
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),
                                                x2.ravel()]).T).
                                                reshape(x1.shape),
                                                alpha = 0.75,
                                                cmap = ListedColormap(("red","green")))


plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j,0],
                xset[yset == j,1],
                c = ListedColormap(("red","green"))(i),
                label = j)

plt.title("Decision Tree Classifier on Training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Visualising test set result
from matplotlib.colors import ListedColormap
xset, yset = X_test, y_test
x1, x2 = np.meshgrid(np.arange(start = xset[:, 0].min()-1,
                               stop  = xset[:, 0].max()+1,
                               step  = 0.01),
                     np.arange(start = xset[:, 1].min()-1,
                               stop  = xset[:, 1].max()+1,
                               step  = 0.01)
)
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),
                                                x2.ravel()]).T).
                                                reshape(x1.shape),
                                                alpha = 0.75,
                                                cmap = ListedColormap(("red","green")))


plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j,0],
                xset[yset == j,1],
                c = ListedColormap(("red","green"))(i),
                label = j)

plt.title("Decision Tree Classifier on Training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()
























