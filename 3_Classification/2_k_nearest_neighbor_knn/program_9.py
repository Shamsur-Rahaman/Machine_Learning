#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 16:54:48 2020

@author: Shamsur

Implimenting K-Nearest Neighbor Algorithm

"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25,
                                                    random_state = 0)

# Scaling train and test IV
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)

# Fitting classifier to train set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,p=2, metric='minkowski')
classifier.fit(X_train, y_train)

# Predicting IV test set
y_pred = classifier.predict(X_test)

# Confusion Matrix, apply on y_test and y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Make visualization for train set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1,
                               stop  = X_set[:, 0].max()+1,
                               step  = 0.01),
                     np.arange(start = X_set[:, 1].min()-1,
                               stop  = X_set[:, 1].max()+2,
                               step  = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                          X2.ravel()]).T).reshape(X1.shape),
                                        alpha = .75, cmap = ListedColormap(("red","green")))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],
                X_set[y_set == j,1],
                c = ListedColormap(("red","green"))(i),
                label=j)
plt.title("KNN classification on training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()


# Make visualization for test set
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min()-1,
                               stop  = X_set[:, 0].max()+1,
                               step  = 0.01),
                     np.arange(start = X_set[:, 1].min()-1,
                               stop  = X_set[:, 1].max()+2,
                               step  = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
                                                          X2.ravel()]).T).reshape(X1.shape),
                                        alpha = .75, cmap = ListedColormap(("red","green")))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],
                X_set[y_set == j,1],
                c = ListedColormap(("red","green"))(i),
                label=j)
plt.title("KNN classification on training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()









