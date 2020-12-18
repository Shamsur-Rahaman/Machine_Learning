#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 20:24:57 2020

@author: Shamsur

Implimenting SVM classifier
"""
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Dataset 
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting Dataset in training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.25,
                                                    random_state = 0)

# Scaling I.V
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)

# Fitting SVM classifier to training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting classifier result on X_test
y_pred = classifier.predict(X_test)

# Confusion_matrix for y_test, y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising training set
from matplotlib.colors import ListedColormap
x_set, y_set = X_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() -1,
                               stop  = x_set[:, 0].max() +1,
                               step  = 0.01),
                     np.arange(start = x_set[:, 1].min() -1,
                               stop  = x_set[:, 1].max() +1,
                               step  = 0.01)
                    )
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(),
                                                  x2.ravel()]).T).reshape(x1.shape),
                                        alpha = .75,
                                        cmap  = ListedColormap(("red","green"))
                                        )
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],
                x_set[y_set == j,1],
                c = ListedColormap(("red","green"))(i),
                label = j)

plt.title("SVM classifier for training set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Visualising test set
from matplotlib.colors import ListedColormap
xset, yset = X_test, y_test
x1, x2 = np.meshgrid(np.arange(start = xset[:,0].min()-1,
                               stop  = xset[:,0].max()+1,
                               step  = 0.01),
                     np.arange(start = xset[:,1].min()-1,
                               stop  = xset[:,1].max()+1,
                               step  = 0.01)
    )
plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),
                                                x2.ravel()]).T).reshape(x1.shape),
                                        alpha = 0.75,
                                        cmap  = ListedColormap(("red","green"))
                                        )
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

for i, j in enumerate(np.unique(yset)):
    plt.scatter(xset[yset == j,0],
                xset[yset == j,1],
                c = ListedColormap(("red","green"))(i),
                label = j)
    
plt.title("SVM classifier for test set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()















































