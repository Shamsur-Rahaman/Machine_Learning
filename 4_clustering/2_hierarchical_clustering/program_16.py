#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 07:27:44 2020

@author: Shamsur Rahaman

Importing Hierarchical Cluster

"""
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import Dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [2,3]].values

# Finding Optimal Cluster
from scipy.cluster.hierarchy import dendrogram, linkage
linked = linkage(X, method='ward')
dendrogram(linked,show_leaf_counts=True)
plt.show()

# Finding Optimal Cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Finding optimal cluster")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# Fitting hierarchical cluster at mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean',
                          linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visulasing the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = '1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = '2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = '3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = '4')
plt.title("Cluster with HC")
plt.xlabel("Annual Salary")
plt.ylabel("Spending score (1-100)")
plt.legend()
plt.show()























