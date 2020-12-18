#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 06:57:27 2020

@author: Shamsur Rahaman

Implimenting KMeans Cluster

"""

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Datasets
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [2,3]].values

# Finding Optimal number of cluster
from sklearn.cluster import KMeans 
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10,
                    max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Visualising optimal cluster
plt.plot(range(1,11),wcss)
plt.xticks(range(1, 11))
plt.title("Kmean Cluster Optimal Cluster")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()    

# Fit and Predict result with optimal cluster
kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init = 10,
                max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising result for KMeans
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red',
            label='Cluster_1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue',
            label='Cluster_2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green',
            label='Cluster_2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'pink',
            label='Cluster_2')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'violet',
            label='Cluster_2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s = 100, c = 'yellow', label="Centroids")
plt.title("Cluster of Clients")
plt.xlabel("Annual Income K$")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

