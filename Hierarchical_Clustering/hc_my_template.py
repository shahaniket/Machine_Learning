# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 14:48:46 2018

@author: aniketsha
"""
print("Hello World")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendogram for finding optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Distance')
plt.show()

#Fitting HC to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#Displaying cluster
plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=100, c='red', label='Cluster 1')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=100, c='green', label='Cluster 3')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=100, c='magenta', label='Cluster 5')

plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Hierarchical clusters')
plt.legend()
plt.show()
