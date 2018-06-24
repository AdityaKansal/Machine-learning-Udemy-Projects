# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 10:31:30 2018

@author: akansal2
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#importing data set
Dataset =  pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv')


#designing feature matrix
X = Dataset.iloc[:,[3,4]].values


#Drawing Dendograms to find number of optimal clusters
import scipy.cluster.hierarchy as sch
Dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendograms  ')
plt.xlabel('observations')
plt.ylabel('Euclidian Distance')
plt.plot()
plt.show()


#fitting the HC to data
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X) 

#plotting the graph for final visualiazation
plt.scatter(X[y_hc== 0,0],X[y_hc == 0,1], s=50, color = 'red',label = 'Cluster 1')
plt.scatter(X[y_hc== 1,0],X[y_hc == 1,1], s=50, color = 'blue',label = 'Cluster 2')
plt.scatter(X[y_hc== 2,0],X[y_hc == 2,1], s=50, color = 'green',label = 'Cluster 3')
plt.scatter(X[y_hc== 3,0],X[y_hc == 3,1], s=50, color = 'cyan',label = 'Cluster 4')
plt.scatter(X[y_hc== 4,0],X[y_hc == 4,1], s=50, color = 'magenta',label = 'Cluster 5')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.title('Data')

plt.show()

