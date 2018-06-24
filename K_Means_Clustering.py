# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:03:20 2018

@author: akansal2
"""

#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#importing dataset
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 4 - Clustering\\Section 24 - K-Means Clustering\\Mall_Customers.csv')

#Getting X features(last two columns)
X = Dataset.iloc[:,[3,4]].values
print(X[:,0])

#ploting this data
plt.scatter(X[:,0],X[:,1])
plt.title('Raw Data')
plt.xlabel('Annual Income')
plt.ylabel('Spending score')
plt.show()


#checking WCSS and number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,max_iter=300,n_init=10,init = 'k-means++',random_state= 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plotting wcss
plt.plot(range(1,11),wcss)
plt.ylabel('wcss')
plt.xlabel('number of clusters')
plt.title('WCSS vs Number of cluster')
plt.grid()
plt.show()



# 5 is the apppropriate number of clusters
#lets fit K means with 5 clusters
kmeans = KMeans(n_clusters=5, max_iter=300, n_init=10,init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(X)


#plotting the graph for final visualiazation
plt.scatter(X[y_kmeans== 0,0],X[y_kmeans == 0,1], s=50, color = 'red',label = 'Cluster 1')
plt.scatter(X[y_kmeans== 1,0],X[y_kmeans == 1,1], s=50, color = 'blue',label = 'Cluster 2')
plt.scatter(X[y_kmeans== 2,0],X[y_kmeans == 2,1], s=50, color = 'green',label = 'Cluster 3')
plt.scatter(X[y_kmeans== 3,0],X[y_kmeans == 3,1], s=50, color = 'cyan',label = 'Cluster 4')
plt.scatter(X[y_kmeans== 4,0],X[y_kmeans == 4,1], s=50, color = 'magenta',label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s= 200,color = 'yellow',label = 'Centroids')
plt.xlabel('Annual income')
plt.ylabel('Spending score')
plt.legend()
plt.title('Data')

plt.show()





