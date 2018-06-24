# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:47:49 2018

@author: akansal2
"""

# for non -linear separable data
#PCA and LDA was for linearly separable data
#PCA used only independent varaibles(unsupervised learning)
#LDA used both dependendent and independent variables(supervided learning)


#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#importing dataset
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 9 - Dimensionality Reduction\\Section 43 - Principal Component Analysis (PCA)\\Wine.csv')

#creating X and y matirx
X = Dataset.iloc[:,0:13].values
y = Dataset.iloc[:,13].values

#dividing into Test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state = 0)


#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#applying kernal PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components= 2,kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)


#checking variance
explained_variance=  pca.explained_variance_ratio_

#fitting classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train,y_train)

#predicting y
y_pred = classifier.predict(X_test)

#creating confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)




