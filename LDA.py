# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 10:29:28 2018

@author: akansal2
"""



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

#applying PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components= 2)
X_train = lda.fit_transform(X_train,y_train)
X_test = lda.transform(X_test)


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




