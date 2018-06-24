# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:17:46 2018

@author: akansal2
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold = 100)     #to increase the view of ndarray

#importing data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 1 - Data Preprocessing\\Section 2 -------------------- Part 1 - Data Preprocessing --------------------\\Data.csv')
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values



#Spilitting data into Traning and Test data set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""





