# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:49:32 2018

@author: akansal2
"""



#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold = 100)     #to increase the view of ndarray

#importing data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 2 - Regression\Section 4 - Simple Linear Regression\\Salary_Data.csv')
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,1].values



#Spilitting data into Traning and Test data set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#fitting training data to linear regrssion model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#predicting the values
y_pred = regressor.predict(X_test)


#plotting training set and the predicted line
plt.scatter(X_train,y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color ='blue')
plt.title('Plotting Training set')
plt.xlabel('Years of Exerience')
plt.ylabel('Salary')
plt.show()



#plotting test set and the predicted line
plt.scatter(X_test,y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color ='blue')
plt.title('Plotting Test set')
plt.xlabel('Years of Exerience')
plt.ylabel('Salary')
plt.show()









