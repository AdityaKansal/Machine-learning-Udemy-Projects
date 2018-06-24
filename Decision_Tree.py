# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:01:39 2018

@author: akansal2
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold = 100)     #to increase the view of ndarray
#importing data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 2 - Regression\\Section 6 - Polynomial Regression\\Position_Salaries.csv')
X = Dataset.iloc[:,1:2].values
y = Dataset.iloc[:,2].values

#Spilitting data into Traning and Test data set
"""from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)"""

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


#letscreate the new regression and fit the X,y
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state= 0)
regressor.fit(X,y)

#predicting result
y_pred = regressor.predict((6.5))



#plot Lin_reg_2

plt.scatter(X,y,color = 'Red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show



#fitting and plotting for higher resolution
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len((X_grid)),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show




