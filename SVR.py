# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:19:51 2018

@author: akansal2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:17:40 2018

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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

#letscreate the new regression and fit the X,y
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)


#predicting result
y_pred = sc_y.inverse_transform(regressor.predict((sc_X.transform(np.array([[6.5]])))))



#plot Lin_reg_2

plt.scatter(X,y,color = 'Red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show



#fitting and plotting for higher resolution
X_grid = np.arange(min(X),max(X),0.1)
X_grid.reshape(len((X_grid)),1)
plt.scatter(X_grid,y,color = 'Red')
plt.plot(X_grid,regressor.predict(X_grid), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show




