# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:24:43 2018

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

# Dividing into test set and traning set is not required as we have very less data
#feature scaling is not required

#lets fit Linea reqgression first and plot it
from sklearn.linear_model import LinearRegression
Lin_Reg = LinearRegression()
Lin_Reg.fit(X,y)

#lets transform the X matrix to X_poly and then fit it to linear regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

#lets fit the new X_poly into linear regression model
from sklearn.linear_model import LinearRegression
Lin_reg_2 = LinearRegression()
Lin_reg_2.fit(X_poly,y)


#plot Lin_reg
plt.scatter(X,y,color = 'Red')
plt.plot(X,Lin_Reg.predict(X), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show

#plot Lin_reg_2

plt.scatter(X,y,color = 'Red')
plt.plot(X,Lin_reg_2.predict(X_poly), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show



#predict using Lin_Reg
Lin_Reg.predict(6.5)

#predicting using Lin_reg_2
Lin_reg_2.predict(poly_reg.fit_transform(6.5))

#fitting Poly to X_grid
X_grid = np.arange(min(X),max(X),0.1)
X_grid.reshape(len((X_grid)),1)
X_grid_poly = poly_reg.fit_transform(X_grid)
plt.scatter(X_grid,y,color = 'Red')
plt.plot(X,Lin_reg_2.predict(X_grid_poly), color = 'blue')
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show




