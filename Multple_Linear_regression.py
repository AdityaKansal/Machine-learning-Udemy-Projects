# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 07:11:59 2018

@author: akansal2
"""


#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold = 100)     #to increase the view of ndarray

#importing data set
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 2 - Regression\\Section 5 - Multiple Linear Regression\\50_Startups.csv')


X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:,-1].values



#categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(  categorical_features = [3] )
X = onehotencoder.fit_transform(X).toarray()


#avoiding dummy variable trap
X = X[:,1:]

#Spilitting data into Traning and Test data set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


# Fitting Multple Linear regression to the Traning Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#bulding the Optimal model using Backward Elmination
X= np.append(arr = np.ones((X.shape[0],1)).astype(int),values = X,axis=1)
import statsmodels.formula.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
regressor_OLS.summary()

print(X_opt)

#doing the backward elimiation using a loop
import statsmodels.formula.api as sm


def BackwardElimination(x,y,SL):
    regressor_OLS = sm.OLS(endog = y,exog = x).fit()
    print(regressor_OLS.summary())
    Pmax = max(regressor_OLS.pvalues.astype(float))
    Pmax_index = regressor_OLS.pvalues.argmax(axis = 0)   
    if(Pmax > SL):
        x = np.delete(x,Pmax_index,1)
        return BackwardElimination(x,y,SL)
    else:
        return x

X_opt = X[:,[0,1,2,3,4,5]]
SL = 0.05
X_Modeled = BackwardElimination(X_opt,y,SL)







