# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:55:54 2018

@author: akansal2
"""

#importing libraries
import pandas as pd
import numpy as np
import tensorflow as tf
import keras





#getting dataset
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)\\Churn_Modelling.csv')

#Data Preporcessing
#Removing unnnecessary columns
Dataset = Dataset.drop(['CustomerId','RowNumber','Surname'],axis = 1)

#identifying the categorical variables
Dataset.info()


#converting all categorcial variables to numbers
from sklearn.preprocessing import LabelEncoder
le_Country = LabelEncoder()
le_Gender = LabelEncoder()
Dataset['Geography'] = le_Country.fit_transform(Dataset['Geography'].values)
Dataset['Gender'] = le_Gender.fit_transform(Dataset['Gender'].values)


#get dummies method
temp =  pd.get_dummies(Dataset['Geography'],drop_first = True,prefix = 'Geography_dummies')
Dataset = pd.concat([Dataset,temp],axis= 1)
Dataset.drop(['Geography'],axis = 1, inplace = True)


#Getting X and y matrix
X = Dataset.iloc[:,:].values
X = np.delete(X,9,1)
y = Dataset.iloc[:,-3].values


#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


#dividing into Xtrain and X test
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


#applying classifier now
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()

#Adding layers - First hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu',input_dim = 11))

#Adding layers - First second layer
classifier.add(Dense(output_dim = 6, init = 'uniform',activation = 'relu'))


# adding final layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))


#compiling ANN
classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])



#fiting the classifier
classifier.fit(X_train,y_train,batch_size = 10,nb_epoch = 100)


#predicting the values
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
for i in range(y_pred.shape[0]):
    if y_pred[i] > 0.5:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
        


#confusiojn matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
























