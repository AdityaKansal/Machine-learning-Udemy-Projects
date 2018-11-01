# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 16:16:55 2018

@author: akansal2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#Ex1
#Eigen Vector
A = np.array([[0.3,0.6,0.2],[0.5,0.2,0.3],[0.4,0.1,0.5]])
v = np.array([1/3,1/3,1/3])


dist_values = []
for i in range(25):
    v_prime = v.dot(A)
    dist = np.linalg.norm(v_prime - v)
    dist_values.append(dist)
    v = v_prime
    
plt.plot(range(25),dist_values)
plt.show()    
    




#Ex2
#CLT
X = np.random.rand(100)
plt.hist(X)


Y = 0
Y_values = []
for i in range(10):
    X = np.random.rand(100)
    Y += X
    #Y_values.append(Y)
    
plt.hist(Y)
print(Y.mean())
print(Y.var())
    







#Ex3
#Mean image of MNSIT dataset
path = 'C:/A_stuff/Learning/Machine Learning/Udemy/Computer Vision OpenCV/MNSIT Dataset/'
df = pd.read_csv(path+'train.csv')
M = df.as_matrix()

#M.shape
#M[M[:,0] == 1][:,1:][0].shape
numbers = [1,7]
total = 0
for i in numbers:
    total += M[M[:,0] == int(i)][:,1:][0]
    
mean = total/len(numbers)
mean = mean.reshape(28,28)

plt.imshow(mean,cmap ='gray')



#Ex4
#Rotate clockwise image
M = df.as_matrix()
X = M[:,1:]
img = X[10].reshape(28,28)
plt.imshow(img,cmap ='gray')
#using numpy transpose
img_rotated = img.T
plt.imshow(img_rotated,cmap ='gray')

#using for loops
img_2 = np.zeros((28,28))
for i in range(img.shape[0]):
    img_2[:,i] = img[i,:]
        
plt.imshow(img_2,cmap ='gray')



#Ex5 
#Symmetric Matrix
#Just check the A == A.t



#Ex6
#Generate and plot XOR data set
data = np.random.rand(2,2)
np.interp(data,[0,1],[0.05,0.095])


x1 = [1,2,3,4]
x2 = [1,2,3,4]
y = [0,1,1,0]

plt.scatter(x1,x2,c= y,cmap = 'bwr_r')
















































    
    