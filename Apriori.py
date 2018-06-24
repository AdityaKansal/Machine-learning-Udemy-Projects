# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:08:04 2018

@author: akansal2
"""
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Importing dataset
Dataset = pd.read_csv('C:\\A_stuff\\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 5 - Association Rule Learning\\Section 28 - Apriori\\Market_Basket_Optimisation.csv',header = None)
Transactions = []


for i in range(0,7501):
    Transactions.append([str(Dataset.values[i,j]) for j in range(0,20)])
        
#fitting apriori
from apyori import apriori
rules = apriori(Transactions,min_support =0.003,min_confidence =0.2 , min_lift = 3, min_length =2 )


#visualizing
results = list(rules)