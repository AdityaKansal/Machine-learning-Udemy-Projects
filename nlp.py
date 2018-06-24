# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:51:32 2018

@author: akansal2
"""
#importing libraries
import pandas as pd
import numpy as np



#getting Dataset
Dataset = pd.read_csv('C:\\A_stuff\Learning\\Machine Learning\\Udemy- A-Z Machine Learning\\Machine Learning A-Z\\Part 7 - Natural Language Processing\\Section 36 - Natural Language Processing\\Restaurant_Reviews.tsv',delimiter='\t',quoting = 3)


#Data clean
# 1. Remove all !,.numbers.. Keep only letters and keep words distance with space
#2. Convert all in lower case
#3. Remove all prepositions and keep the main words
#3. Keep the root word

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = Dataset.iloc[i,0]
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


#reformating them in matrix or bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = Dataset.iloc[:,1]


#we have Data ready
#featue scaling not needed as all are 0 or 1
#Onehot encoder or label encoder also not needed
#null values also not there

#divide into traning and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


#fitting classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10,criterion = 'entropy',random_state = 0)
classifier.fit(X_train,y_train)



#predicting y_pred
y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)














