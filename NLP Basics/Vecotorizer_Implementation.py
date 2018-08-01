# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 16:08:04 2018

@author: akansal2
"""
import numpy as np

data = ['hello hello i','hello how','how am i']

#building a vocabulary
V = []
for sent in data:
    for word in sent.split():
        word = word.lower()
        V.append(word)
V = list(set(V))

#creating a matrix of zeros
rows = len(data)
columns = len(V)
X_binary = np.zeros((rows,columns))
X_count = np.zeros((rows,columns))
X_tfidf = np.zeros((rows,columns))

#deifining dictionary for word to vector mapping
word_map_vector = {}
i =0
for word in V:
    word_map_vector[word] = i
    i +=1
       

#binary vectorizer
for word in V:
    for sent in data:
        if word in sent:
            X_binary[data.index(sent),word_map_vector[word]] = 1
            
    

#count vectorizer
for sent in data:
    for word in sent.split():    
        if word in V:
            X_count[data.index(sent),word_map_vector[word]] += 1



#tfidf vecotrizer
'''
similar way , iterate thru the words and calculate
1) term freq in every sentence
2) log of (total documents/no of documents containing that word)
3) multiple both to give a better number
'''
