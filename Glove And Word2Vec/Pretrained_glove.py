# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:54:14 2018

@author: akansal2
"""


#importing libraries
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances




#function for calculating distance
#Euclidean Distance
def dist1(a,b):
    return np.linalg.norm(a-b)

#cosine distance
def dist2(a,b):
    return 1- np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


#pick a distance type
dist,metric = dist2,'cosine'




#find analgoies
#more intutive but with for loop and hence slower
def find_analogies1(w1,w2,w3):
    for w in (w1,w2,w3):
        if w not in word2vec:
            print('%s not in dictionary' %w)
            return
    
    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman
    
    
    min_dist = float('inf')
    best_word = ''
    for word,v1 in word2vec.items():
        if word not in (w1,w2,w3):
            d = dist(v0,v1)
            if d < min_dist:
                min_dist = d
                best_word = word
                
    
    print(w1 ,"-",w2,"=",best_word,"-",w3)




#less intutive,without for loops, a vectorzied approach and hence faster
def find_analogies(w1,w2,w3):
    for word in (w1,w2,w3):
        if word not in word2vec:
            print('%s not in dictionary' %word)
            return
        
    
    king = word2vec[w1]
    man = word2vec[w2]
    woman = word2vec[w3]
    v0 = king - man + woman
    
    
    
    distances = pairwise_distances(v0.reshape(1,D),embedding,metric= metric).reshape(V)
    idxs = distances.argsort()[:4]
    for id in idxs:
        word = idx2word[id]
        if word not in (w1,w2,w3):
            best_word = word
            break
    
    print(w1, "-", w2, "=", best_word, "-", w3)




#finding nearest neighboaurs
def nearest_neighbours(word,n):
    if word not in word2vec:
        print('%s not in dictionary'%word)
        return
    v1 = word2vec[word]
    distances = pairwise_distances(v1.reshape(1,D),embedding,metric = metric).reshape(V)
    idxs = distances.argsort()[1:n+1]
    for id in idxs:
        print(idx2word[id])
    




#loading pretrained vectors
print('loading pretrained vectors ....')
word2vec = {}
embedding = []
idx2word = []
with open('C:/A_stuff/Learning/Machine Learning/Udemy/NLP 2/glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:],dtype = 'float32')
        word2vec[word] = vec
        embedding.append(vec)
        idx2word.append(word)
        
        
        
print('Found %s word vectors'%len(word2vec))
embedding = np.asarray(embedding)
V,D = embedding.shape


find_analogies('king', 'man', 'woman')
find_analogies('france', 'paris', 'london')
find_analogies('france', 'paris', 'rome')
find_analogies('paris', 'france', 'italy')
find_analogies('france', 'french', 'english')
find_analogies('japan', 'japanese', 'chinese')
find_analogies('japan', 'japanese', 'italian')
find_analogies('water', 'ice', 'water')


nearest_neighbours('china',5)









l = (1,2)
print(type(l))



        
    