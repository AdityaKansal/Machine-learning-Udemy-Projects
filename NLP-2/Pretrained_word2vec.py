# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 20:18:21 2018

@author: akansal2
"""

#importing libraires
import gensim
from gensim.models import KeyedVectors



#loading pretrained vectors
word_vectors = KeyedVectors.load_word2vec_format('C:/A_stuff/Learning/Machine Learning/Udemy/NLP 2/GoogleNews-vectors-negative300.bin',binary = True)


#find analogies
def find_analogies(w1,w2,w3):
    r = word_vectors.most_similar(positive=[w1,w3],negative=[w2])
    print('%s - %s = %s -%s'%(w1,w2,r[0][0],w3))





#find nearest neighbours
def nearest_neighbours(w1):
    r = word_vectors.most_similar(positive =[w1])
    print('Nearest neighbours of %s are'%w1)
    for word,score in r:
        print('\t%s'%word)
        



nearest_neighbours('king')
find_analogies('france', 'paris', 'london')