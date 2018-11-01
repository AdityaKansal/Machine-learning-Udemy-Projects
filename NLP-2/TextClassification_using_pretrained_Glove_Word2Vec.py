# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 06:32:00 2018

@author: akansal2
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from gensim.models import KeyedVectors




#fetching data
train = pd.read_csv('C:/A_stuff/Learning/Machine Learning/Udemy/NLP 2/r8-train-all-terms.txt',header = None,sep = '\t')
test =  pd.read_csv('C:/A_stuff/Learning/Machine Learning/Udemy/NLP 2/r8-test-all-terms.txt',header = None,sep = '\t')

train.columns = ['label','content']
test.columns = ['label','content']





#GloVe Vecotizer
class GloveVectorizer():
    def __init__(self):
        #load in pretrained glove vectorizer
        print('loading pretrained glove vectors....')
        word2vec = {}
        embedding = []
        idx2word = []
        with open('C:/A_stuff/Learning/Machine Learning/Udemy/NLP 2/glove.6B.50d.txt',encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:],dtype = 'float32')
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        print('found %s word vectors.'%len(word2vec))
        
        
        
        #save for later
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v:k for k,v in enumerate(idx2word)}
        self.V,self.D = self.embedding.shape
        
        
    def fit(self,data):
        pass
    
    
    def transform(self,data):
        X = np.zeros((len(data),self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()
            vecs = []
            for word in tokens:
                if word in self.word2vec:    
                    vec = self.word2vec[word]
                    vecs.append(vec)
                
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis =0)
            else:
                emptycount +=1
            n+=1
        print('Number of samples with no words found : %s / %s' %(emptycount,len(data)))
        return X
        
    
    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)
    
    
    
   


#word2vec
class Word2VecVectorizer:
  def __init__(self):
    print("Loading in word vectors...")
    self.word_vectors = KeyedVectors.load_word2vec_format(
      'C:/A_stuff/Learning/Machine Learning/Udemy/NLP 2/GoogleNews-vectors-negative300.bin',
      binary=True
    )
    print("Finished loading in word vectors")

  def fit(self, data):
    pass

  def transform(self, data):
    # determine the dimensionality of vectors
    v = self.word_vectors.get_vector('king')
    self.D = v.shape[0]

    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split()
      vecs = []
      m = 0
      for word in tokens:
        try:
          # throws KeyError if word not found
          vec = self.word_vectors.get_vector(word)
          vecs.append(vec)
          m += 1
        except KeyError:
          pass
      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X


  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)



             
        
        
#Initializing vectorizer    
#vectorizer = GloveVectorizer()
vectorizer = Word2VecVectorizer()
Xtrain = vectorizer.fit_transform(train.content)
Ytrain = train.label

Xtest = vectorizer.transform(test.content)
Ytest = test.label




# create the model, train it, print scores
model = RandomForestClassifier(n_estimators=200)
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))










