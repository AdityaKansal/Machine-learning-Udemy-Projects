# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:11:05 2018

@author: akansal2
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE





#Data

doc1 = 'Data Science Machine Learning'
doc2 = 'Money fun family kids home'
doc3 = 'Programming Java Data Structures'
doc4 = 'love food health games energy fun'
doc5 = 'Algorithm Data Computers fun'


doc_complete = [doc1,doc2,doc3,doc4,doc5]



#tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(doc_complete).todense().T

Vocab = vectorizer.get_feature_names()

idx2word = {i:w for i,w in enumerate(Vocab)}

#tsne
tsne = TSNE()
Z = tsne.fit_transform(X)
plt.scatter(Z[:,0], Z[:,1])

            
#annotation
for i in range(V):
        try:
            plt.annotate(s=idx2word[i].encode("utf8").decode("utf8"), xy=(Z[i,0], Z[i,1]))
        except:
            print("bad string:", idx2word[i])
plt.draw()
            
            
    