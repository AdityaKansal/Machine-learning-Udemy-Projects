# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:11:06 2018

@author: akansal2
"""

#importing libraries
import numpy as np
import nltk




#importing nltk brown corpus
from nltk.corpus import brown




#working with model
print(brown.categories())
corpus = list(brown.words(categories ='news'))
#corpus[10:40]
#corpus = corpus.lower()

#vocabulary for smoothing. It is length of ditinct words
V = len(set(corpus))

#create list of all bigrams
bigram_list = []
for i in range(len(corpus)-1):
    bigram = corpus[i] + ' ' + corpus[i+1]
    bigram_list.append(bigram)


#function to modify the inout setence
def sentence_prob(sentence):
    s1 = sentence.split()
    length = len(s1)
    prob_sum=0
    for i in range(length):
        if i==0:
            #probability of first word
            word_count = corpus.count(s1[0])
            prob = (word_count +1)/(len(corpus)+ V)
        else:
            #prob of bigram for i and i-1
            bigram = s1[i-1] + ' ' + s1[i]
            preceeding_word = s1[i-1]
            preceeding_word_count= corpus.count(preceeding_word)
            bigram_count = bigram_list.count(bigram)
            prob = (bigram_count+1)/(preceeding_word_count+V)
            
        log_prob = np.log2(prob)
        prob_sum +=log_prob
        
    prob_sum/=length
    return prob_sum
    
        


#find the probabilty of sentence
s1 = 'The jury'
print('The prob of this sentence is %s'%sentence_prob(s1))
