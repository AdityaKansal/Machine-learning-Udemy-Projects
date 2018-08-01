# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 09:52:31 2018

@author: akansal2
"""

#importing libraries
import nltk
import random
import numpy as np
from bs4 import BeautifulSoup




#importing data
positive_reviews = BeautifulSoup(open('C:/A_stuff/Learning/Machine Learning/Udemy/NLP Basic/Sentiment Analyzer/electronics/positive.review').read())
positive_reviews = positive_reviews.find_all('review_text')



#creating trigrams(all possible combinations)
trigrams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) -2):
        k = (tokens[i],tokens[i+2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+1])
        
        

#trigrams probabilities
trigrams_probabilities = {}

for k,words in trigrams.items():
    if(len(set(words))) > 1:
        d = {}
        n= 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] +=1
            n+=1
        for w,c in d.items():
            d[w] = float(c)/n
        trigrams_probabilities[k] = d



#taking random sample
def random_sample(d):
    r = random.random()
    cumulative = 0
    for w,p in d.items():
        cumulative += p
        if r < cumulative:
            return w

def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print('Original : ',s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens)-2):
        if random.random() < 0.2:
            k = (tokens[i],tokens[i+2])
            if k in trigrams_probabilities:
                w = random_sample(trigrams_probabilities[k])
                tokens[i+1] = w
                
                
    print('Spun : ')
    print(' '.join(tokens).replace(' .','.').replace(" '","'").replace(' ,',',').replace('$ ','$').replace(' !','!'))


test_spinner()




##########################################################################################
#trying with five grams:


#creating trigrams(all possible combinations)
five_grams = {}
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) -4):
        k = (tokens[i],tokens[i+1],tokens[i+3],tokens[i+4])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i+2])
        
        

#trigrams probabilities
trigrams_probabilities = {}

for k,words in trigrams.items():
    if(len(set(words))) > 1:
        d = {}
        n= 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] +=1
            n+=1
        for w,c in d.items():
            d[w] = float(c)/n
        trigrams_probabilities[k] = d



#taking random sample
def random_sample(d):
    r = random.random()
    cumulative = 0
    for w,p in d.items():
        cumulative += p
        if r < cumulative:
            return w

def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print('Original : ',s)
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens)-2):
        if random.random() < 0.2:
            k = (tokens[i],tokens[i+2])
            if k in trigrams_probabilities:
                w = random_sample(trigrams_probabilities[k])
                tokens[i+1] = w
                
                
    print('Spun : ')
    print(' '.join(tokens).replace(' .','.').replace(" '","'").replace(' ,',',').replace('$ ','$').replace(' !','!'))


test_spinner()






























        
        
        

