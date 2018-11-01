# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 20:04:17 2018

@author: akansal2
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime

from Brown import get_sentences_with_word2idx_limit_vocab, get_sentences_with_word2idx

from Markov import get_bigram_probs



'''
Earlier we caculated bigram prob by taking count of A and count of A-->B
With this we are able to calculate the probability of given sentence

We could draw a bigram probablity matirx which tell the prob of each bigram


Now we want to repressent each word using one hot encoding
that would be the input
and predict the next word

We shall loop over the corpus.. and train our model to get the relationship between bigrams in form of W matrix


'''




if __name__ == '__main__':
    
    sentences, word2idx = get_sentences_with_word2idx_limit_vocab(2000)
    
    #Vocab Size
    V = len(word2idx)
    
    start_idx = word2idx['START']
    end_idx = word2idx['END']
    
    # a matrix where:
    # row = last word
    # col = current word
    # value at [row, col] = p(current word | last word)
    ''' 
    bigram_probs = get_bigram_probs(sentences,V,start_idx,end_idx,smoothing = 0.1)
    ''' 
    
    
    #train a logistic model
    W = np.random.randn(V,V)/np.sqrt(V)
    
    
    
    losses = []
    epochs = 1
    lr = 1e-1
    
    
    def softmax(a):
        a = a- a.max()
        exp_a = np.exp(a)
        return exp_a/exp_a.sum(axis = 1,keepdims = True)
    
    
    
    
    #what is the loss if we set W = log(bigram_probs)
    ''''
    W_bigram = np.log(bigram_probs)
    bigram_losses = []
    '''
    
    t0 = datetime.now()
    
    for epoch in range(epochs):
        random.shuffle(sentences)
        
        
        
        j = 0 #keep track of iterations
        
        for sentence in sentences:
            
            #convert sentence into one hot encoded inputs and targets
            sentence = [start_idx] +    sentence + [end_idx]
            n = len(sentence)
            
            print(n)
            print(sentence)
            print(np.arange(n-1))
            print(sentence[:n-1])
            
            break
            ''' 
            inputs = np.zeros((n-1,V))
            targets = np.zeros((n-1,V))
            
            inputs[np.arange(n-1),sentence[:n-1]] = 1
            targets[np.arange(n-1),sentence[1:]] =1
            
           
            
            #get output predictions
            predictions = softmax(inputs.dot(W))
            
            
            #do a gradient descent step
            W = W - lr*inputs.T.dot(predictions - targets)
            
            
            
            #keep track of losses
            loss = -np.sum(targets*np.log(predictions))/(n-1)
            losses.append(loss)
            
            
            
            #keep track of bigram loss
            #only do it for the first epoch to avoid redundancy
            if epoch == 0:
                bigram_predictions = softmax(inputs.dot(W_bigram))
                bigram_loss = -np.sum(targets*np.log(bigram_predictions))/(n-1)
                bigram_losses.append(bigram_loss)
                
                
                
            if j%10 == 0:
                print("epoch:", epoch, "sentence: %s/%s" % (j, len(sentences)), "loss:", loss)
            j+=1
            
            
            print("Elapsed time training:", datetime.now() - t0)
        plt.plot(losses)
        plt.show()
                
            
            
            
        '''
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
































    
    
    
    
    
    
    







