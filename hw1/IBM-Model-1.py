
# coding: utf-8

# In[1]:

import os
import sys
import numpy as np
import collections
import cPickle
import time


# In[2]:

start_t = time.time()
path_to_data = 'data/dev-test-train.de-en'


# In[6]:

lines = [line.strip().split('|||') for line in open(path_to_data, 'r')]


# In[7]:

german_lines = [collections.Counter(line[0].strip().lower().split()) for line in lines]
english_lines = [collections.Counter(line[1].strip().lower().split()) for line in lines]


# In[8]:

assert len(german_lines) == len(english_lines) == len(lines)


# In[9]:

english_vocab = {}
word_count = 0
for line in english_lines:
    for word in line:
        if word not in english_vocab:
            english_vocab[word] = word_count
            word_count += 1


# In[10]:

german_vocab = {}
word_count = 0
for line in german_lines:
    for word in line:
        if word not in german_vocab:
            german_vocab[word] = word_count
            word_count += 1


# In[11]:

uniform_probability = 1.0 / len(german_vocab)
def func():
    return uniform_probability

t = collections.defaultdict(func) 


# In[12]:

print "start EM. time: %d" %(time.time()-start_t)
num_iter = 72

prev_prob = uniform_probability
for ind in range(num_iter):

    count = collections.defaultdict(float)
    total = collections.defaultdict(float)
    index = 0
    
    for english_sentence, german_sentence in zip(english_lines, german_lines):
        index += 1
        if index%1000 == 1:
            print "completed sentence: %d, %d " %(index,time.time()-start_t) 
    
        for english_word in english_sentence:
            total_sentence = 0
            for german_word in german_sentence:
               
                total_sentence += t[(english_word, german_word)]*english_sentence[english_word]
            
            for german_word in german_sentence:
                x = t[(english_word, german_word)]*english_sentence[english_word]*german_sentence[german_word]/ total_sentence
                count[(english_word, german_word)] += x
                total[german_word] += x

    for english_word, german_word in count.keys():
        t[(english_word, german_word)] = count[(english_word, german_word)] / total[german_word]
    
    print "EM completed. time: %d" %(time.time()-start_t)


with open('t72.pkl','w') as f:
    cPickle.dump(t, f)


german_lines = [line[0].strip().lower().split() for line in lines]
english_lines = [line[1].strip().lower().split() for line in lines]

f = open('output.txt','w')
for english_sent, german_sent in zip(english_lines, german_lines):
    aligned_words = []
    for ind1, eng_word in enumerate(english_sent):
        translation_scores = [t[(eng_word, german_word)] for ind2, german_word in enumerate(german_sent)]
        aligned_word = np.argmax(np.array(translation_scores).astype(np.float32))
        aligned_words.append('%d-%d' % (aligned_word, ind1) )
    f.write(' '.join(aligned_words) + '\n')
f.close()
