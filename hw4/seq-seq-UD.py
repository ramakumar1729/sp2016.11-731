# coding: utf-8

import os
import sys
import numpy as np
import theano.tensor as T
import codecs
import theano.sandbox.cuda
theano.sandbox.cuda.use("gpu2")
from collections import Counter
import math
import copy


from network import LSTM
from layer import HiddenLayer, EmbeddingLayer
from learning_method import LearningMethod

import ConfigParser

def load_data(fname):
    sents = list()
    try:
        with codecs.open(fname,'r',encoding='utf-8') as f:
            for line in f:
                sents.append(line.strip().split())
    except IOError:
        sents = list()
    return sents

def get_words(sents):
    words = set()
    for sent in sents:
        for word in sent:
            words.add(word)
    words.add('<s>')
    words.add('</s>')
    return words

def bleu_stats(hypothesis, reference):
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1,5):
        s_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in xrange(len(hypothesis)+1-n)])
        r_ngrams = Counter([tuple(reference[i:i+n]) for i in xrange(len(reference)+1-n)])
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis)+1-n, 0]))
    return stats


# In[43]:

def bleu(stats):
    if len(filter(lambda x: x==0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum([math.log(float(x)/y) for x,y in zip(stats[2::2],stats[3::2])]) / 4.
    return math.exp(min([0, 1-float(r)/c]) + log_bleu_prec)


# In[46]:

def get_validation_bleu(hypotheses):
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, dev_tgt):
        stats += np.array(bleu_stats(hyp, ref))
    return "%.2f" % (100*bleu(stats))


def get_validation_predictions():
    validation_predictions = []    
    for ind, sent in enumerate(dev_src[:100]):
        
        if ind % 300 == 0:
            print ind, len(dev_src)
        src_words = np.array([source_word2ind[x] for x in sent]).astype(np.int32)
        current_outputs = [target_word2ind['<s>']]

        while True:
            next_word = f_eval(src_words, current_outputs).argmax(axis=1)[-1]
            current_outputs.append(next_word)
            #print [target_ind2word[x] for x in current_outputs]
            if next_word == target_word2ind['</s>'] or len(current_outputs) >= 15:
                validation_predictions.append([target_ind2word[x] for x in current_outputs])
                break
    return validation_predictions

# In[39]:

def get_test_predictions():
    test_predictions = []    
    for ind, sent in enumerate(test_src):
        
        if ind % 300 == 0:
            print ind, len(test_src)
        src_words = np.array([source_word2ind[x] for x in sent]).astype(np.int32)
        current_outputs = [target_word2ind['<s>']]

        while True:
            next_word = f_eval(src_words, current_outputs).argmax(axis=1)[-1]
            current_outputs.append(next_word)
            #print [target_ind2word[x] for x in current_outputs]
            if next_word == target_word2ind['</s>'] or len(current_outputs) >= 15:
                test_predictions.append([target_ind2word[x] for x in current_outputs])
                break
    return test_predictions

def main():
    config =  ConfigParser.ConfigParser()
    train_src = load_data(config.get('Data','train_src'))
    dev_src   = load_data(config.get('Data','dev_src'))
    test_src  = load_data(config.get('Data','test_src'))

    train_tgt = load_data(config.get('Data','train_tgt'))
    dev_tgt   = load_data(config.get('Data','dev_tgt'))
    test_tgt  = load_data(config.get('Data','test_tgt'))

    assert len(train_src) == len(train_tgt)

    UD_path = config.get('Path','UD')

    sys.path.append(UD_path+'/')

    words_src = get_words(train_src + dev_src)
    words_tgt = get_words(train_tgt + dev_tgt)

    source_word2ind = {word:ind for ind, word in enumerate(words_src)}
    source_ind2word = {ind:word for ind, word in enumerate(words_src)}
    target_word2ind = {word:ind for ind, word in enumerate(words_tgt)}
    target_ind2word = {ind:word for ind, word in enumerate(words_tgt)}


    # In[24]:
    
    #
    # Model
    #
    src_emb_dim      = 256  # source word embedding dimension
    tgt_emb_dim      = 256  # target word embedding dimension
    src_lstm_hid_dim = 512  # source LSTMs hidden dimension
    tgt_lstm_hid_dim = 2 * src_lstm_hid_dim  # target LSTM hidden dimension
    proj_dim         = 104  # size of the first projection layer
    dropout          = 0.5  # dropout rate
    
    n_src = len(source_word2ind)  # number of words in the source language
    n_tgt = len(target_word2ind)  # number of words in the target language
    
    # Parameters
    params = []
    
    # Source words + target words embeddings layer
    src_lookup = EmbeddingLayer(n_src, src_emb_dim, name='src_lookup') # lookup table for source words
    tgt_lookup = EmbeddingLayer(n_tgt, tgt_emb_dim, name='tgt_lookup') # lookup table for target words
    params += src_lookup.params + tgt_lookup.params
    
    # LSTMs
    src_lstm_for = LSTM(src_emb_dim, src_lstm_hid_dim, name='src_lstm_for', with_batch=False)
    src_lstm_rev = LSTM(src_emb_dim, src_lstm_hid_dim, name='src_lstm_rev', with_batch=False)
    tgt_lstm = LSTM(2 * tgt_emb_dim, tgt_lstm_hid_dim, name='tgt_lstm', with_batch=False)
    params += src_lstm_for.params + src_lstm_rev.params + tgt_lstm.params[:-1]
    
    # Projection layers
    proj_layer1 = HiddenLayer(tgt_lstm_hid_dim + 2 * src_lstm_hid_dim, n_tgt, name='proj_layer1', activation='softmax')
    proj_layer2 = HiddenLayer(2 * src_lstm_hid_dim, tgt_emb_dim, name='proj_layer2', activation='tanh')
    params += proj_layer1.params # + proj_layer2.params
    
    
    # Train status
    is_train = T.iscalar('is_train')
    # Input sentence
    src_sentence = T.ivector()
    # Current output translation
    tgt_sentence = T.ivector()
    # Gold translation
    tgt_gold = T.ivector()
    
    src_sentence_emb = src_lookup.link(src_sentence)
    tgt_sentence_emb = tgt_lookup.link(tgt_sentence)
    print 'src_sentence_emb', src_sentence_emb.eval({src_sentence: src_sentence_t}).shape
    print 'tgt_sentence_emb', tgt_sentence_emb.eval({tgt_sentence: tgt_sentence_t}).shape
    
    src_lstm_for.link(src_sentence_emb)
    src_lstm_rev.link(src_sentence_emb[::-1, :])
    
    print 'src_lstm_for.h', src_lstm_for.h.eval({src_sentence: src_sentence_t}).shape
    print 'src_lstm_rev.h', src_lstm_rev.h.eval({src_sentence: src_sentence_t}).shape
    
    src_context = T.concatenate([src_lstm_for.h, src_lstm_rev.h[::-1, :]], axis=1)
    print 'src_context', src_context.eval({src_sentence: src_sentence_t}).shape
    
    tgt_lstm.h_0 = src_context[-1]
    print 'tgt sentence emb', tgt_sentence_emb.eval({src_sentence: src_sentence_t, tgt_sentence: tgt_sentence_t}).shape
    tgt_lstm.link(tgt_sentence_emb)
    print 'tgt_lstm.h', tgt_lstm.h.eval({src_sentence: src_sentence_t, tgt_sentence: tgt_sentence_t}).shape
    
    transition = tgt_lstm.h.dot(src_context.transpose())
    transition = transition.dot(src_context)
    print 'transition', transition.eval({src_sentence: src_sentence_t, tgt_sentence: tgt_sentence_t}).shape
    
    transition_last = T.concatenate([transition, tgt_lstm.h], axis=1)
    print 'transition_last', transition_last.eval({src_sentence: src_sentence_t, tgt_sentence: tgt_sentence_t}).shape
    
    prediction = proj_layer1.link(transition_last)
    print 'prediction', prediction.eval({src_sentence: src_sentence_t, tgt_sentence: tgt_sentence_t}).shape
    
    cost = T.nnet.categorical_crossentropy(prediction, tgt_gold).mean()
    cost += beta * T.mean((tgt_lstm.h[:-1] ** 2 - tgt_lstm.h[1:] ** 2) ** 2) # Regularization of RNNs from http://arxiv.org/pdf/1511.08400v6.pdf
    
    print 'cost', cost.eval({src_sentence: src_sentence_t, tgt_sentence: tgt_sentence_t, tgt_gold: tgt_gold_t})
    
    
    # In[26]:
    
    updates=LearningMethod(clip=5.0).get_updates('adam', cost, params)
    
    
    # In[27]:
    
    f_train = theano.function(
        inputs=[src_sentence, tgt_sentence, tgt_gold],
        outputs=cost,
        updates=updates
    )
    
    
    # In[28]:
    
    f_eval = theano.function(
        inputs=[src_sentence, tgt_sentence],
        outputs=prediction,
    )
    
