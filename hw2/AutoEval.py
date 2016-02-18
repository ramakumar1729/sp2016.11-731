#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import math

from nltk.util import ngrams
from collections import Counter
 
# DRY
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)
    # or sum(w in ref for w in f) # cast bool -> int
    # or sum(map(ref.__contains__, h)) # ugly!

def modified_precision(h,ref,n):
    ng_counts_h = Counter(ngrams(h,n))
    ng_counts_ref = Counter(ngrams(ref,n))
    modified_counts = Counter()   

    
    if not ng_counts_h:
        return 0
    for ng in ng_counts_h.keys():
        modified_counts[ng] = max(modified_counts[ng], ng_counts_ref[ng])
    truncated_cts = Counter((ng, min(ng_counts_h[ng],modified_counts[ng])) for ng in ng_counts_h)
    return sum(truncated_cts.values())/float(sum(ng_counts_h.values()))

def weighted_F1(h,ref,rho):
    #h = list(ngrams(h_i,2)) + list(h_i)
    #ref = list(ngrams(ref_i,2)) + list(ref_i)
    
    precision = float(sum(1 for w in h if w in ref))/len(h)
    precision = 0.5*modified_precision(h,ref,2) + 0.5*precision 
    
    recall = float(sum(1 for w in ref if w in h))/len(ref)
    if precision == 0 or recall == 0:
        return 0
    ret = (1-rho) * (precision)**-1 + rho * (recall)**-1
    return  ret**-1

def BLEU(h,ref,n):
    def brevity_penalty(h,ref):
        c = len(h)
        r = len(ref)
        if c >= r:
            return 1
        else:
            return math.exp(1 - r/c)
        
    weights = [1.0/n for x in xrange(n)]
    mod_p = 0
    for i,w in enumerate(weights):
        x = modified_precision(h,ref,i)
        if x == 0:
            return 0
        mod_p += w*math.log(x)
      
    return brevity_penalty(h,ref) * math.exp(mod_p)
    
    
 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-rho','--weight_F1',default=1,type=float,
            help='weight for P-R tradeoff')
    parser.add_argument('-e','--epsilon',default=0,type=float,
            help='epsilon for 0')
    # note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
    opts = parser.parse_args()
 
    # we create a generator and avoid loading all sentences into a list
    def sentences():
        with open(opts.input) as f:
            for pair in f:
                yield [sentence.strip().split() for sentence in pair.split(' ||| ')]
 
    # note: the -n option does not work in the original code
    for h1, h2, ref in islice(sentences(), opts.num_sentences):
        rset = set(ref)
        #h1_match = word_matches(h1, rset)
        #h2_match = word_matches(h2, rset)
        rho = opts.weight_F1
        h1_match = weighted_F1(h1,rset,rho) #+ BLEU(h1,ref,3)
        h2_match = weighted_F1(h2,rset,rho) #+ BLEU(h2,ref,3)
        #h1_match = BLEU(h1,ref,2)
        #h2_match = BLEU(h2,ref,2)
        diff = h1_match - h2_match
        eps = opts.epsilon
        
        if diff > eps:
            print -1
        elif diff < -eps:
            print 1
        else:
            print 0
        
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
