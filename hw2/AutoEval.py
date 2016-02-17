#!/usr/bin/env python
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
 
# DRY
def word_matches(h, ref):
    return sum(1 for w in h if w in ref)
    # or sum(w in ref for w in f) # cast bool -> int
    # or sum(map(ref.__contains__, h)) # ugly!

def weighted_F1(h,ref,rho):
    precision = float(sum(1 for w in h if w in ref))/len(h)
    recall = float(sum(1 for w in ref if w in h))/len(ref)
    if precision == 0 or recall == 0:
        return 0
    ret = (precision)**-1 + (rho*recall)**-1
    return ret**-1
 
def main():
    parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
    # PEP8: use ' and not " for strings
    parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
            help='input file (default data/train-test.hyp1-hyp2-ref)')
    parser.add_argument('-n', '--num_sentences', default=None, type=int,
            help='Number of hypothesis pairs to evaluate')
    parser.add_argument('-rho','--weight_F1',default=1,type=float,
            help='weight for P-R tradeoff')
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
        h1_match = weighted_F1(h1,rset,rho)
        h2_match = weighted_F1(h2,rset,rho)
        print(-1 if h1_match > h2_match else # \begin{cases}
                (0 if h1_match == h2_match
                    else 1)) # \end{cases}
 
# convention to allow import of this file as a module
if __name__ == '__main__':
    main()
