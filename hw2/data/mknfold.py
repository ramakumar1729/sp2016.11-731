#!/usr/bin/python
import sys
import random

import time

if len(sys.argv) < 2:
    print ('Usage:<filename> <k> [nfold = 5]')
    exit(0)

#random.seed( time.time() )

k = float( sys.argv[2] )
if len(sys.argv) > 3:
    nfold = int( sys.argv[3] )
else:
    nfold = 5

fi = open( sys.argv[1], 'r' )
ftr = open( sys.argv[1]+'.train', 'w' )
fte = open( sys.argv[1]+'.test', 'w' )
for l in fi:
    if random.random() > k:
        fte.write( l )
    else:
        ftr.write( l )

fi.close()
ftr.close()
fte.close()

