author: rpasumar

There are three Python programs here (`-h` for usage):

 - `./align` aligns words using Dice's coefficient.
 - `./check` checks for out-of-bounds alignment points.
 - `./grade` computes alignment error rate.

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./align -t 0.9 -n 1000 | ./check | ./grade -n 5


The `data/` directory contains a fragment of the German/English Europarl corpus.

 - `data/dev-test-train.de-en` is the German/English parallel data to be aligned. The first 150 sentences are for development; the next 150 is a blind set you will be evaluated on; and the remainder of the file is unannotated parallel data.

 - `data/dev.align` contains 150 manual alignments corresponding to the first 150 sentences of the parallel corpus. When you run `./check` these are used to compute the alignment error rate. You may use these in any way you choose. The notation `i-j` means the word at position *i* (0-indexed) in the German sentence is aligned to the word at position *j* in the English sentence; the notation `i?j` means they are "probably" aligned.


IBM Model 1 has been implemented in this homework.

IBM Model 1 was run using EM for various iterations, and for several decoding logic schemes.

decoding schemes:

    1. a_i = argmax_j p(e_i|g_j) -- most probable alignment based on translation probabilities computed by EM.

    2. a_i = argmax_j 1/(1+penalty) * p(e_i|g_j) -- penalty = (i-j)**2

    3. a_i = argmax_j 1/(1+penalty) * p(e_i|g_j) -- penalty == |i/m - j/n|


