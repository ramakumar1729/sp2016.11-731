There are three Python programs here (`-h` for usage):

 - `./evaluate` evaluates pairs of MT output hypotheses relative to a reference translation using counts of matched words
 - `./check` checks that the output file is correctly formatted
 - `./grade` computes the accuracy

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./evaluate | ./check | ./grade


The `data/` directory contains the following two files:

 - `data/train-test.hyp1-hyp2-ref` is a file containing tuples of two translation hypotheses and a human (gold standard) translation. The first 26208 tuples are training data. The remaining 24131 tuples are test data.

 - `data/train.gold` contains gold standard human judgements indicating whether the first hypothesis (hyp1) or the second hypothesis (hyp2) is better or equally good/bad for training data.

Until the deadline the scores shown on the leaderboard will be accuracy on the training set. After the deadline, scores on the blind test set will be revealed and used for final grading of the assignment.

Our Approach

- To begin with, we tried using an LSTM to learn a representation for both hypothesis translations and the reference translation and make a prediction -1, 0, 1 by concatenating them and projecting them onto a softmax layer of size 3. I really wish this had worked but it didn't :-(

- We then resorted to feature engineering to build a classifier. We used the following features.

     * Pre-trained word embeddings using Wang2vec structured skipngram (https://github.com/wlin12/wang2vec). We used pairwise cosine similarity of word embeddings to compute a n x m matrix where n is the number of words in one of the hypotheses and m is the number of words in the reference translation. We enforced a non-diagonal penalty |exp(-alpha) * i/n - j/m|. We then used scipy's linear sum assignment (http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) to compute the sum of the scores of the "best" assignment of each word in the hypothesis. NOTE: Instead of tuning alpha, we lazily used a few alphas and threw them all in as features. 
     * Word-level F1-scores: We extended the baseline implementation to come up with an F1 score using Precision (Number of words in hypothesis that are in reference) and Recall (Number of words in reference that are in the hypothesis). We also normalized by the number of the words in the sentence. Additionally, we also used bi-gram precisions as well.
     * POS Tag F1-scores: We computed features for POS tags in the EXACT SAME manner as words described above.
     * Parse Tree kernels: We used Stanford CoreNLP's constituency parser to get parse trees for every sentence and then computed a tree kernel which we defined as the union of the number of identical subtrees in the hypothesis and reference translations. (We gave a higher score to larger subtrees that were identical). I don't think this feature helped a lot though.
     * N-gram language model: We used KenLM to train a language model on the PennTreebank and Broadcast News and used the scores of the both the hypothesis translations as features.

- We also tried, and in some cases used scores from 1-line BLEU variant, modified precision (as in BLEU), and changing precision and recall to incorporate number of times a word is seen in both hypothesis and reference.
  We extended precision and recall to N-gram settings (1,2,3).

- Multi-class classifier using Gradient Boosted Decision Trees:
     * We used XGBoost (https://github.com/dmlc/xgboost) an open source GBDT implementation, available as a python module. Gradient Boosting is a low-variance, low-bias technique that perform well in general, and  was used in winning solutions of several Kaggle competitions.
     * Parameters were tuned for optimal performance on a held-out validation set. Depth of trees, number of trees and learning rate are the parameters that we tuned, keeping in mind the trade-offs.
     * We checked whether the operating point of the parameters was good, by checking for various randomly generated train/validation set split up.
     * We then generated the model on all of training data set at this operating point for the parameter. We used this model to generate scores for all of training data.
