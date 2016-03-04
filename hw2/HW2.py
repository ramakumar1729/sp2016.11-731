
# coding: utf-8

# In[81]:

#! /usr/bin/python
import numpy as np
import xgboost as xgb


trainData = '/Users/rpasumar/Desktop/CMU/11-731/sp2016.11-731/hw2/data/all.data.libsvm.train'
bigTrainData = '/Users/rpasumar/Desktop/CMU/11-731/sp2016.11-731/hw2/data/all.data.libsvm'
testData = '/Users/rpasumar/Desktop/CMU/11-731/sp2016.11-731/hw2/data/all.data.libsvm.test'
xg_train = xgb.DMatrix( trainData)
xg_test = xgb.DMatrix(testData)
xg_btrain = xgb.DMatrix(bigTrainData)
param = {}
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 3
param['silent'] = 1
param['nthread'] = 8
param['num_class'] = 3
test_y = list(xg_test.get_label())
watchlist = [ (xg_btrain,'train'), (xg_test, 'test') ]
# no. of trees built
num_round = 1500

bst = xgb.train(param, xg_btrain, num_round, watchlist );
# get prediction
pred = bst.predict( xg_test );

print ('predicting, classification accuracy=%f' % (sum( int(pred[i]) == test_Y[i] for i in range(len(test_Y))) / float(len(test_Y)) ))


# In[82]:

# features.embedding.final 2
# features.f1 1
#features.lm 3
#features.lm2.l 3
#features.parse 2
#features.pos 1
#features.embedding.inv 2
#features.embedding.inv.norm 2


# In[83]:

allData = '/Users/rpasumar/Desktop/CMU/11-731/sp2016.11-731/hw2/data/features.all.libsvm'
xg_alldata = xgb.DMatrix(allData)
pred_all = bst.predict(xg_alldata)
with open('/Users/rpasumar/Desktop/CMU/11-731/sp2016.11-731/hw2/output.xgb.txt','w') as f:
    f.write('\n'.join([str(x) if x != 2 else str(-1) for x in pred_all ]))
          

