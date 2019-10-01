#!/usr/bin/env python
# coding: utf-8

# # Strategies for estimating a model performance
# 
# by Oscar Takeshita
# 
# Initially published: February 23, 2018
# 
# Updated:             March 1, 2018 (Added graphs)
# 
# This study was motivated by this [**discussion**](https://www.kaggle.com/questions-and-answers/48707#post286918). The question was about strategies for estimating
# model performance on unseen data. This may not be a question straightforward to answer because we have
# several variables to think about and assumptions to be made. Although this is not a Titanic kernel, I had to find a home for it. Hopefully you'll find the results useful or at least entertaining.
# 
# One possible strategy is to leave a portion of the train data out that is not touched during training.
# This is typically used in one of the last steps to check how well our model does with guaranteed unseen data. The remaining data is used to tune the model using K-Fold cross validation (CV). K-Fold CV also outputs information about how well it may do with unseen data. However, a valid concern is that in K-Fold CV we no longer have guaranteed unseed data by design. The question is how CV and hold-out compare with each other. 
# 
# Statistically, since the hold-out set is typically smaller than the entire CV set, it should suffer from more variance. I thought then about a [**theorem**](https://www.kaggle.com/questions-and-answers/48707#286918) in the discussion. It states that as long as the partitions are statistically very similar (more so in an information-theoretical sense), the estimation for unseen data using a hold-out set should give the same average estimate as in the cross validation. If that's the case, CV only would then suffice. The approach might be objectionable if we absolutely don't want to use data that has been used during training to forcast performance of unseen data. It is really a design choice. Hopefully we may see pros and cons. This topic is also discussed by [DanB](https://www.kaggle.com/dansbecker) in this tutorial [**notebook**](https://www.kaggle.com/dansbecker/cross-validation/notebook). We provide here a controlled experimental platform. 
# 
# Since mathematical proofs are quite abstract, this notebook tests the theorem in practice with synthesized data. This way we can precisely control the data and understand the laws and mechanisms behind. 
# 
# The synthesized data has four independent features A, B, C and D. Each may take the discrete values {0,1,2,3}. A class is then associated by the following non-linear formula CLASS = A + B*C + A*D. We then generate a data frame with this rule.
# Next, 20% of the rows have features randomly scrambled. With this, if a classifier is trained on this data, we expect 80% of classification precision. Let's then examine what happens...
# 
# 1. [Creating independent features A, B, C and D](#features)
# 
# 2. [Creating data frame](#dataframe)
# 3. [Adding classification errors](#add_errors)
# 4. [Defining partitions](#def_partitions)
# 5. [Training a classifier](#train)
# 6. [Results](#results)
# 7. [Conclusions](#conclusions)
# 8. [Appendix](#appendix)

# In[ ]:


import pandas as pd
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt


# ## Creating independent features A, B, C and D <a class="anchor" id="features"></a>
# 
# Each feature may assume a values in {0,1,2,3}. They are subsequently shuffled to make them independent.

# In[ ]:


def gen_features():
    a_levels  = 4     # number of levels per feature
    a_len     = 16000 # number of instances per feature
    A = [ i % a_levels for i in range(a_len)]
    B = [ i % a_levels for i in range(a_len)]
    C = [ i % a_levels for i in range(a_len)]
    D = [ i % a_levels for i in range(a_len)]
    shuffle(A)
    shuffle(B)
    shuffle(C)
    shuffle(D)
    return A, B, C, D


# ## Creating data frame <a class="anchor" id="dataframe"></a>
# 
# We define the CLASS by the non-linear relation CLASS = A + B*C + A*D. Let's make the classifier job not too easy by avoiding a simple linear relation!

# In[ ]:


def gen_df():
    A, B, C, D = gen_features()
    df = pd.DataFrame({'A':A,'B':B,'C':C,'D':D})
    df['CLASS'] = df['A'] + df['B']*df['C'] + df['A']*df['D']
    return df

df = gen_df()
df.head(10)


# ### Inspecting the first 100 rows of feature A

# In[ ]:


df['A'][0:100].plot(title='A', style='.')


# ## Adding classification errors  <a class="anchor" id="add_errors"></a>
# 
# We shuffle 20% of the rows. This should create about 20% of classification errors with respect to the "ground truth" formula. We then add a GOOD indicator to mark rows that still follows our non-linear formula.

# In[ ]:


def add_errors(df):
    # shuffle the first shuffle_feature rows for each feature 20%
    shuffle_features = int(df.shape[0] * 0.20)
    shuffle(df['A'][0:shuffle_features])
    shuffle(df['B'][0:shuffle_features])
    shuffle(df['C'][0:shuffle_features])
    shuffle(df['D'][0:shuffle_features])
    return df

df = add_errors(df)

def add_marker(df):
    # create a marker for tracking rows that no longer follows the non-linear relation
    df['GOOD'] = (df['A'] + df['B']*df['C'] + df['A']*df['D']) == df['CLASS'] 
    df['GOOD'].replace([True,False],[1,0],inplace = True)
    return df

df = add_marker(df)
df.head(10)    


# ### Shuffle entire data frame rows
# 
# We shuffle again the entire data frame to spread the concentration of classification errors at the top of the data frame.

# In[ ]:


df = df.sample(frac=1).reset_index(drop=True)
df['CLASS'][0:200].plot(title='CLASS',style='.')


# Although we shuffle 20% of the rows, some may still end up following the formula. We compute the percentage that still follows the formula.

# In[ ]:


print('Whole set true mean classification accuracy {0:2.4f}'.format(df['GOOD'].mean()))


# ## Defining partitions <a class="anchor" id="def_partitions"></a>
# 
# We are now ready to set partitions up. We first reserve 80% of the data to emulate unseen data.  It is a large enough portion to be of statistical significance. The remaining 20% (df2) is what is left for the "data scientist" to work with. Next we reserve 10% of df2 for hold-out. We may at this point calculate the expected accuracy in the unseen data and the hold-out set. Recall that we expect the accuracy to be 80% by design. Differences are caused by the hold-out sampling. With the default settings, you'll see hold out has great variance because of its small size (please run this kernel multiple times to verify).

# In[ ]:


from sklearn.model_selection import train_test_split

# reserving emulated unseen data. df2 is what's visible for the "data scientist"
df2, unseen = train_test_split(df, test_size=0.8)

# let's leave 10% data out for hold-out
train, hold_out = train_test_split(df2,test_size=0.10)
print(train.shape)
print(hold_out.shape)
print('hold out true mean classification accuracy {0:2.4f}'.format(hold_out['GOOD'].mean()))
print('unseen   true mean classification accuracy {0:2.4f}'.format(unseen['GOOD'].mean()))


# We now compute the expected accuracy for the remaining of the train set. This will be typically closer to the whole set true mean accuracy than the hold-out estimate by the law-of-large-numbers. A random seed is not set in this script so you may verify the outcome for multiple runs and verify the claims in this notebook. A single run won't cut it!

# In[ ]:


print('train (cv) true mean classification accuracy {0:2.4f}'.format(train['GOOD'].mean()))


# ## Training a classifier <a class="anchor" id="train"></a>
# 
# We will now train a KNeighborsClassifier. The tunable parameters are n_neighbors and the minkowski metric parameter p. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
knclass = KNeighborsClassifier(n_neighbors=11, metric = 'minkowski')
kn_param_grid={'n_neighbors':[3,5,7,9,11], 'p':[1,1.5,2]}
gs = GridSearchCV(knclass, kn_param_grid, cv = 5, return_train_score = True, n_jobs=4)


# In[ ]:


gs.fit(X=np.array(train[['A','B','C','D']]), y = np.array(train['CLASS']))


# ### Let's print the best tuned parameters

# In[ ]:


gs.best_params_


# ## Unseen data
# 
# Let's see first how our model does with our simulated unseen data

# In[ ]:


unseen_data_score = gs.score(unseen[['A','B','C','D']],unseen['CLASS'])
print('accuracy of the model on unseen data {0:2.4f}'.format(unseen_data_score))


# ### Let's also print the train score
# 
# These are usually expected to have higher precision than the test score (computed next) because the train portion is typically overfit. I encourage the understanding of the contents of "cv_results_" and avoiding using GridSearchCV as a black-box. It contains many useful statistics.

# In[ ]:


gs.cv_results_['mean_train_score']


# ### Let's now print the test score
# 
# This is the indicator for the performance of the model for unseen data using the cross validation technique. The bigger the gap with the train score the more overfit your model is.

# In[ ]:


# cross validation score
cv_best_score = gs.best_score_
gs.cv_results_['mean_test_score']


# # Results <a class="anchor" id="results"></a>
# 
# ## hold-out score
# 
# We compare the model score on the hold-out set and on unseen data.

# In[ ]:


# hold out score
hold_out_score = gs.score(hold_out[['A','B','C','D']], hold_out['CLASS'])


# In[ ]:


print('Accuracy by model on hold-out {0:2.4f}; accuracy by model on unseen {1:2.4f}'.format(hold_out_score, unseen_data_score))


# ## CV score
# 
# We compare the model score on CV data and on unseen data. It is good with the added advantage that is statistically more sound since it is based on a larger portion of the data. Despite potential concerns that using all data might be bad for estimating performance on unseen data.

# In[ ]:


print('Accuracy by model on CV       {0:2.4f}; accuracy by model on unseen {1:2.4}'.format(cv_best_score, unseen_data_score))


# Let's now repeat the above process a number of times to obtain more statistically significant results.

# In[ ]:


def one_run():
    # create base data frame
    df = gen_df()
    df = add_errors(df)
    df = add_marker(df)
    df = df.sample(frac=1).reset_index(drop=True)
    
    # make partitions
    df2,   unseen   = train_test_split(df, test_size= 0.8)
    train, hold_out = train_test_split(df2,test_size= 0.1)
    
    # grid search and fit
    gs.fit(X=np.array(train[['A','B','C','D']]), y = np.array(train['CLASS']))
    
    # compute hold out score
    hold_out_score = gs.score(hold_out[['A','B','C','D']], hold_out['CLASS'])
    
    # compute cv score
    cv_best_score = gs.best_score_
    
    # compute unseen score
    unseen_score = gs.score(unseen[['A','B','C','D']], unseen['CLASS'])
    
    return unseen_score, cv_best_score, hold_out_score

n_loops = 30
result = np.empty([n_loops,3],dtype=float)

for i in range(n_loops):
    result[i,:] = np.array(one_run())
    print('round {3:2}   unseen {0:2.4f}  CV {1:2.3f} hold-out {2:2.3f}'.format(result[i,0],result[i,1],result[i,2],i))
    


# In[ ]:


plt.plot(result[:,0]-result[:,1],label='unseen-cv')
plt.plot(result[:,0]-result[:,2],label='unseen-hold-out')
plt.xlabel('round')
plt.ylabel('delta')
plt.legend()


# In[ ]:


print('cv                      mean error:', np.mean(result[:,0]-result[:,1]))
print('cv        error standard deviation:', np.std(result[:,0]-result[:,1]))
print('hold out                mean error:', np.mean(result[:,0]-result[:,2]))
print('hold out  error standard deviation:', np.std(result[:,0]-result[:,2]))


# We verify that using K-fold cv yields a better estimate than using a hold-out for this controlled data set. 

# # Conclusions <a class="anchor" id="conclusions"></a>
# 
# We demonstrated the implications of the "partition theorem". It says K-Fold CV is sufficient for the estimation of unseen data without a need for a separate hold-out set when the partitions are information-theoretically similar. Not only that, but CV estimate might be more accurate on average as a consequence of the law of large numbers. The similarity requirement might be a big "if" for small sets, however, it demonstrates assumptions and recommendations must be carefully examined and understood. Please don't just accept them as irrefutable! I'm also eager to hear about counter examples. Lastly, I believe a number of other interesting conclusions can be drawn from the theorem.
# 
# Thank you for reading and please let me know about any comments!

# # Appendix <a class="anchor" id="appendix"></a>
# 
# We will add some related materials found on the web. This is not to endorse any particular source.
# 
# [Coursera](https://www.coursera.org/learn/deep-neural-network/lecture/cxG1s/train-dev-test-sets) video by Prof. Andrew Ng.
# 
# [CMU](https://www.cs.cmu.edu/~schneide/tut5/node42.html) by Prof. Jeff Schneider
# 
# [stackoverflow](https://stackoverflow.com/questions/34549396/holdout-vs-k-fold-cross-validation-in-libsvm), which references [The Elements of 
# Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) by Trevor Hastie, Robert Tibshirani, Jerome Friedman
# 
# [stackexchange](https://stats.stackexchange.com/questions/104713/hold-out-validation-vs-cross-validation)
# 
# [Illustrations and diagrams](https://www.kdnuggets.com/2017/08/dataiku-predictive-model-holdout-cross-validation.html)
# 
# [A 2016 IEEE paper](Analysis of k-Fold Cross-Validation over Hold-Out Validation on Colossal Datasets for Quality Classification)
# 
