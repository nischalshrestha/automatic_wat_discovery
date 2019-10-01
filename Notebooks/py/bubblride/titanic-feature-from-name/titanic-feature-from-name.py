#!/usr/bin/env python
# coding: utf-8

# ## Summary
# * `mr`, `mrs` and `ms` are the only phonetics from the `Name` columns that have matches with survival
# * this are somewhat redundant information as these overlap with `Sex` (male, female), SibSp (e.g. a Miss)

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv("../input/train.csv")


# ## Convert Name to DoubleMetaphone tags
# all words a replace by phonetic tags with the [Double Metaphone algorithm](https://en.wikipedia.org/wiki/Metaphone), see [pypi](https://pypi.org/project/Metaphone/)

# In[ ]:


def word_to_phonetic(arr):
    from metaphone import doublemetaphone as dmp
    import re
    out = list()
    for e in arr:
        p = list()
        for s in re.findall(r"[\w']+", e): #for s in e.split():
            p += dmp(s)          #add a 2x tuple
        p = [s for s in p if s]  #rm empty elements
        out.append(' '.join(p))
    return out


# In[ ]:


tags = word_to_phonetic(df['Name'])
tags[:7]


# ## One Feature for each word
# 

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer().fit(tags)
nam = np.array(vec.get_feature_names())
cnt = vec.transform(tags) > 0  #ignore double counts
cnt = cnt.toarray()
#cnt


# In[ ]:


pct = .035
idx = cnt.sum(axis=0) > (cnt.shape[0] * pct)


# display some example

# In[ ]:


tmp = pd.DataFrame(index=nam[idx], columns=['cnt'] )
tmp['cnt'] = cnt.sum(axis=0)[idx]
tmp = tmp.sort_values(by='cnt', ascending=0)
tmp.tail()


# In[ ]:


tmp.head()


# ## Do these variables match with survival?
# The baseline is that everybody died

# In[ ]:


from sklearn.metrics import accuracy_score
baseline = accuracy_score(df['Survived'], np.zeros((len(df['Survived']),)))
baseline


# In[ ]:


scores = list()
for i in np.where(idx)[0]:
    scores.append((i, accuracy_score(df['Survived'].values, cnt[:,i])  ))


# In[ ]:


scores


# We can already eyeball some very promising phonetics tags.
# And these are gender approximinations ...

# In[ ]:


print( nam[722], nam[743], nam[751])


# ## Wrap it into a function

# In[ ]:


def find_relevant_tags(target, arr, dev=0.2):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics import accuracy_score

    # count words
    vec = CountVectorizer().fit(arr)
    nam = np.array(vec.get_feature_names())
    cnt = vec.transform(arr) > 0  #ignore double counts
    cnt = cnt.toarray()
    
    # establish baseline
    baseline = accuracy_score(target, np.zeros((len(target),)))
    #ranges
    u = baseline + (1 - baseline) * dev
    d = baseline * (1 - dev)
    
    
    # compute scores
    scores = list()
    for i in range(cnt.shape[1]):
        s = accuracy_score(target, cnt[:,i])
        f = True if s>u else (True if s<d else False) 
        scores.append((i, s, f))
    
    # filter relevant scores
    relevant = [r for r in scores if r[2]]
    
    # done
    return relevant, scores, (baseline, u, d)


# In[ ]:


arr = df['Name'].values
target = df['Survived'].values

# convert to phonetic
tags = word_to_phonetic(arr)

# score
relevant, scores, opt = find_relevant_tags(target, arr, dev=0.05)


# In[ ]:


relevant

