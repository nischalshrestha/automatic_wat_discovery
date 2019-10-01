#!/usr/bin/env python
# coding: utf-8

# The main purpose of this notebook is to share a trick to handle ordered categorical features, so if you want to learn more about EDA, model selection, tuning hyperparameters, and so on, I recommend you to this [great kernel](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard).
# 
# Now, let's start my main topic.
# 
# Given a dataset, categorical features often appears. Some of them might be ordered. For instance,  **Grade** is a feature contain 4 levels: "Ex", "Gd", "Ag", "Po", meaning "excellent", "good", "average", "poor" respectively. To convert it into numerical type using dummy (or one-hot-encoding) is not a advisable choice. Please think in this way: Let D(x, y) be the distance between x and y, if D("Excellent", "Good")=1, D("Good", "Average")=1, D("Excellent", "Average")=2. If we use dummy, the distance between each level are all equal to 1, which is intuitively wrong, especially when we want to build a regression model such as linear regression. 
# 
# Thus we should convert such features into numerical values which can indicate the high or low of the level. To achieve this purpose, in the past, I chose map in pandas .
# 

# In[1]:


import pandas as pd
S = pd.DataFrame({'Grade':['Ex','Gd','Ag','Po']})
print('The original feature Grade:\n',S)
S['New_Grade'] = S['Grade'] .map({'Ex':4,'Gd':3,'Ag':2,'Po':1})
print('The feature Grade after transformation:\n',S)


# If you have experience in great machine learning package sklearn, you may say,"Oh, it's too troublesome! Why don't you just use **LabelEncoder**?" Well, it's a really good question, let us do an experiment here together, and I believe that you will find out the answer.

# In[2]:


from sklearn.preprocessing import LabelEncoder
lab = LabelEncoder()
S['Grade_by_lab'] = lab.fit_transform(S['Grade'])
print('The feature Grade after transformation:\n',S)


# Now you must know the answer, that is, LabelEncoder can only conver string to int and sort it by alphabet order, and we can't customize the order according to the [official documentation](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html#sklearn.preprocessing.LabelEncoder).
# 
# Associated with the ordered factor in experiment analysing, I decided to use factorize in pandas to achieve our final purpose. Unfortunately, I failed:

# In[4]:


S['Grade_factorized'] = S['Grade'].factorize(sort=True)[0]
print(S[['Grade','Grade_factorized']])


# It seems factorize function give the same result with that given by LabelEncoder. But don't be frustrated, I add a little trick in this process and finally it succeed! You may concern why there is an indicator **[0]** following **factorize(sort=True)**, don't worry, I will explain it later.
# 

# In[5]:


S['Grade_ordered'] = S['Grade'].astype('category', ordered=True, categories=['Po', 'Ag','Gd','Ex'])
print(S['Grade_ordered'])
print('-----------------------------')
print(S['Grade_ordered'].factorize(sort=True))
S['Grade_ordered_factorize'] = S['Grade_ordered'].factorize(sort=True)[0]
print('-----------------------------')
print(S[['Grade','Grade_ordered_factorize']])


# Look, we succeed to our initial purpose, don't we? Now let me tell you the **[0]** part we skipped just now: Look at the second part of output, you will find that the return of **factorize** is a two elements tuple, and the first element of it is the result we need, while the second element of it is index.
# 
# Finally I want to point out is that **sort=True** guarentees that the order of the final result is what we customized in **astype** part rather than the original order decided by the column of feature.
# 
# If you regard this kernel helps you or is useful in any way, please **vote UP**, thank you very much!
