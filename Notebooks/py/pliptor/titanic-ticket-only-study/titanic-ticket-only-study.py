#!/usr/bin/env python
# coding: utf-8

# # What score can we expect using just the Ticket feature?
# 
# This notebook investigates working with the Titanic set using only the **Ticket** feature for prediction.
# 
# ![Ticket](https://upload.wikimedia.org/wikipedia/commons/a/ad/Ticket_for_the_Titanic%2C_1912.jpg)
# 
# The above is a public domain image from [**this link**](https://commons.wikimedia.org/wiki/File:Ticket_for_the_Titanic,_1912.jpg)
# 
# This is motivated by an attempt to understand how each feature alone may contribute to the
# prediction of survival. Doing so for simple features such as gender (public score **0.76555**) or passenger class (public score **0.65550**) poses no problem and we know those scores are the optimum solutions under those restrictions. Other examples can be found [**here**](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score). Single feature prediction is not so straightforward for features such as **Name, Ticket** or **Cabin** because or their more complex structure. 
# 
# The notebook is also motivated by this [**Dec 2014 discussion**](https://www.kaggle.com/c/titanic/discussion/11127). It is therein discussed that close ticket numbers (rather than just identical) are often tied to people that traveled together. 
# 
# We will parse and dissect the Ticket value into 400+ features that intends to capture the relation between its owners. Finally, we will apply the **KNeighbors** algorithm solve it. We will optimize the KNeighbors parameter with GridSearch and also compute the cross validation to estimate the expected public score. 
# 
# I recently found a kernel or discussion that mentions that the ticket number is correlated to Pclass (if you happen to know the the kernel/discussion, I'd like to include proper credit here). We will verify that fact in this kernel. Consequently we expect prediction using the Ticket-only information to be no worse than the [**optimal solution for Pclass-only solution**](https://www.kaggle.com/pliptor/optimal-titanic-for-pclass-only-0-65550)   (**public score 0.65550**). Are we going to succeed?
# 
# ## Contents
# 
# 1. [Reading data](#read_data)
# 2. [Parsing the Ticket feature](#parsing)
# 3. [Modeling](#modeling)
# 4. [Predicting and creating a submission file](#submission)
# 
# [Conclusions](#conclusions)
# 
# 

# # 1) Reading data <a class="anchor" id="read_data"></a>
# 
# We will only read the relevant columns of the data. It will keep the data clean and prevent any accidental leakage of other features into our setup. Pclass will not be used for predictions but we will show how the Ticket feature is strongly correlated with Pclass. We will drop the Pclass feature before doing the modeling and predictions.

# In[1]:


import pandas as pd
import numpy  as np
from matplotlib import pyplot as plt

np.random.seed(2018)

# load data sets 
predictors = ['Ticket','Pclass']
train = pd.read_csv('../input/train.csv', usecols =['Survived','PassengerId'] + predictors)
test  = pd.read_csv('../input/test.csv' , usecols =['PassengerId'] + predictors)

# combine train and test for joint processing 
test['Survived'] = np.nan
comb = pd.concat([ train, test ])
comb.head()


# A close inspection reveals that the Ticket feature can be split between those that are purely numeric and those that have an alpha numeric prefix. There is one exception to this rule in which crew members are issued the ticket 'LINE' and have only an alpha numeric component.  

# In[2]:


comb.loc[comb['Ticket']=='LINE']


# In order to make them consistent with other tickets, we replace the LINE ticket with "LINE 0".

# In[3]:


comb['Ticket'] = comb['Ticket'].replace('LINE','LINE 0')


# It is well known there are duplicate ticket values in the set. Let's create a feature that counts the number of duplications.

# In[4]:


dup_tickets = comb.groupby('Ticket').size()
comb['DupTickets'] = comb['Ticket'].map(dup_tickets)
plt.xlabel('duplications')
plt.ylabel('frequency')
plt.title('Duplicate Tickets')
comb['DupTickets'].hist(bins=20)


# We see that unique tickets are the overwhelming majority. 

# # 2) Parsing the Ticket feature <a class="anchor" id="parsing"></a>
# 
# 
# We now parse and generate extra features from the Ticket feature. We first remove punctuations and then extract the alphanumeric prefix as a new feature 'Prefix' when it exists.

# In[5]:


# remove dots and slashes
comb['Ticket'] = comb['Ticket'].apply(lambda x: x.replace('.','').replace('/','').lower())
def get_prefix(ticket):
    lead = ticket.split(' ')[0][0]
    if lead.isalpha():
        return ticket.split(' ')[0]
    else:
        return 'NoPrefix'
comb['Prefix'] = comb['Ticket'].apply(lambda x: get_prefix(x))


# Next we extract the numeric portion of the ticket. We create extra features such as:
# 
# 1. The numerical component (TNumeric)
# 2. Number of digits (TNlen)
# 3. Leading digit (LeadingDigit)
# 4. Group similar tickets by discarding the last digits (TGroup)
# 
# TGroup is a feature that I thought it would help to capture what has been discussed in [**here**](https://www.kaggle.com/c/titanic/discussion/11127). The idea is similar ticket numbers (not just identical numbers) is tied to groups of people. This is important because we know that identifying [**groups is helpful**](https://www.kaggle.com/pliptor/divide-and-conquer-0-82296) and looking at just family ties is not sufficient for the best model. 

# In[6]:


comb['TNumeric'] = comb['Ticket'].apply(lambda x: int(x.split(' ')[-1])//1)
comb['TNlen'] = comb['TNumeric'].apply(lambda x : len(str(x)))
comb['LeadingDigit'] = comb['TNumeric'].apply(lambda x : int(str(x)[0]))
comb['TGroup'] = comb['Ticket'].apply(lambda x: str(int(x.split(' ')[-1])//10))
comb.head()


# ## 2.1) Pclass is highly correlated to the first digit of the numeric component
# 
# Here we show that the leading digit tells much about Pclass. I read about this fact in some kernel/discussion.
# If you happen to know the original source, I'd like to add credit here. Note that because of this high correlation, we expect a Ticket-only solution to be no worse than the optimal Pclass-only solution. We will see that it is indeed is significantly better.

# In[7]:


pd.crosstab(comb['Pclass'],comb['LeadingDigit'])


# We now drop the original Ticket and TNumeric features and keep only the generated features. We also drop the Pclass feature as we are done showing they are highly correlated.

# In[8]:


comb = comb.drop(columns=['Ticket','TNumeric','Pclass'])


# Now we one-hot-encode the Prefix and TGroup features.

# In[9]:


comb = pd.concat([pd.get_dummies(comb[['Prefix','TGroup']]), comb[['PassengerId','Survived','DupTickets','TNlen','LeadingDigit']]],axis=1)


# Let's check our data frame shape. It has 449 features and one target (total of 450 columns).

# In[10]:


comb.shape


# In[11]:


predictors = sorted(list(set(comb.columns) - set(['PassengerId','Survived'])))


# ## 2.2) Build df_train and df_test data frames

# In[12]:


# comb2 now becomes the combined data in numeric with the PassengerId feature removed
comb2 = comb[predictors + ['Survived']]
comb2.head()


# Now we split back comb2 as we are done pre-processing

# In[13]:


df_train = comb2.loc[comb2['Survived'].isin([np.nan]) == False]
df_test  = comb2.loc[comb2['Survived'].isin([np.nan]) == True]

print(df_train.shape)
df_train.head()


# In[14]:


print(df_test.shape)
df_test.head()


# Note how we have now a very detailed list (over 400 features!) of possible connection between the passengers. We are now ready for modeling!

# # 3) Modeling <a class="anchor" id="modeling"></a>
# 
# We will use a KNeighborsClassifier for the model and use GridSearchCV to tune it.

# In[15]:


from sklearn.model_selection import GridSearchCV


# In[16]:


from sklearn.neighbors import KNeighborsClassifier
knclass = KNeighborsClassifier(n_neighbors=11, metric = 'manhattan')
param_grid = ({'n_neighbors':[6,7,8,9,11],'metric':['manhattan','minkowski'],'p':[1,2]}) 
grs = GridSearchCV(knclass, param_grid, cv = 28, n_jobs=1, return_train_score = True, iid = False, pre_dispatch=1)
grs.fit(np.array(df_train[predictors]), np.array(df_train['Survived']))


# Now that the tuning is completed, we print the best parameter found and also the estimated accuracy for the unseen data.  

# In[17]:


print("Best parameters " + str(grs.best_params_))
gpd = pd.DataFrame(grs.cv_results_)
print("Estimated accuracy of this model for unseen data:{0:1.4f}".format(gpd['mean_test_score'][grs.best_index_]))


# # 4) Predicting and creating a submission file<a class="anchor" id="submission"></a>

# In[18]:


pred_knn = grs.predict(np.array(df_test[predictors]))

sub = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':pred_knn})
sub.to_csv('ticket_only_knn.csv', index = False, float_format='%1d')
sub.head()


# # Conclusions <a class="anchor" id="conclusions"></a>
# 
# We tackled the Titanic problem using only Ticket feature processing. The Ticket feature was transformed to a 449-feature data frame. The features are supposed to capture fine details on how passengers were connected by just the ticket values! The prediction was then made with a KNneighbor optimizer tuned with a cross-validated grid search. As a by-product, we found the estimated accuracy for unseen data to be **0.7168**. The obtained public score is **0.7129**, which indicates the model is fairly well tuned to unseen data. The score is also significantly better than the optimal Pclass-only solution (public score **0.65550**)
# 
# Future work: Unfortunately we can't guarantee this solution for Ticket-only is optimal. It might be still possible to improve it both from the perspective of feature generation and from KNneighbor optimization.
# I hope to do it at some time. 
# 
# Please let me know if you have questions or comments!
# 
