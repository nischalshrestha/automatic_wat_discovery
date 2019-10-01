#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


import pandas as pd
df = pd.read_csv('../input/train.csv')


# Lucky ticket (in Russia) is a ticket with even number of digits, where  summ of the first half of the digits equald to the second part.
# Let's find out whether it helped Titanic passengers.
# Here is a function for checking if the ticket is lucky:

# In[ ]:


import re

def is_lucky(df):
    ticket_no_pattern = re.compile(r'\d\d+')
    ticket_no = ticket_no_pattern.findall(str(df))
    if ticket_no:
        if len(ticket_no[0]) % 2 == 0:
            firstpart, secondpart = ticket_no[0][:len(ticket_no[0])//2],                                    ticket_no[0][len(ticket_no[0])//2:]
            return int(sum([int(x) for x in firstpart]) == sum([int(x) for x in secondpart]))
        else:
            return 0
    else:
        return 0


# Now let's add a column to dataframe, with True if this passemger's ticket is lucky or not

# In[ ]:


df['lucky'] = pd.Series(map(is_lucky, df.Ticket.values))


# Now how many survivors had lucky tickets, how nmany survivors are there and how many lucky tickets were sold:

# In[ ]:


surv_lucky = len(df[(df.lucky == 1)&(df.Survived == 1)])
surv_all = len(df[(df.Survived == 1)])
lucky_all = len(df[(df.lucky == 1)])
print('Survivors with lucky tickets {}, all suvivors {}, # of lucky tickets {}'.format(surv_lucky, surv_all, lucky_all))


# What is the overall chance of surviving and a chance to survive with lucky ticket:

# In[ ]:


surv_chance = len(df[(df.Survived == 1)])/len(df)
surv_chanve_lucky = len(df[(df.Survived == 1)&(df.lucky == 1)])/len(df[(df.lucky == 1)])
print('Chance to survive {:.2f}, chance to survive with lucky ticket {:.2f}'.format(surv_chance, surv_chanve_lucky ))


# Let's get rid of some columns and perform one-hot

# In[ ]:


df.drop('Ticket', axis=1, inplace=True)
df.drop('PassengerId', axis=1, inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df['emb_S'] = pd.Series(map(lambda x: 1 if x == 'S' else 0 , df.Embarked.values))
df['emb_C'] = pd.Series(map(lambda x: 1 if x == 'C' else 0 , df.Embarked.values))
df['emb_Q'] = pd.Series(map(lambda x: 1 if x == 'Q' else 0 , df.Embarked.values))
df['sex'] = pd.Series(map(lambda x: 1 if x == 'male' else 0 , df.Sex.values))
df.drop('Embarked', axis=1, inplace=True)
df.drop('Sex', axis=1, inplace=True)
df.drop('Name', axis=1, inplace=True)


# And impute mission age data

# In[ ]:


from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='median', axis=0)
imr = imr.fit(df.values)
X = imr.transform(df.values)


# Now let's plot correlation matrix for different features

# In[ ]:


import numpy as np
import seaborn as sns
cm = np.corrcoef(X.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',annot_kws={'size': 10},yticklabels=df.columns,xticklabels=df.columns)


# Correlation between survival and having a lucky ticket is small - magic doesn't work. 
# However Pclass and Parch shows some correlation with having a lucky ticket.
# Alse this correlation matrix shows that sex, Pclass, Fare and embarkment in Cherbourg are somehow correlated with 
# Survived feature
