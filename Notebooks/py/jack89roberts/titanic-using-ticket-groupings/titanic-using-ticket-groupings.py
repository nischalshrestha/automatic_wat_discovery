#!/usr/bin/env python
# coding: utf-8

# # Titanic
# 
# This is a quick response to some thoughts I had on [Chris Deotte's](https://www.kaggle.com/cdeotte) great kernel [Titanic using Name only](https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818/notebook), which I definitely suggest you look at if you haven't already.
# 
# To sumarise quickly, his approach makes predictions based on the following rules:
# 
# * All males die except boys in families where all females and boys live.
# 
# * All females live except those in families where all females and boys die.
# 
# Boys are defined as passengers with the title "Master", and families are defined as groups of people with the same surname.
# 
# I was wondering what would happen if you group passengers sharing the same ticket number, instead of passengers with the same surname. My main thought process for this was:
# 
# * Passengers with the same surname may not come from the same family (could come from multiple families).
# 
# * There may be important non-surname based groups, for example travelling friends, non-married couples etc.  
# 
# So this notebook uses Chris' approach with ticket grouping and compares the results.

# In[1]:


# load the data
import pandas as pd
df = pd.read_csv('../input/train.csv',index_col='PassengerId')


# In[2]:


# select females and masters (boys)
boy = (df.Name.str.contains('Master')) | ((df.Sex=='male') & (df.Age<13))
female = df.Sex=='female'
boy_or_female = boy | female

# no. females + boys on ticket
n_ticket = df[boy_or_female].groupby('Ticket').Survived.count()

# survival rate amongst females + boys on ticket
tick_surv = df[boy_or_female].groupby('Ticket').Survived.mean()


# In[3]:


# function to create relevant features for test data
def create_features(frame):
    frame['Boy'] = (frame.Name.str.contains('Master')) | ((frame.Sex=='male') & (frame.Age<13))
    
    frame['Female'] = (frame.Sex=='female').astype(int)

    # if ticket exists in training data, fill NTicket with no. women+boys
    # on that ticket in the training data.
    frame['NTicket'] = frame.Ticket.replace(n_ticket)
    # otherwise NTicket=0
    frame.loc[~frame.Ticket.isin(n_ticket.index),'NTicket']=0

    # if ticket exists in training data, fill TicketSurv with
    # women+boys survival rate in training data  
    frame['TicketSurv'] = frame.Ticket.replace(tick_surv)
    # otherwise TicketSurv=0
    frame.loc[~frame.Ticket.isin(tick_surv.index),'TicketSurv']=0

    # return data frame only including features needed for prediction
    return frame[['Female','Boy','NTicket','TicketSurv']]


# In[4]:


# predict survival for a passenger
def did_survive(row):
    if row.Female:
        # predict died if all women+boys on ticket died
        if (row.NTicket>0) and (row.TicketSurv==0):
            return 0
        # predict survived for all other women
        else:
            return 1
        
    elif row.Boy:
        # predict survived if all women+boys on ticket survived
        if (row.NTicket>0) and (row.TicketSurv==1):
            return 1
        # predict died for all other boys
        else:
            return 0
        
    else:
        # predict all men die
        return 0


# In[6]:


# load test data
df_test = pd.read_csv('../input/test.csv',index_col='PassengerId')

# extract the features to use
X = create_features(df_test)

# predict test data
pred = X.apply(did_survive,axis=1)

# create submission file
pred = pd.DataFrame(pred) 
pred.rename(columns={0:'Survived'},inplace=True)
pred.to_csv('submission.csv')

print(pred.Survived.value_counts())


# # Results
# 
# This approach gives me a leaderboard score of 0.813 compared to Chris Deotte's score of 0.823, so is marginally worse than the surname based approach.
# 
# By comparing with his output file, I see that I predict three passengers in the test data survive, whilst Chris predicts they die. A quick bit of Maths suggests I need to predict two more passengers correctly to match Chris' score, which I think means exactly two of these passengers are in the public test set.
# 
# The three passengers in question are:

# In[7]:


# passengers where I predict differently
# (based on comparison with Chris Deotte's output file)
id_nomatch = [910,929,1172]

# display these passengers
display(df_test.loc[id_nomatch])


# And the passengers in the training data with surnames matching the three test data passengers above are:

# In[8]:


# families in training data with member in test data that I predict differently
display(df.loc[df.Name.str.contains('Ilmakangas')])
display(df.loc[df.Name.str.contains('Cacic')])
display(df.loc[df.Name.str.contains('Oreskovic')])


# The Ilmakangas appear to be adult sisters travelling together (indicated by SibSp=1), but who bought tickets separately. But the Cacics and Oreskovics have no siblings, spouses, children or parents onboard according to the data. However, they all travel in the same class, with the same fare and with similar ticket numbers. They may be unrelated, but are likely to be travelling cousins/similar who bought tickets separately.
# 
# In summary, whilst I was hoping grouping by ticket would catch a greater variety of groups I'd missed the point that groups of friends etc. are very likely to have bought tickets separately. So only grouping by ticket appears to miss a few relationships, rather than catch more. Also, as the additional rules (vs. a gender only prediction) only apply to groups of women and boys who either all survive or die, there are not so many examples and cases of unrelated families having the same surname are unlikely.
