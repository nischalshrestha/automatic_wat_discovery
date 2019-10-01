#!/usr/bin/env python
# coding: utf-8

# # the simplest and  easiest way to solve the titanic problem as a beginner in the data science (with a great accuracy)

# ### importing the important packages for processing 

# In[4]:


import pandas as pd # pandas reads and manipulates the data  
import numpy as np  # numpy is basically a package for calculation 


# ### reading the training and the testing  from their location 

# In[5]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ### to see the first 2 lines of the data 

# In[6]:


train.head(2)


# ### get the count of  the null values (nan values = not a number) click [here](https://en.wikipedia.org/wiki/NaN) for more

# In[7]:


train.isnull().sum()


# ## our goal now is to  : 
# - fill tha nan values 
# - encode the values to numbers 
# ### encoding : (eg. from embarked (S) to (1) ) 
# - we use encoding because the models work with calculations and the data have to be numbers for the models to process

#  ### As a start we delete the passenger id since it gives no indication on the data whatsoever 
#  ### and we delete the Embarked since logically the departure area doesn't affect the probabilty of survival from the crash

# In[8]:


del train['PassengerId']
del train['Embarked'] 
# Embarked means where did people get to the chip from (and that has no indication on the data in a logical manner so we delete it )


# ### now we map the Sex to numbers (encoding)
# - mapping is basically changing the values to numbers (e.g from (male) to 1 )

# In[9]:


# map Sex
train.Sex=train.Sex.map({'male':1,'female':2})
# use the name to get (mr.,ms. ... etc)


# In[10]:


train.head(2)


# ### we notice on the passenger Name there is Mr. and Mrs. so it gives indication that we can use instead of the name for the models (since the models doesn't accept strings as we mentioned )
# - we use regex : it's a way to search strings by patterns (e.g in the data Mr is between ((, then space) and (,)) so we wrote the pattern for it and it got us the symbols (Mr,Ms,etc..) 
# ### we get the answer and save it on a new column 

# In[11]:


import re 

def search_pattern(index):
    return re.search(',.*[/.\b]',train.Name[index])[0][2:-1]

train['Social_name']=[search_pattern(counter) for counter in range(train.shape[0]) ]


# In[12]:


train.Social_name.unique() # see the unique values from the Social_name column 


# ### as we can see there is some names that the pattern couldn't catch so we do the miss-spelled words manually 

# In[13]:


# cleaning the things that the regex couldn't get
train.Social_name.replace({"""Mrs. Martin (Elizabeth L""":'Mrs',
                       'Mme':'Miss',
                       'Ms':'Miss',
                       'the Countess':'Countess',
                        'Mr. Carl':'Mr',
                        'Mlle':'Miss'},inplace=True)


# In[14]:


train.Social_name.unique()


# In[15]:


train.head(2)


# ### Now we have a new great column ! 
# ### but as always we have to encode it since we did everything for it 
# - after that we delete the passenger Name since we don't need it anymore

# In[16]:


train.Social_name.unique()


# In[17]:


# encoding the values into numbers
for index in range(len(train.Social_name.unique())):
    
    a=train.Social_name.unique()[index] # the string (e.g. (Mr.))
    train.Social_name.replace({a:index},inplace=True)

# delete the name because we don't need it anymore 
del train['Name']


# In[18]:


train.head(2)


# ## pretty cool huh  ? 
# ### now we head to fill the age nan values (we will do it using the Sex, Pclass and Social_name) 
# 
# 

# In[19]:


train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median() 
# this is why we use the 3 of them to get the best median ever to fill the nans with 


# In[20]:


# fill na age values given (Pclass , Sex , Social_name)
grouped=train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()

pclass=grouped.index.labels[0] 

sex=grouped.index.labels[1] 

social_name=grouped.index.labels[2]




# In[21]:


for counter in range(len(grouped.index.labels[1])):
    # HERE
    train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
              (train.Sex==train.Sex.unique()[sex[counter]]) &
              (train.Social_name==train.Social_name.unique()[social_name[counter]])),
              'Age']=\
    train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
              (train.Sex==train.Sex.unique()[sex[counter]]) &
              (train.Social_name==train.Social_name.unique()[social_name[counter]])),
              'Age'].fillna(value=grouped.values[counter])
    # THERE

# from HERE to THERE is the same as putting inplace=True on the fillna but it seems that inplace doesn't work for no specific reason . . . 
 # 


# In[22]:


train.head(2)


# ### so now we want to use the Cabin for something useful 
# - since there is alot of unique Cabins (use train.Cabin.unique() to view it ) and the Cabins are origionally divided into levels(see the picture bellow) we can reduce them by using the first letter as the Cabin level 
# - we fill the nan values as the letter 'N' so we don't have a problem incoding it without any errors 
# - then we change the Cabins into dummies because they're not complete and we need to delete the nan but use the real levels since they give a great indication on the survival rate
# #### p.s : the dummie variable is a way to make columns from categorical data and give an indication if it's there or not (1 or 0 ) (e.g if the passenger is in Cabin A rise 1 as true)
# 
# - then we concat them to the data (merge the new columns with the old columns )
# - after we used the cabin we delete it's column and the N column that have the nan values (since the values in it isn't known in what level is it ) 

# <img src='https://camo.githubusercontent.com/288d4af66c81e7bd7e506482f95cf1af9520731f/687474703a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d30422d797839555549704236755932685164336c66634538325a4555' >

# In[23]:


# name the cabin with it's first letter 

for x in range(len(train)):
    if pd.isnull(train.loc[x,'Cabin']):
        continue  # pass the nan values 
    else : 
        train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]

# filling the nan cabin with a defaulted value 
train.Cabin.fillna('N',inplace=True)

# add dummies to the data and concating them to the origional dataset 
train = pd.concat([train, pd.get_dummies(train.Cabin)], axis=1)

# delete the nan values and the origional Cabin column 
del train['N']
del train['Cabin']


# In[24]:


train.head(2)


# ### there is alot of unique Tickets (more than 600) and it's hard to encode that so we will delete it 
# ### (this is for the sake of making the notebook simple as possible )

# In[25]:


len(train.Ticket.unique())


# In[26]:


# i can't see any useful information from keeping the tickets so i will just delete it 
del train['Ticket']


# In[27]:


train.head(2)


# In[28]:


train.isnull().sum()


# ## i gathered all what we did above with one function 
# - when you edit the training data you have to edit the testing data as well (and make the same changes ) so we will do it on one function 
# - you should give the bad/missed up data and it will return it clean and nice 

# In[29]:


def main(train):
    
    import numpy as np 

    # delete the passenger id since it gives no indication on the data whatsoever 

    del train['PassengerId']
    # map Sex
    train.Sex=train.Sex.map({'male':1,'female':2})
    # use the name to get (mr.,ms. ... etc)
    import re 

    def search_pattern(index):
        return re.search(',.*[/.\b]',train.Name[index])[0][2:-1]

    train['Social_name']=[search_pattern(counter) for counter in range(train.shape[0]) ]

    # cleaning the things that the regex couldn't get 
    train.Social_name.replace({"""Mrs. Martin (Elizabeth L""":'Mrs',
                           'Mme':'Miss',
                           'Ms':'Miss',
                           'the Countess':'Countess',
                            'Mr. Carl':'Mr',
                            'Mlle':'Miss'},inplace=True)

    # mapping the values 
    for x in range(len(train.Social_name.unique())):
        a=train.Social_name.unique()[x]
        b=x

        train.Social_name.replace({a:b},inplace=True)


    # delete the name because we don't need it anymore 
    del train['Name']

    # fill na age values given (Pclass , Sex , Social_name)
    grouped=train.Age.groupby([train.Pclass,train.Sex,train.Social_name]).median()

    pclass=grouped.index.labels[0] ; sex=grouped.index.labels[1] ; social_name=grouped.index.labels[2]


    for counter in range(len(grouped.index.labels[1])):
        # HERE
        train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
                  (train.Sex==train.Sex.unique()[sex[counter]]) &
                  (train.Social_name==train.Social_name.unique()[social_name[counter]])),
                  'Age']=\
        train.loc[((train.Pclass==train.Pclass.unique()[pclass[counter]]) &
                  (train.Sex==train.Sex.unique()[sex[counter]]) &
                  (train.Social_name==train.Social_name.unique()[social_name[counter]])),
                  'Age'].fillna(value=grouped.values[counter])
        # THERE

    # from HERE to THERE is the same as putting inplace=True on the fillna but it seems that inplace doesn't work for no specific reason . . . 



    # map Embarked 
    train.Embarked=train.Embarked.map({'S':1,'C':2,'Q':3})

    # fill Embarked nans 
    train.Embarked.groupby(train.Embarked).count() # the max is 1 so we fill the nans with it 
    train.Embarked.fillna(1,inplace=True)

    # name the cabin with it's first letter 

    for x in range(len(train)):
        if pd.isnull(train.loc[x,'Cabin']):
            continue 
        else : 
            train.loc[x,'Cabin'] = train.loc[x,'Cabin'][0]

    # filling the nan cabin with a defaulted value 
    train.Cabin.fillna('N',inplace=True)

    # add dummies to the data and concating them to the origional dataset 
    train = pd.concat([train, pd.get_dummies(train.Cabin)], axis=1)

    # delete the nan values and the origional Cabin column 
    del train['N']
    del train['Cabin']
    

    


    # rounding the ages 

    train.Age=train.Age.values.round().astype(int)

     # i can't see any useful information from keeping the tickets so i will just delete it 

    del train['Ticket']

    # rounding the fares to give less unique numbers 

    train.Fare=train.Fare.round().astype(int)

    # Embarked means where did people get to the chip from (and that has no indication on the data in a logical manner so we delete it )
    del train['Embarked']
    return train


# ### the models are divided into:
# - models that hates continuous data and loves categorical (doesn't benefit from the continuous data )
# - models that deals with both of them equally 
# ### one of the models that  work greatly with continuous  data is called Logistic Regression
# - it's simple and easy to use and you can modify certain things with it and get a heigher score (this is out of the scope for this notebook ) 

# In[30]:


df_train = pd.read_csv('../input/train.csv')
 # read the tainng data 
df_train=main(df_train) # pass the data to the manipulating function
# there is train['T'] but there is only one value in the whole data so we will delete it 
#model_training['T'].sum() # unhash the line to see the number 
del df_train['T']
model_training=df_train.loc[:,df_train.columns!='Survived']
model_testing=df_train.loc[:,'Survived']


# In[31]:


df_test= pd.read_csv('../input/test.csv')
# df_test=main(df_test) # this code doesn't work because there is an nan value in the Fare (just one nan)
df_test.Fare.fillna(df_test.Fare.median(skipna=True),inplace=True) # filling the nan value with the median fare 
df_test=main(df_test) # now it works 


# ### importing the model and fitting the data into it (fitting the data is the way for the model to see the patterns in the data 
# ### predict is the way for the model to test the assumptions that it made (it's "results" here)

# In[32]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()

the_model=logistic.fit(model_training,model_testing)
results=the_model.predict(df_test) # the output is returned as a list 


# ### we get the PassengerId from the testing data and fill the other columns with the predicted results 
# - then we set the PassengerId as the index (check the file gender_submission.csv (from the 3 files that kaggle provides) to see the format)
# ### finally we save the data into a file and it's ready for the submission 

# In[33]:


final=pd.read_csv('../input/test.csv',usecols=['PassengerId'])
final['Survived']=results

final=final.set_index('PassengerId')

final.to_csv('final.csv') # 0.77511 on kaggle for that 


# ##  by doing thoese simple steps you can get a score of %77.5 accuracy from kaggle
# - you can run the code from the function and bellow and you will get the same results 

# ## I had so much fun making the kernal for you and i hope it was good for you 

# # please upvote the kernal if you find it helpful 
