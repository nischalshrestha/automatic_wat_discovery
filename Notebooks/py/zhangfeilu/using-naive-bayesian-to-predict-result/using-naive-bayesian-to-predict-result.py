#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')
# Load the training data in a dataframe
train = pd.read_csv("../input/train.csv")

# Load the test data in a dataframe
test = pd.read_csv("../input/test.csv")
# test data don't have survived value， but when we doing feature engineer put it together with train data
merged = train.append(test)
#data.astype
merged.head()


# In[2]:


print(merged.isnull().sum()) #checking for total null values, null survived value from test data


# The **Age, Cabin and Embarked **have null values. I will try to fix them.

# As we had seen earlier, the Age feature has 177 null values. To replace these NaN values, we can assign them the mean age of the dataset.
# 
# But the problem is, there were many people with many different ages. We just cant assign a 4 year kid with the mean age that is 29 years. Is there any way to find out what age-band does the passenger lie??
# 
# Bingo!!!!, we can check the Name feature. Looking upon the feature, we can see that the names have a salutation like Mr or Mrs. Thus we can assign the mean values of Mr and Mrs to the respective groups.
# 
# Okay so there are some misspelled Initials like Mlle or Mme that stand for Miss. I will replace them with Miss and same thing for other values.

# In[3]:


merged['NameTitle']=0
for i in merged:
    merged['NameTitle']=merged.Name.str.extract('([A-Za-z]+)\.') #lets extract the Salutations
print(pd.unique(merged['NameTitle'].values))    


# In[4]:


merged['NameTitle'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[5]:


merged.groupby('NameTitle')['Age'].mean() #lets check the average age by Initials


# In[6]:


## Assigning the NaN Values with the Ceil values of the mean ages
merged.loc[(merged.Age.isnull())&(merged.NameTitle=='Mr'),'Age']=33
merged.loc[(merged.Age.isnull())&(merged.NameTitle=='Mrs'),'Age']=37
merged.loc[(merged.Age.isnull())&(merged.NameTitle=='Master'),'Age']=5
merged.loc[(merged.Age.isnull())&(merged.NameTitle=='Miss'),'Age']=22
merged.loc[(merged.Age.isnull())&(merged.NameTitle=='Other'),'Age']=45
merged.loc[(merged.Age.isnull())&(merged.NameTitle=='Dona'),'Age']=39


# In[ ]:


merged.Age.isnull().any()


# In[7]:


embarked_values = pd.unique(merged['Embarked'].values)
for v in embarked_values:
    print(v,len(merged.loc[merged['Embarked']==v]))


# In[9]:


# As we saw that maximum passengers boarded from Port S, we replace NaN with S.
merged['Embarked'].fillna('S',inplace=True)


# Part2: Feature Engineering and Data Cleaning
# Now what is Feature Engineering?
# 
# Whenever we are given a dataset with features, it is not necessary that all the features will be important. There maybe be many redundant features which should be eliminated. Also we can get or add new features by observing or extracting information from other features.
# 
# An example would be getting the Initals feature using the Name Feature. Lets see if we can get any new features and eliminate a few. Also we will tranform the existing relevant features to suitable form for Predictive Modeling.
# 
# Age_band
# Problem With Age Feature:
# As I have mentioned earlier that Age is a continous feature, there is a problem with Continous Variables in Machine Learning Models.
# 
# Eg:If I say to group or arrange Sports Person by Sex, We can easily segregate them by Male and Female.
# 
# Now if I say to group them by their Age, then how would you do it? If there are 30 Persons, there may be 30 age values. Now this is problematic.
# 
# We need to convert these continous values into categorical values by either Binning or Normalisation. I will be using binning i.e group a range of ages into a single bin or assign them a single value.
# 
# Okay so the maximum age of a passenger was 80. So lets divide the range from 0-80 into 5 bins. So 80/5=16. So bins of size 16.

# In[10]:


merged['Age_band']=0
merged.loc[merged['Age']<=16,'Age_band']=0
merged.loc[(merged['Age']>16)&(merged['Age']<=32),'Age_band']=1
merged.loc[(merged['Age']>32)&(merged['Age']<=48),'Age_band']=2
merged.loc[(merged['Age']>48)&(merged['Age']<=64),'Age_band']=3
merged.loc[merged['Age']>64,'Age_band']=4


# Family_Size and Alone
# At this point, we can create a new feature called "Family_size" and "Alone" and analyse it. This feature is the summation of Parch and SibSp. It gives us a combined data so that we can check if survival rate have anything to do with family size of the passengers. Alone will denote whether a passenger is alone or not.

# In[11]:


merged['Family_Size']=0
merged['Family_Size']=merged['Parch']+merged['SibSp']#family size
merged['Alone']=0
merged.loc[merged.Family_Size==0,'Alone']=1#Alone
merged.head()


# Fare_Range
# Since fare is also a continous feature, we need to convert it into ordinal value. For this we will use pandas.qcut.
# 
# So what qcut does is it splits or arranges the values according the number of bins we have passed. So if we pass for 5 bins, it will arrange the values equally spaced into 5 seperate bins or value ranges.

# In[12]:


merged['Fare_Range']=pd.qcut(merged['Fare'],4)
#print(merged.groupby(['Fare_Range'])['Survived'])


# In[13]:


#Now we cannot pass the Fare_Range values as it is. We should convert it into singleton values same as we did in Age_Band
merged['Fare_cat']=0
merged.loc[merged['Fare']<=7.91,'Fare_cat']=0
merged.loc[(merged['Fare']>7.91)&(merged['Fare']<=14.454),'Fare_cat']=1
merged.loc[(merged['Fare']>14.454)&(merged['Fare']<=31),'Fare_cat']=2
merged.loc[(merged['Fare']>31)&(merged['Fare']<=513),'Fare_cat']=3


# In[14]:


#data.head()
data = pd.DataFrame(merged.head(len(train)))
submission_data = pd.DataFrame(merged.iloc[len(train):]) 
# after the feature engineering we need to split the test data from the train data
data.head()


# Dropping UnNeeded Features
# Name--> We don't need name feature as it cannot be converted into any categorical value.
# 
# Age--> We have the Age_band feature, so no need of this.
# 
# Ticket--> It is any random string that cannot be categorised.
# 
# Fare--> We have the Fare_cat feature, so unneeded
# 
# Cabin--> A lot of NaN values and also many passengers have multiple cabins. So this is a useless feature.
# 
# Fare_Range--> We have the fare_cat feature.
# 
# PassengerId--> Cannot be categorised.

# In[41]:


#remove useless columns, it's useless column
#data.drop(['Name','Age','Ticket','Fare','Cabin','Fare_Range','PassengerId'],axis=1,inplace=True)

total_count = len(data)
train_count = int(len(data) * 0.8)
test_count = total_count - train_count

train_data = data[:train_count]
test_data = data[train_count-1:-1]
# define columns of X and column of Y
x_columns = ['Pclass','Sex','Age_band','SibSp','Parch','Fare_cat','Embarked','Alone','Family_Size','NameTitle']
y_columns = ['Survived']
train_data_X = train_data[x_columns]
train_data_Y = train_data[y_columns]
test_data_X = test_data[x_columns]
test_data_Y = test_data[y_columns]
test_data_X.head()
#print(total_count,train_count, test_count, len(train_data),len(test_data))
#891 712 179 712 179
p0 = len(train_data.loc[train_data['Survived']==0])/len(train_data)
p1 = len(train_data.loc[train_data['Survived']==1])/len(train_data)


# In[53]:


# get each attribute conditional probability
def getConditionProb(data,attribute,y_attribute):
    probDict = {}
    data_y0 = data.loc[data[y_attribute]==0]
    data_y1 = data.loc[data[y_attribute]==1]
    y0_count = len(data_y0)
    y1_count = len(data_y1)
    for att in attribute:
        att_values = pd.unique(data[att].values)
        for att_v in att_values:
            # laplace smoothing
            #print("Pclass=3", len(data_y0.loc[data_y0[]]))
            p_att_v_y0 = (len(data_y0.loc[data_y0[att]==att_v]) + 1)/(y0_count + len(att_values))
            p_att_v_y1 = (len(data_y1.loc[data_y1[att]==att_v]) + 1)/(y1_count + len(att_values))
            #print("att:" , att , ", att_v:" , att_v, p_att_v_y0, p_att_v_y1)
            y0_key= str(att) + "_" + str(att_v) + "_y0"
            y1_key= str(att) + "_" + str(att_v) + "_y1"
            #probDict[att  "_"  att_v  "_y0"] = p_att_v_y0
            #probDict[att  "_"  att_v  "_y1"] = p_att_v_y1
            probDict[y0_key] = p_att_v_y0
            probDict[y1_key] = p_att_v_y1
    return probDict


# In[59]:


# use only sex attribute to predict
# train the model

def classifyPassenger(passenger, attrs,probDict, p0, p1):
    # use log to avoid overflow
    p_survived = np.log(p1)  
    p_not_survived = np.log(p0)  
    for att in attrs:
        att_v = passenger[att]
        y0_key= str(att) + "_" + str(att_v) + "_y0"
        y1_key= str(att) + "_" + str(att_v) + "_y1"
        #print(y1_key + ":" + str(probDict[y1_key]))
        #print(y0_key + ":" + str(probDict[y0_key]))
        p_survived = p_survived + np.log(probDict[y1_key])
        p_not_survived = p_not_survived + np.log(probDict[y0_key])
    if p_survived > p_not_survived:
        return 1
    return 0

def get_err_rate(data,data_Y):
    data_Y['predict'] = data.apply(lambda row: classifyPassenger(row,['Sex','Age_band'],probDict,p0,p1),axis=1)
    data_Y['notEqual'] = data_Y['predict'] - data_Y['Survived']
    #data.head(20)
    # calculate the error rate
    err_rate = len(data_Y.loc[data_Y['notEqual']!=0])/len(data_Y)
    print(err_rate)
    return err_rate
probDict = getConditionProb(data, x_columns, 'Survived')

train_err_rate = get_err_rate(data, data)
#test_err_rate = get_err_rate(test_data_X, test_data_Y)
#print(train_err_rate)
# create a submission file, use sex only the score is 0.76555
#test_data_for_submission = pd.read_csv('../input/test.csv')# test data don't have survived value
submission_data['predict'] = submission_data.apply(lambda row: classifyPassenger(row,['Sex','Age_band'],probDict,p0,p1),axis=1)
submission01 = pd.DataFrame({'PassengerId':submission_data['PassengerId'],'Survived':submission_data['predict']})
submission01.to_csv('submission01.csv',index=False)    


# In[ ]:




