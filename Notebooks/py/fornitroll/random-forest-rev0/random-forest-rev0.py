#!/usr/bin/env python
# coding: utf-8

# ### Random Forest Classifier is used in this Kernel. Dashboard score is 0.78468

# In[262]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()
import re
#Sklearn OneHot Encoder to Encode categorical integer features
from sklearn.preprocessing import OneHotEncoder
#Sklearn train_test_split to split a set on train and test 
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split      # for old sklearn version use this to split a dataset 
# Random Forest Classifier from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# In[385]:


#Import the training data set
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
data.head()


# In[386]:


data.isnull().sum()


# In[387]:


test.isnull().sum()


# In[388]:


#Construct an X matrix
x_train = data[['Name', 'Pclass','Sex','Age','Parch','SibSp','Embarked', 'Fare', 'Cabin', 'Survived']].copy()
x_test = test[['Name', 'Pclass','Sex','Age','Parch','SibSp','Embarked', 'Fare', 'Cabin']].copy()
x_train.shape, x_test.shape


# In[389]:


PassengerID = np.array(test['PassengerId'])


# ## Embarked data

# In[390]:


print(x_train.Embarked.unique())
print(x_test.Embarked.unique())
set(x_train.Embarked.unique()) == set(x_test.Embarked.unique())   # CHeck that values in the train and in the test were similar


# So, there is a NaN value in the train data, only 2 rows, as we have seen ealier, lets drop them

# In[391]:


x_train = x_train.dropna(subset=['Embarked'],axis=0)


# In[392]:


print(x_train.Embarked.unique())
set(x_train.Embarked.unique())==set(x_test.Embarked.unique())   # CHeck that values in the train and in the test were similar


#  replace these values with (0,1,2)

# In[393]:


x_train.Embarked = pd.factorize(x_train.Embarked)[0]
x_test.Embarked = pd.factorize(x_test.Embarked)[0]


# In[394]:


x_train.head()


# ## Sex data

# There are np missed values so factorize these values (0,1)

# In[395]:


x_train.Sex = pd.factorize(x_train.Sex)[0]
x_test.Sex = pd.factorize(x_test.Sex)[0]


# ## Sibsp and Parch data
# #### We should be carefull with these data to not overfit our model. Lets create a Family feature, which show a size of family, and the feture which show is a passenger alone

# In[396]:


x_train['Family'] = x_train['SibSp'] + x_train['Parch']
x_test['Family'] = x_test['SibSp'] + x_test['Parch']

x_train['Alone'] = x_train['Family'].map(lambda x: 1 if x==0 else 0)
x_test['Alone'] = x_test['Family'].map(lambda x: 1 if x==0 else 0)


# ## Age data

# ### I am going to firstly categorize these data and after make a factorization. But first of all lets define how to categorize the Ages, what intervals will be most efficient, and how to deal with missed values

# In[397]:


# Find a mean Age in overall data
age = pd.concat([x_test.Age, x_train.Age], axis=0)


# In[398]:


mean = age[1].mean()


# In[399]:


# Identify the rows with missed Age in special column
x_train['Missed_Age'] = x_train['Age'].map(lambda x: 1 if pd.isnull(x)  else 0)
x_test['Missed_Age'] = x_test['Age'].map(lambda x: 1 if pd.isnull(x) else 0)


# In[400]:


# Fill all age values with Age mean
x_train['Age'] = x_train['Age'].fillna(mean)
x_test['Age'] = x_test['Age'].fillna(mean)


# In[401]:


data[data.Survived==1].Age.plot.hist(alpha=0.5,color='blue',stacked=True, bins=50)
data[data.Survived==0].Age.plot.hist(alpha=0.5,color='red', stacked=True, bins=50)
plt.legend(['Survived','Died'])
plt.show()


# In[402]:


sns.countplot(x="Survived", data=data[data['Age'].isnull()])


# In[403]:


sns.countplot(x="Survived", data=data[data['Age'].isnull()], hue='Pclass')


# In[404]:


def process_age(df,cut_points,label_names):
    df["Age"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5,16,100]        
label_names = [0,1,2,3]

x_train = process_age(x_train,cut_points,label_names)
x_test = process_age(x_test,cut_points,label_names)


# In[405]:


set(x_train['Age'].unique()) == set(x_test['Age'].unique())


# In[406]:


x_train.head()


# ## Fare data

# In[407]:


# Fill one missed fare in the train set with mean Fare for this class
x_test.loc[x_test['Fare'].isnull()]['Pclass']  # determine a Class for this passenger


# In[408]:


# Find the mean Fare for Class 3
fare_mean = pd.concat([x_train.loc[x_train['Pclass']==3]['Fare'], x_test.loc[x_test['Pclass']==3]['Fare']], axis=0).mean()


# In[409]:


# Fill the data gap
x_test['Fare'] = x_test['Fare'].fillna(fare_mean)


# In[410]:


x_test.isnull().sum()


# In[411]:


x_train['Fare'] = (x_train['Fare']/20).astype('int64')
x_test['Fare'] = (x_test['Fare']/20).astype('int64')


# In[412]:


set(x_train['Fare'].unique()) == set(x_test['Fare'].unique()) # Check the train and test data identity


# ## Cabin data

# In[413]:


# There a lot of missed values so lets just check do passenger have a Cabin number or not


# In[414]:


x_train['Missed_Cabin'] = x_train['Cabin'].map(lambda x: 0 if pd.isnull(x)  else 1)
x_test['Missed_Cabin'] = x_test['Cabin'].map(lambda x: 0 if pd.isnull(x) else 1)


# Also we can see that some passenger has a few cabins number, lets make a special column, where missed cabin will be zero, 
# and 1,2.... so on the number of cabins

# In[415]:


x_train['Cabin_num'] = x_train['Cabin'].map(lambda x: 0 if pd.isnull(x)  else len(x.split()))
x_test['Cabin_num'] = x_test['Cabin'].map(lambda x: 0 if pd.isnull(x) else len(x.split()))


# In[416]:


x_train.head()


# ## Name data

# In[417]:


# Lets try to extract a Title data from name using regular expression
x_train['Title'] = x_train['Name'].map(lambda x: str(re.findall("^.*[, *](.*)[.] *", x)[0]))
x_test['Title'] = x_test['Name'].map(lambda x: str(re.findall("^.*[, ](.*)[.] *", x)[0]))


# In[418]:


x_train['Title'].unique()


# By the way - Wiki: Count (male) or countess (female) is a title in European countries for a noble of varying status, but historically deemed to convey an approximate rank intermediate between the highest and lowest titles of nobility

# In[419]:


sns.countplot(x="Title", data=x_train)


# In[420]:


x_train.Title = pd.factorize(x_train.Title)[0]
x_test.Title = pd.factorize(x_test.Title)[0]


# In[421]:


x_train.head()


# Lets also count define length

# In[422]:


x_train['Name_Len_char'] = x_train['Name'].map(lambda x: len(x))
x_train['Name_Len_words'] = x_train['Name'].map(lambda x: len(x.split()))

x_test['Name_Len_char'] = x_test['Name'].map(lambda x: len(x))
x_test['Name_Len_words'] = x_test['Name'].map(lambda x: len(x.split()))


# In[423]:


x_train.head()


# Create Y-array with SUrvived values

# In[424]:


#Create Y array
y = np.array(x_train[['Survived']])
print(y.shape)


# Drop Columns which we do not need

# In[425]:


x_train=x_train.drop(['SibSp', 'Parch', 'Name', 'Cabin', 'Survived'], axis=1)
x_test=x_test.drop(['SibSp', 'Parch', 'Name', 'Cabin'],axis=1)


# In[426]:


x_train.head()


# Create Test and Train sets

# In[430]:


xn_train, xn_test, yn_train, yn_test = train_test_split(x_train, y, test_size=0.3, random_state=32)
xn_train.shape, xn_test.shape, yn_train.shape, yn_test.shape


# In[431]:


# We can optimize the parameters using special function in sclearn, but here I will do it manually
C=np.array([100,150,200,250,300,350,400,450,500,550,600,650,700,750])
scores = np.zeros(C.shape)
for i in range (len(C)):
        clf = RandomForestClassifier(n_estimators = int(C[i]), max_depth=10, random_state=0, criterion='entropy') 
        clf.fit(xn_train, yn_train) 
        scores[i] = clf.score(xn_test,yn_test)


# In[432]:


ind = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
print('max Score = ',scores[ind],'\noptimal C = ',C[ind])


# In[433]:


clf = RandomForestClassifier(n_estimators = 150, max_depth=10, random_state=0, criterion='entropy') 
clf.fit(xn_train, yn_train) 
print(clf.score(xn_train,yn_train))
print(clf.score(xn_test,yn_test))


# In[434]:


importance = clf.feature_importances_


# In[435]:


importance = pd.DataFrame(importance, index=x_test.columns, 
                          columns=["Importance"])


# In[436]:


print(importance)


# In[381]:


clf = RandomForestClassifier(n_estimators = 100, max_depth=10, random_state=0, criterion='entropy') 
clf.fit(x_train, y) 
prediction = clf.predict(x_test)


# In[382]:


print(clf.score(xn_train,yn_train))
print(clf.score(xn_test,yn_test))
print(clf.score(x_train,y))


# In[383]:


# Submit the result

submission_df = {"PassengerId": PassengerID,
                 "Survived": prediction}
submission = pd.DataFrame(submission_df)


# In[384]:


submission.to_csv("submission.csv",index=False)

