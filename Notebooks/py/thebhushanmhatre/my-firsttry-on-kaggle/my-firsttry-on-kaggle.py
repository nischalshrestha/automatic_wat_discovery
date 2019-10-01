#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
full_data = [train_data, test_data]
train_data.head(3)


# In[ ]:


test_data.head(3)


# In[ ]:


train_data.info()


# There are 891 entries so every column must contain 891 but as we can see above **Age**, **Cabin** and **Embarked** doesn't have 891 entries so they must have some Missing Values

# In[ ]:


test_data.info()


# Here **Age**, **Cabin**, **Fare** have Missing Values

# - So there are Missing Values in Both Training Data and Testing Data
# - So we will need to Fill both Data

# ## Feature Engineering
# - Go through all The Features One by one against  **Survived**
# - Since there is No Survived Column for Test Data , we wil only Deal with training data for this Section

# In[ ]:


train_data.columns


# In[ ]:


# Pclass 
train_data[['Pclass','Survived']].groupby('Pclass', as_index=False).mean()


# In[ ]:


# Name is text data We can extract information like name-initials (like Mr. Sir. Mrs. Miss. ) 
# May be this can be used as Class between ppl or to show Respect (Experimenting;)
def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''
for df in full_data:
    df['Title'] = df['Name'].apply(get_title)

train_data[['Title','Survived']].groupby('Title', as_index=False).mean()


# In[ ]:


# Sex
train_data[['Sex','Survived']].groupby('Sex', as_index=False).mean()


# In[ ]:


# Age - Lets Make Age-Groups and then Find Survival Rate

# Getting Rid of Missing Values from Age column (Part of Cleaning Data Step)
for df in full_data:
    age_avg = df['Age'].mean()
    age_std = df['Age'].std()
    age_null_count = df['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    df['Age'][np.isnan(df['Age'])] = age_null_random_list
    df['Age'] = df['Age'].astype(int)


# Mapping Age
for dataset in full_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4



train_data[['Age','Survived']].groupby('Age', as_index=False).mean()


# In[ ]:


# SibSp & Parch
# With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size.
# Since we are Making New Column we can Delete these Columns in Data Cleaning Step
for df in full_data:
    df['FamSize'] = df['SibSp'] + df['Parch'] + 1

train_data[['FamSize','Survived']].groupby('FamSize', as_index=False).mean()


# In[ ]:


# Fare Missing in Testing Data
for df in full_data:
    
    df['Fare'] = df['Fare'].fillna(train_data['Fare'].median())
    
    # Mapping Fare (Part of Data Cleaning Data)
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 4

train_data[['Fare','Survived']].groupby('Fare', as_index=False).mean()


# In[ ]:


# Cabin
for df in full_data:
    # Those with Cabin Seats be 1 and others be 0 (Nan)
    df['Cabin'].fillna(0, inplace =True)
    #df.loc[df['Cabin']==np.nan, 'Cabin'] = 0
    df.loc[df['Cabin']!= 0, 'Cabin'] = 1

train_data[['Cabin','Survived']].groupby('Cabin', as_index=False).mean()


# In[ ]:


# Embarked
for df in full_data:
    df['Embarked'] = df['Embarked'].fillna('S')

train_data[['Embarked','Survived']].groupby('Embarked', as_index=False).mean()


# # Cleaning Data

# In[ ]:


for df in full_data:
    
    # Getting Rid of Useless Columns
    df.drop(['Name','PassengerId','SibSp','Parch','Ticket'], axis=1, inplace=True)
    
    # Converting into desired Datatypes
    df['Sex'] = df['Sex'].astype('category').cat.codes
    df['Title'] = df['Title'].astype('category').cat.codes
    df['Embarked'] = df['Embarked'].astype('category').cat.codes


# In[ ]:


train_data.head(3)


# In[ ]:


test_data.tail(3)


# In[ ]:


train_data.info()


# In[ ]:


test_data.info()


# # Successfully Converted All Columns into Numeric Data

# # Skipping the Data Visualization Step (for now)

# In[ ]:


plt.figure(figsize=(15,8))
plt.imshow(train_data.corr(), cmap=plt.cm.Blues, interpolation='nearest')  # or plt.cm.RdBu
plt.colorbar( )
tick_marks = [i for i in range(len(train_data.columns))]
plt.xticks(tick_marks, train_data.columns, rotation='vertical')
plt.yticks(tick_marks, train_data.columns)

plt.show()


# In[ ]:


y = train_data['Survived']
X = train_data.drop(['Survived'],axis=1)


# # ML

# In[ ]:


from sklearn.tree import DecisionTreeClassifier as dtc
model1 = dtc()
model1.fit(X, y)

from sklearn.neighbors import KNeighborsClassifier as knc
model2 = knc(n_neighbors=5)
model2.fit(X, y)

from sklearn.svm import SVC
model4 = SVC(C=1.0, kernel='rbf', degree=3)
model4.fit(X, y)

from sklearn.ensemble import RandomForestClassifier as rfc
model3 = rfc(n_estimators=100, max_depth=3, max_features=0.5, min_samples_leaf = 32)
model3.fit(X, y)

modelx = rfc(n_estimators=100, max_depth=3, max_features=0.5, min_samples_leaf = 50)
modelx.fit(X, y)


# When I set min_samples_leaf = 50; I get Test Score of 100 and train score of 81.6;
# Any plausible explanation will be appreciated.

# In[ ]:


#DTC
score01 = model1.score(X, y)

#KNC
score02 = model2.score(X, y)

#RFC
score03 = model3.score(X, y)

#SVC
score04 = model4.score(X, y)


score_x = modelx.score(X, y)


# In[ ]:


print("Training Scores: ")
print("Decision Tree Classifier : ", score01*100)
print("K Neighbors Classifier   : ", score02*100)
print("Support Vector Classifier: ", score04*100)
print("Random Forest Classifier : ", score03*100)

print("Trial & Error RFC model  : ", score_x*100)


# ## Scores on Testing Data
# The Scores that Really matter

# In[ ]:


Xt = test_data
yt = pd.read_csv('../input/gender_submission.csv')['Survived']


# In[ ]:


#DTC
score1 = model1.score(Xt, yt)

#KNC
score2 = model2.score(Xt, yt)

#SVC
score4 = model4.score(Xt, yt)

# The following Random Forest Classifier Test score is the best result
#RFC
score3 = model3.score(Xt, yt)

scorex = modelx.score(Xt, yt)


# In[ ]:


print("Testing Scores: ")
print("Decision Tree Classifier : ", score1*100)
print("K Neighbors Classifier   : ", score2*100)
print("Support Vector Classifier: ", score4*100)
print("Random Forest Classifier : ", score3*100)

print("Trial & Error RFC model  : ", scorex*100)


# In[ ]:


print("Random Forest Classifier train : ", score03*100)
print("Random Forest Classifier test  : ", score3*100)
print("Trial & Error RFC model  : ", scorex*100)


# # Submitting

# In[ ]:


test_file = pd.read_csv('../input/test.csv')
pred = modelx.predict(Xt)
sub_df = pd.DataFrame({'PassengerId':test_file['PassengerId'], 'Survived':pred})
sub_df.head()


# In[ ]:


sub_df.to_csv('RF_result.csv', index = False)


# Self Note: After this Submission, Go to Output and then click on Submit predictions which uploads our result

# # THE END

# 

# 

# ignore the following ;)

# ## *Without Neural network; Random Forest gives Best Result for this Dataset *
# **Now Lets Dive Deep**

# # Going Deep

# In[ ]:


from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,Dropout


# In[ ]:


print('Shape of X is : ', X.shape)
print('Shape of y is : ', y.shape)


# In[ ]:


# I picked this Layer architecture by Trail and Error
dmodel = Sequential()
dmodel.add(Dense(200, activation = 'relu', input_shape = (8,)))
dmodel.add(Dense(150, activation = 'relu'))
dmodel.add(Dense(50, activation = 'relu'))
dmodel.add(Dense(25, activation = 'relu'))
dmodel.add(Dense(150, activation = 'relu'))
dmodel.add(Dense(1, activation='sigmoid'))


# In[ ]:


dmodel.summary()


# In[ ]:


dmodel.compile(loss = keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Training data
dmodel.fit(X, y, batch_size=50, epochs=6, validation_split=0.2)


# In[ ]:


# Score on Testing Data
dmodel.evaluate(Xt, yt)[1]


# In[ ]:


# predict against test set
predictions = dmodel.predict(Xt)
predictions = predictions.ravel() # To reduce ND-array to 1D-array
#data_to_submit = pd.DataFrame({'PassengerId': test_index, 'Survived': predictions})
# output results to results.csv
#data_to_submit.to_csv("results.csv", index=False)


# ## $To$ $be$ $Continued...$
# I have used all ML algos with Default Parameters and Visualization stuff has not been Incorporated
