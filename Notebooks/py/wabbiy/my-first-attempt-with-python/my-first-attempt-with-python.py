#!/usr/bin/env python
# coding: utf-8

# **Titanic Dataset**
# * This is a test of my python knowledge and a chance to identify areas of improvement 
# * Suggestions and upvotes are highly welcomed.

# In[1]:


#load necessary modules
import numpy as np 
import pandas as pd 
from sklearn import tree
import seaborn as sns

import os
print(os.listdir("../input"))



# In[2]:


#load the train and test data
traind=pd.read_csv("../input/train.csv")
testd=pd.read_csv("../input/test.csv")


# In[3]:


#view the data ....head and tail
traind.head()


# In[4]:


traind.tail()


# In[5]:


#summary of all the data to check for any nulls, data types
traind.info()
#Age, Cabin, and Embarked have nulls
#Drop Cabin-is missing alot. . may be missing important information
#Drop PassengerId - merely a continuous number with probably no much to learn from
testd.info()


# In[6]:


traind.describe()
#age missing 205
#numeric variables - passengerId,Survived,Pclass,Age,SibSp,Parch,Fare


# Identify any relationship between the variables and the rate of survival
# 

# In[7]:


#train data shows 38% survived while 62% did not

sns.countplot(traind['Survived'])


# In[8]:


traind['Survived'].value_counts(normalize=True)


# In[9]:


#there was a lower chance of survival in 3rd class wich had lower socio-economic people

sns.countplot(traind['Pclass'], hue=traind['Survived'])


# In[ ]:


#Name ?


# In[10]:


#There seems to be some form of a relationship between rate of survival and sex. .more men 
#perished than those that survived.
#more women survived compered to those that died.
sns.countplot(traind['Sex'], hue=traind['Survived'])


# In[11]:


#Age
#fix the missing data to get a better picture of the variable? maybe later
#the ages seems to be relatively evenly distributed same are their survival rate
#A higher survival rate for the younger age group 0.4-19, however no clear correlation 

pd.qcut(traind['Age'],5).value_counts()


# In[12]:


sns.countplot (pd.qcut(traind['Age'],5), hue=traind['Survived'])


# In[13]:


traind['Survived'].groupby(pd.qcut(traind['Age'],5)).mean()


# In[14]:


#possible a high number of siblings affected survival rate?
traind['SibSp'].value_counts()


# In[15]:


traind['Survived'].groupby(traind['SibSp']).mean()


# In[16]:


#parc
#possible a higher number of children reduced chances of survival
traind['Parch'].value_counts()


# In[17]:


traind['Survived'].groupby(traind['Parch']).mean()


# In[18]:


#Tickets...ticket number could probably be an indication of how much paid, class, cabin one was in
# and thus a clear indication of the chances of survival
# The numbers are different...further analysis may reveal the meaning and implications there of
traind['Ticket'].head(30)


# In[19]:


# ther is an increase in the survival mean as the fare increases
pd.qcut(traind['Fare'],5).value_counts()


# In[20]:


traind['Survived'].groupby(pd.qcut(traind['Fare'],5)).mean()


# In[21]:


#most people embarked at Southampton, survival rate of those that embarked in cherbourg was higher
sns.countplot(traind['Embarked'], hue=traind['Survived'])


# Prepare the variables for analysis
# 1. Fill in the missing age values....
# 2. Fill in the 2 null values in embarked with S which has the highest count
# 3. convert categorical values to numeric for Embarked, Sex 
# 4. Drop Cabin, Ticket ,  passengerId(for train data)  and then name

# In[ ]:





# In[22]:


#drop Ticket, Cabin, PassengerId
print("Before", traind.shape, testd.shape, )

traind = traind.drop(['Ticket', 'Cabin', 'PassengerId'], axis=1)
testd = testd.drop(['Ticket', 'Cabin', ], axis=1)
combine = [traind, testd]

"After", traind.shape, testd.shape, combine[0].shape, combine[1].shape


# In[23]:


# convert sex to numeric
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

traind.head()


# In[24]:


#replace the missing values in the Embarked feature with S
traind = traind.fillna({"Embarked": "S"})
testd = testd.fillna({"Embarked": "S"})


# In[25]:


#convert Embarked to numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
traind['Embarked'] = traind['Embarked'].map(embarked_mapping)
testd['Embarked'] = testd['Embarked'].map(embarked_mapping)

traind.head()


# **Filling in the missing age values**
# -Decided to get the age by finding the mean of the individuals as per their titles
# -problem with the codes. . . opted for a simpler option for now- fill in the missing values with the mean
# 
# 

# In[26]:


traind.describe()


# In[27]:


testd.describe()


# In[28]:


#replace the missing values in the Age with the age means train and test respectively
traind = traind.fillna({"Age": 29})
testd = testd.fillna({"Age": 30})


# In[29]:


#replace the missing values in the test (fare) with the fare mean 
testd = testd.fillna({"Fare": 35})
testd.info()


# In[30]:


#Map age into 5 numeric groups
traind['Agegroup']= pd.qcut(traind['Age'], 5, labels = [1, 2, 3, 4, 5])
testd['Agegroup']= pd.qcut(testd['Age'], 5, labels = [1, 2, 3, 4, 5] )                      


# In[31]:


#map fare into 5 numeric groups
traind['Faregroup'] = pd.qcut(traind['Fare'], 5, labels = [1, 2, 3, 4, 5])
testd['Faregroup'] = pd.qcut(testd['Fare'], 5, labels = [1, 2, 3, 4, 5])


# In[32]:


traind = traind.drop(['Age', 'Fare', 'Name' ], axis=1)
testd = testd.drop(['Age', 'Fare', 'Name' ], axis=1)


# In[33]:


traind.head()


# In[34]:


testd.head()


# **Machine Learning**
# Will test 
# 1. Decision Tree
# 2. Random forest

# In[35]:



from sklearn.model_selection import train_test_split

# identify the features (x) and the target (y)
X = traind.drop(['Survived'], axis=1)
y = traind['Survived']

# randomly split the training data to test the model 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)


# In[40]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
print(round(acc_decision_tree,2,), "%")



# In[42]:


# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
print(round(acc_random_forest,2,), "%")



# **Submission**
# Use random forest to predict the test data (82.68%)

# In[43]:


#view the test data
testd.head()


# In[44]:


# predict with random forest
submission = pd.DataFrame({
    "PassengerId" : testd['PassengerId'],
    "Survived" : random_forest.predict(testd.drop('PassengerId', axis=1))
})


# In[52]:


submission.head(8)


# In[50]:


submission.info()


# In[53]:


submission.to_csv('submission.csv', index=False)

