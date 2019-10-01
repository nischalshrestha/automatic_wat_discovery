#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing all the modules we are going to use
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# In[ ]:


# Import the csv and creating a DF
df_train = pd.read_csv('../input/train.csv')


# In[ ]:


# With this heatmap we can see all the missing values in our DF at a glance.
sns.heatmap(df_train.isnull(), cbar = False, cmap = 'viridis')


# In[ ]:


# Creating a empty column so that we can apply later a function.
# We are going to use a for loop. The DF is not very big, but we should avoid
# For loops and use instead apply and map functions.
df_train['Filled Age'] = ""


# In[ ]:


# We are going to use the age values in order to full fill the column "Filled Age" in 
# our next cell.
df_train[df_train['Sex'] == 'female']['Age'].mean()
df_train[df_train['Sex'] == 'male']['Age'].mean()
df_train.groupby(by = ['Pclass', 'Sex']).mean()


# In[ ]:


# Creating our Filled Age column.
# Try to avoid for loops in your code. They are slow :(
for i in range(len(df_train['Age'])):
    if (df_train['Age'].isna()[i] == True) & (df_train['Pclass'][i] == 1) & (df_train['Sex'][i] == 'female'):
        df_train['Filled Age'][i] = round(34.611765,1)
    elif (df_train['Age'].isna()[i] == True) & (df_train['Pclass'][i] == 1) & (df_train['Sex'][i] == 'male'):
        df_train['Filled Age'][i] = round(41.281386,1)
    elif (df_train['Age'].isna()[i] == True) & (df_train['Pclass'][i] == 2) & (df_train['Sex'][i] == 'female'):
        df_train['Filled Age'][i] = round(28.722973,1)
    elif (df_train['Age'].isna()[i] == True) & (df_train['Pclass'][i] == 2) & (df_train['Sex'][i] == 'male'):
        df_train['Filled Age'][i] = round(30.740707,1)
    elif (df_train['Age'].isna()[i] == True) & (df_train['Pclass'][i] == 3) & (df_train['Sex'][i] == 'female'):
        df_train['Filled Age'][i] = round(21.750000,1)
    elif (df_train['Age'].isna()[i] == True) & (df_train['Pclass'][i] == 3) & (df_train['Sex'][i] == 'male'):
        df_train['Filled Age'][i] = round(26.507589,1)
    else:
        df_train['Filled Age'][i] = df_train['Age'][i]


# In[ ]:


# We are assigning the Value 0 if the passenger is a women
# And 1 if it is a male.
df_train['Sex_male'] = df_train['Sex'].map({'female':0, 'male':1})


# In[ ]:


# Creating dummy variables from the Embarked Column.
Embarked = pd.get_dummies(df_train['Embarked'], drop_first=True)


# In[ ]:


# Concatenating the 2 df together
df_train = pd.concat([df_train, Embarked], axis = 1)


# In[ ]:


df_train.head()


# In[ ]:


df_train['Family Size'] = df_train['SibSp'] + df_train['Parch'] + 1
df_train.drop(columns=['SibSp', 'Parch'], inplace=True)


# In[ ]:


# we are going to try to find the title of each person
# using regex
def get_title(string):
    tittle_regex = re.compile(r'[aA-zZ]+\.')
    return tittle_regex.findall(string)[0]


# In[ ]:


df_train['Title'] = df_train['Name'].apply(get_title)


# In[ ]:


pd.crosstab(df_train['Title'], df_train['Sex'])


# In[ ]:


# Normalizing the titles we found previously
def replace_title(title):
    new_title = ""
    list_of_rare = ['Lady.', 'Countess.','Capt.', 'Col.','Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.']
    if title in list_of_rare:
        new_title = 'Rare'
    elif title == 'Mlle.':
        new_title = 'Miss.'
    elif title == 'Ms.':
        new_title = 'Miss.'
    elif title == 'Mne.':
        new_title == 'Mrs.'
    elif title == 'Mme.':
        new_title = 'Miss.'
    else:
        new_title = title
    return new_title


# In[ ]:


df_train['Title Normalized'] = df_train['Title'].apply(replace_title)


# In[ ]:


# Now we have much less titles and we will create some more dummy variables
pd.crosstab(df_train['Title Normalized'], df_train['Sex'])


# In[ ]:


# We transform the Titles to numbers so that our model can work with them
df_train['Title Normalized'] = df_train['Title Normalized'].map({'Master.':1, 'Miss.':2, 'Mr.':3, 'Mrs.':4, 'Rare':5})


# In[ ]:


# Creating different classes based from the fare of the ticket
def fare_range(fare):
    return_fare = None
    if fare <= 7.91:
        return_fare = 0.0
    elif fare <= 14.454:
        return_fare = 1.0
    elif fare <= 31.0:
        return_fare = 2.0
    elif fare > 31.0:
        return_fare = 3.0
    return return_fare


# In[ ]:


df_train['Fare Band'] = df_train['Fare'].apply(fare_range)


# In[ ]:


df_train.columns


# In[ ]:


df_train.head()


# In[ ]:


df_train.drop(columns = ['Name', 'Sex', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Title', "Age"], inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


sns.heatmap(df_train.isnull(), cbar = False, cmap = 'viridis')


# In[ ]:


# splitting our data for train and test
X = df_train.drop(columns = 'Survived')
y = df_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


# Initializing our model.
random_forest = RandomForestClassifier()


# In[ ]:


# Training our model
random_forest.fit(X_train, y_train)


# In[ ]:


# Predicting using our model
predict = random_forest.predict(X_test)


# In[ ]:


# The results of our model.
# It performed not very well, but still decent for my first Kernel.
print(confusion_matrix(y_test, predict))
print(classification_report(y_test,predict))


# In[ ]:


# Thank you for your time.

