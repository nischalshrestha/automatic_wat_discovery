#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


# Import modules
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split

# Figures inline and set visualization style
get_ipython().magic(u'matplotlib inline')
sns.set()


# In[ ]:


os.listdir('../input')


# ## Read Data

# In[ ]:


# Import data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


print(df_train.shape)
df_train.head()


# In[ ]:


print(df_test.shape)
df_test.head()


# ## Basic EDA

# In[ ]:


sns.countplot(x='Survived', data=df_train);


# In[ ]:


sns.countplot(x='Sex', data=df_train);


# In[ ]:


sns.factorplot(x='Survived', col='Sex', kind='count', data=df_train);


# Women were more likely to survive than men.

# In[ ]:


df_train.groupby(['Sex']).Survived.sum()


# proportion of women that survived and the proportion of men

# In[ ]:


print(df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print(df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# 74% of women survived, while 19% of men survived.

# In[ ]:


sns.factorplot(x='Survived', col='Pclass', kind='count', data=df_train);


# Passengers that travelled in first class were more likely to survive. On the other hand, passengers travelling in third class were more unlikely to survive.

# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# Passengers that embarked in Southampton were less likely to survive

# In[ ]:


sns.distplot(df_train.Fare, kde=False);


# Most passengers paid less than 100 for travelling with the Titanic.

# In[ ]:


df_train_drop = df_train.dropna()
sns.pairplot(df_train_drop, hue='Survived');


# ## Data Wrangling

# In[ ]:


# Store target variable of training data in a safe place
survived_train = df_train.Survived

# Concatenate training and test sets
data = pd.concat([df_train.drop(['Survived'], axis=1), df_test])


# In[ ]:


data.info()


# In[ ]:


# Dealing with missing numerical variables
data['Age'] = data.Age.fillna(data.Age.median())
data['Fare'] = data.Fare.fillna(data.Fare.median())

# Check out info of data
data.info()


# In[ ]:


# Tranform Sex feature to numeric value
# create a new column for each of the options in 'Sex'
# creates a new column for female, called 'Sex_female', 
# creates a new column for 'Sex_male'
# more then two categorical values it is better to use one-hot-encode
data = pd.get_dummies(data, columns=['Sex'], drop_first=True)
data.head()


# In[ ]:


# Select features columns
data = data[['Sex_male', 'Fare', 'Age','Pclass', 'SibSp']]
data.head()


# In[ ]:


data.info()


# In[ ]:


# split it back into training and test sets
data_train = data.iloc[:891]
data_test = data.iloc[891:]


# In[ ]:


# scikit-learn requires the data as arrays
X = data_train.values
test = data_test.values
y = survived_train.values


# In[ ]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)


# In[ ]:


import graphviz 
#dot_data = tree.export_graphviz(clf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph.render("Titanic") 

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=data_train.columns.values,  
                         class_names=['Survived','Not Survived'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
clf.score(X, y)


# In[ ]:


df_test[['PassengerId', 'Survived']].to_csv('../working/dec_tree.csv', index=False)


# ## Second try

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# In[ ]:


# Setup arrays to store train and test accuracies
dep = np.arange(1, 9)
train_accuracy = np.empty(len(dep))
test_accuracy = np.empty(len(dep))


# In[ ]:


# Loop over different values of k
for i, k in enumerate(dep):
    # Setup a Decision Tree Classifier
    clf = tree.DecisionTreeClassifier(max_depth=k)

    # Fit the classifier to the training data
    clf.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = clf.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = clf.score(X_test, y_test)


# In[ ]:


# Generate plot
plt.title('clf: Varying depth of tree')
plt.plot(dep, test_accuracy, label = 'Testing Accuracy')
plt.plot(dep, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Depth of tree')
plt.ylabel('Accuracy')
plt.show()


# In[ ]:


clf = tree.DecisionTreeClassifier(max_depth=6)
clf.fit(X, y)
clf.score(X, y)


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=data_train.columns.values,  
                         class_names=['Survived','Not Survived'],  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# In[ ]:


# Make predictions and store in 'Survived' column of df_test
Y_pred = clf.predict(test)
df_test['Survived'] = Y_pred
df_test[['PassengerId', 'Survived']].to_csv('../working/6dep_dec_tree.csv', index=False)


# ## Feature Engineering

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df = df_train.append(df_test, sort=False)
df.info()


# In[ ]:


df['Surname'] = df['Name'].str.split(',').str[0]
df['Surname'].head()


# In[ ]:


df['Title'] = df['Name'].str.split(',').str[1].str.split().str[0]  
df['Title'].head()


# In[ ]:


#df['Cabin'][27:32]


# In[ ]:


#df['Cabin Len'] = df.Cabin.str.split().str.len()
#df['Cabin Len'][27:32]


# In[ ]:


df['Cabin Letter'] = df['Cabin'].str[0]
df['Cabin Letter'][27:32]


# the size of a person’s family, the sum of their ‘SibSp’ and ‘Parch’ attributes

# In[ ]:


df['Family_Size'] = df['SibSp'] + df['Parch']
df['Family_Size'].head()


# Persons from same family share tickets, therefore the same Fare

# In[ ]:


df[df['Name'].str.contains('Andersson,')]


# In[ ]:


df['Fare Per Person'] = df['Fare'] / (df['Family_Size'] + 1)
df['Fare Per Person'].head()


# In[ ]:


df['Number of Ticket Uses'] = df.groupby('Ticket', as_index=False)['Ticket'].transform(lambda s: s.count())
df['Number of Ticket Uses'].head()


# In[ ]:


df['Average Fare per Person'] = df['Fare'] / df['Number of Ticket Uses'] 
df['Average Fare per Person'].head()


# ## Data Wrangling

# transform text columns to numbers

# In[ ]:


for col in df.columns:  
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category')  # change text to category
        df[col] = df[col].cat.codes  # save code as column value


# In[ ]:


df['Age'] = df.Age.fillna(df.Age.median())
df['Fare'] = df.Fare.fillna(df.Fare.median())
df['Fare Per Person'] = df['Fare Per Person'].fillna(df['Fare Per Person'].median())
df['Average Fare per Person'] = df['Average Fare per Person'].fillna(df['Average Fare per Person'].median())


# In[ ]:


# RandomForest/Decision Tree it is interesting to replace NA by a value less then the minimum or greater then the maximum
#df.fillna(-1, inplace=True)


# In[ ]:


data_train = df.iloc[:891].copy()
data_test = df.iloc[891:].copy()


# In[ ]:


train, test = train_test_split(data_train, test_size=0.2, random_state=42)


# In[ ]:


# Instantiate model and fit to data
clf = tree.DecisionTreeClassifier()


# In[ ]:


remove = ['Survived', 'PassengerId', 'Name', 'Cabin', 'Embarked']
feats = [col for col in df.columns if col not in remove]


# In[ ]:


clf.fit(train[feats], train['Survived'])


# In[ ]:


preds_train = clf.predict(train[feats])


# In[ ]:


preds = clf.predict(test[feats])


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(train['Survived'], preds_train)


# In[ ]:


accuracy_score(test['Survived'], preds)


# In[ ]:


# train with training and test dataset
clf.fit(data_train[feats],data_train['Survived'])


# In[ ]:


preds_kaggle = clf.predict(data_test[feats])


# In[ ]:


submission = pd.DataFrame({ 'PassengerId': data_test['PassengerId'],
                            'Survived': preds_kaggle }, dtype=int)
submission.to_csv("submission_FEAT.csv",index=False)


# In[ ]:




