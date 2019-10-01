#!/usr/bin/env python
# coding: utf-8

# # 183 DADM Topic 8
# # Titanic (Kaggle)
# Author：Hiroki Miyamoto

# I applied the Decision Tree algorithm to Titanic data on Kaggle based on the topic 8 of 183 DADM.
# 
# I used "Titanic Data Science Solutions" as a reference.
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

# ## Import libraries

# In[ ]:


# Data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support


# ## Acquire Data

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# ## Preview the data

# - Variables
#     - A total of 12 variables is recorded.
#     - According to page 11 of "GMBA71-202_Topic-8b - Classification.pdf", there are only 11 variables. 
#     - The reason for the difference is because 'PassengerId' is not included in the lecture.
# - Train and Test data
#     - Train 891 rows (981 rows in the lecture)
#     - Test 418 rows (327 rows in the lecture)
#     - There are a total of 1309 passengers. This is the same as the lecture.
#     - According to page 14 of "GMBA71-202_Topic-8b - Classification.pdf", the train/test data split ratio is different from the data acquired here.
#     - Therefore, the result of the classifier performance here will not be the same as of the lecture.
# 

# In[ ]:


train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# ## Visualization

# The figure below shows the same figure as page 17 of "GMBA71-202_Topic-8b - Classification.pdf".

# In[ ]:


pd.crosstab(train_df['Pclass'], train_df['Survived']).plot(kind='bar',stacked=True);


# The figure below shows the same figure as page 18 of "GMBA71-202_Topic-8b - Classification.pdf".

# In[ ]:


pd.crosstab(train_df['Sex'], train_df['Survived']).plot(kind='bar',stacked=True);


# The figure below shows the same figure as page 19 of "GMBA71-202_Topic-8b - Classification.pdf".

# In[ ]:


pd.crosstab(train_df['SibSp'], train_df['Survived']).plot(kind='bar',stacked=True);


# The figure below doesn't show the same figure as the page 20 of "GMBA71-202_Topic-8b - Classification.pdf" because I'm not sure how to create the same graph. However, this graph is equivalent to the graph of the lecture.

# In[ ]:


sns.catplot(x="Pclass", y="Survived", hue="Sex",
            palette={"male": "g", "female": "m"},
            markers=["^", "o"], linestyles=["-", "--"],
            kind="point", data=train_df);


# ## Select explanatory variables

# It was not mentioned in the lecture what variables are used as explanatory features.
# 
# I selected the following variables because it seems these variables are used according to page 9 of "GMBA71-202_Topic-8d - Decision Trees.pdf".
# 
# - Objective variable
#     - Survived (1: Survived, 0: Dead)
# - Explanatory variables
#     - Pclass
#     - Sex
#     - SibSp
#     - Parch
#     - Embarked

# In[ ]:


train_df.head()


# In[ ]:


print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1)
test_df = test_df.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin'], axis=1)
combine = [train_df, test_df]

"After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape


# In[ ]:


train_df.head()


# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# In[ ]:


train_df.describe()


# ## Completing a categorical feature

# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

# In[ ]:


train_df.describe(include=['O'])


# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# In[ ]:


test_df.head(10)


# In[ ]:


X_train = train_df[["Pclass", "Sex", "SibSp", "Parch", "Embarked"]]
Y_train = train_df["Survived"]
X_test  = test_df[["Pclass", "Sex", "SibSp", "Parch", "Embarked"]].copy()
X_train.shape, Y_train.shape, X_test.shape


# ## Apply decision tree model to the train data

# ID3 algorithm is applied following the lecture page 12 of "GMBA71-202_Topic-8d - Decision Trees.pdf".
# 
# This gives us an over all accuracy of 83.73%. (77.8% in the lecture)
# 
# - The difference of the accuracy comes from the difference of
#     - Train/Test data split ratio
#     - Selection of explanatory variables (I'm not sure what variables are applied in the lecture)
#     - A number of max leaf nodes of decision tree
#     - Random number
#     

# In[ ]:


# Decision Tree
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                           random_state=1234)
# decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
#                                            random_state=1234,
#                                            max_leaf_nodes=5)
decision_tree.fit(X_train, Y_train)
# Y_pred = decision_tree.predict(X_test)
Y_pred = decision_tree.predict(X_train)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# The confusion matrix below is equivalent to page 15 of "GMBA71-202_Topic-8d - Decision Trees.pdf".

# In[ ]:


# Confusion matrix
conf_mat = confusion_matrix(Y_pred, Y_train)
conf_mat = pd.DataFrame(conf_mat, 
                        index=['Predicted = dead', 'Predicted = survived'], 
                        columns=['Actual = dead', 'Actual = survived'])
conf_mat


# The figure below shows the decision tree.

# In[ ]:



import graphviz 
clf = tree.DecisionTreeClassifier()
dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                feature_names=["Pclass", "Sex", "SibSp", "Parch", "Embarked"], 
                                class_names=['Dead','Suevived'],
                                filled=True, rounded=True, 
                                special_characters=True) 
graph = graphviz.Source(dot_data)
graph


# ## Prediction against the test data

# Now, it's time to test the classifier on the unseen test data.
# 
# I submitted the prediction and scored 0.75598.
# 
# In other words, our overall accuracy on the unseen test data is 75.6%. (78.6% in the lecture)
# 
# Therefore, I could get the result which doesn't look very different from the lecture.

# In[ ]:


Y_pred = decision_tree.predict(X_test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)

