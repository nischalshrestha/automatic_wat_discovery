#!/usr/bin/env python
# coding: utf-8

# #### KAGGLE Competition

# # Titanic: Machine Learning from Disaster

# [Competition Link : https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
# 
# ## Competition Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## Goal
# 
# It is your job to predict if a passenger survived the sinking of the Titanic or not. 
# For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
# 
# ## Metric
# 
# Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
# 
# ## Submission File Format
# 
# You should submit a csv file with exactly 418 entries plus a header row. Your submission will show an error if you have extra columns (beyond PassengerId and Survived) or rows.
# 
# The file should have exactly 2 columns:
# 
# 1. PassengerId (sorted in any order)
# 2. Survived (contains your binary predictions: 1 for survived, 0 for deceased)
#  
# PassengerId|Survived
# ---|---
# 892|0
# 893|1
# 894|0
# Etc|...

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic(u'matplotlib inline')

#Reading input
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


df_train['df_type'] = 'Train'
df_test['df_type'] = 'Test'

#Combining train and test data to perform data structure manipulation
df_combined = pd.concat([df_train,df_test])

df_combined.info()


# ### Data Understanding and Preparation

# In[ ]:


df_combined.describe(include='all')


# In[ ]:


list(df_combined.Embarked.unique())


# In[ ]:


# Add the dummy columns for Embarked to the dataframe
df_combined = pd.concat([df_combined,pd.get_dummies(df_combined.Embarked)],axis=1) 


# In[ ]:


# Drop the variable Embarked as dummies created
df_combined = df_combined.drop('Embarked',axis=1)


# In[ ]:


# Creating variable with total number of family members onboard
df_combined['Fam_Mem_Onboard'] = df_combined.Parch + df_combined.SibSp + 1


# In[ ]:


# Changing gender to numeric : Male = 1, Female = 0
df_combined['Sex'] = df_combined.Sex.map({'male':1,'female':0})


# In[ ]:


list(df_combined.Pclass.unique())


# In[ ]:


# Calculate total passengers with same ticket
df_combined['Total_Passengers'] = df_combined.apply(lambda x: (df_combined['Ticket'] == x['Ticket']).sum() , axis=1)


# In[ ]:


# Drop Ticket column
df_combined = df_combined.drop('Ticket',axis=1)


# In[ ]:


# Checking if passenger was travelling alone (by number of family members and number of people with same ticket)
df_combined['IsAlone'] = df_combined.apply(lambda x: 1 if ((x['Total_Passengers'] == 1) and (x['Fam_Mem_Onboard'] == 1)) else 0,axis=1)


# In[ ]:


# extracting titles from names
df_combined['Title'] = df_combined.Name.str.extract('([A-Za-z]+\.)', expand=False)


# In[ ]:


# Checking for unique Titles
pd.DataFrame(df_combined.Title.unique())


# In[ ]:


# Let's see if we can replace titles with more common terms or mark them as rare
pd.crosstab(df_combined.Title,df_combined.Sex)


# In[ ]:


# Replacing Title with common terms

df_combined['Title'] = df_combined['Title'].replace(['Lady.', 'Countess.','Capt.', 'Col.','Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'],'Rare')

df_combined['Title'] = df_combined['Title'].replace('Mlle.', 'Miss.')
df_combined['Title'] = df_combined['Title'].replace('Ms.', 'Miss.')
df_combined['Title'] = df_combined['Title'].replace('Mme.', 'Mrs.')

pd.crosstab(df_combined.Title,df_combined.Sex)


# In[ ]:


# Dummy variables for titles
df_combined = pd.concat([df_combined,pd.get_dummies(df_combined['Title'])],axis=1)


# In[ ]:


# Dropping Title Column
df_combined = df_combined.drop('Title',axis=1)


# In[ ]:


# dropping Name column
df_combined = df_combined.drop('Name',axis=1)


# In[ ]:


df_combined.head(20)


# In[ ]:


# Separating training and testing data

train = df_combined[df_combined.df_type == 'Train']
train = train.drop('df_type',axis=1)
train.info()


# In[ ]:


test = df_combined[df_combined.df_type == 'Test']
test = test.drop('df_type',axis=1)
test.info()


# In[ ]:


train.describe(include='all')


# In[ ]:


# Dropping Cabin column as it serves no purpose in the analysis
train = train.drop('Cabin',axis=1)
test = test.drop('Cabin',axis=1)


# In[ ]:


# Above we can see that Age has lesser entries denoting NA values
# Filling NA with median for Age
ageMedian = train['Age'].median()
train.fillna(value={'Age':ageMedian}, inplace=True)

ageMedian = test['Age'].median()
test.fillna(value={'Age':ageMedian}, inplace=True)


# In[ ]:


FareMedian = test['Fare'].median()
test.fillna(value={'Fare':FareMedian}, inplace=True)


# ### Exploratory Data Analysis

# In[ ]:


sns.barplot(x='IsAlone',y='Survived',data=train)

# Those who were alone had a lower chance of survival


# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=train)

#This shows an important trend that people with higher class travel had better chance of survival 


# In[ ]:


plt.figure(figsize=[18,7])

plt.subplot(1,2,1)
plt1 = sns.barplot(x='Fam_Mem_Onboard',y='Survived',data=train)

plt.subplot(1,2,2)
plt2 = sns.barplot(x='Total_Passengers',y='Survived',data=train)

plt.show()

# Those who were travelling with less than 4 people onboard had a better chance of survival


# In[ ]:


sns.barplot(x='Sex',y='Survived',data=train)

# Females had a better chance of survival


# In[ ]:


# Percentage of people survived based on Embarkment

for emb in ['C','Q','S']:
    print((emb,round((train[train['Survived'] == 1][emb].sum())/(len(train[train['Survived'] == 1]))*100,2)))


# In[ ]:


sns.countplot(x='Survived',data=train)

# The data is not very unbalanced, hence not performing data balancing


# ### Modelling

# In[ ]:


X_train = train.drop(['PassengerId','Survived'],axis=1)
y_train = train[['PassengerId','Survived']]

X_test = test.drop(['PassengerId','Survived'],axis=1)


# #### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression

lr_1 = LogisticRegression()
model_lr_1 = lr_1.fit(X_train,y_train['Survived'])
prob_lr_1 = model_lr_1.predict_proba(X_train)[:,1]


# In[ ]:


from sklearn import metrics

pred_table_lr_1 = y_train
pred_table_lr_1.is_copy = False
pred_table_lr_1['Prob'] = prob_lr_1
pred_table_lr_1['Pred'] = pred_table_lr_1['Prob'].map(lambda x: 1 if x > 0.5 else 0)
metrics.accuracy_score(pred_table_lr_1.Survived, pred_table_lr_1.Pred)

# We get an accuracy of 82.9% 


# In[ ]:


prob_lr_1_test = model_lr_1.predict_proba(X_test)[:,1]

pred_table_lr_1_test = pd.DataFrame(test['PassengerId'])
pred_table_lr_1_test.is_copy = False
pred_table_lr_1_test['Prob'] = prob_lr_1_test
pred_table_lr_1_test['Pred'] = pred_table_lr_1_test['Prob'].map(lambda x: 1 if x > 0.5 else 0)


# In[ ]:


pred_table_lr_1_test = pred_table_lr_1_test.drop('Prob',axis=1)


# In[ ]:


pred_table_lr_1_test = pred_table_lr_1_test.rename(columns={'Pred':'Survived'})


# ### My First Submission (Logistic Regression)
# #### Not so perfect

# In[ ]:


# pred_table_lr_1_test.to_csv('Logistic_1_solution.csv',index=False)


# ### Continuing with other models

# #### Choosing a better cutoff for Logistic Regression

# In[ ]:


# Creating columns with different probability cutoffs 
y_pred_all_1 = pred_table_lr_1
y_pred_all_1.is_copy = False

numbers = [float(x)/10 for x in range(10)]
numbers

for i in numbers:
    y_pred_all_1[i]= pred_table_lr_1.Prob.map( lambda x: 1 if x > i else 0)
y_pred_all_1.head()

#Checking Accuracy, Sensitivity and Specificity at different probability value

cutoff_df = pd.DataFrame( columns = ['probability','accuracy','sensitivity','specificity'])
for i in numbers:
    cm1 = metrics.confusion_matrix(pred_table_lr_1.Survived, pred_table_lr_1[i] )
    total1 = sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensitivity,specificity]
print(cutoff_df)


# In[ ]:


#Plotting the chart
cutoff_df.plot.line(x='probability', y=['accuracy','sensitivity','specificity'])


# In[ ]:


# Lets choose 0.35 as the cutoff value to get a better score and check

pred_table_lr_2_test = pd.DataFrame(test['PassengerId'])
pred_table_lr_2_test.is_copy = False
pred_table_lr_2_test['Prob'] = prob_lr_1_test
pred_table_lr_2_test['Pred'] = pred_table_lr_2_test['Prob'].map(lambda x: 1 if x > 0.35 else 0)


# In[ ]:


# My 2nd Submission

pred_table_lr_2_test = pred_table_lr_2_test.drop('Prob',axis=1)
pred_table_lr_2_test = pred_table_lr_2_test.rename(columns={'Pred':'Survived'})
# pred_table_lr_2_test.to_csv('Logistic_2_solution.csv',index=False)


# ### Cross Validation : k fold

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10,shuffle=True, random_state=111)


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=15)
score = cross_val_score(clf,X_train,y_train['Survived'],cv=k_fold, scoring='accuracy')
print(round(np.mean(score)*100,2))


# ### Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
score = cross_val_score(clf,X_train,y_train['Survived'],cv=k_fold, scoring='accuracy')
print(round(np.mean(score)*100,2))


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=15)
score = cross_val_score(clf,X_train,y_train['Survived'],cv=k_fold, scoring='accuracy')
print(round(np.mean(score)*100,2))


# ### SVM

# In[ ]:


from sklearn.svm import SVC

clf = SVC()
score = cross_val_score(clf,X_train,y_train['Survived'],cv=k_fold, scoring='accuracy')
print(round(np.mean(score)*100,2))


# ### Going to use RANDOM FOREST for final model

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters = {'max_depth': range(2, 20, 4),
             'min_samples_leaf': range(20, 200, 30),
             'min_samples_split': range(20, 200, 30),
             'n_estimators': range(20,200,30),
             'max_features': range(5,16,4)}

clf = RandomForestClassifier()

#### Commenting out grid search as it takes long duration to execute. I have executed it and got the best estimators out
# grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=k_fold, n_jobs= -1)


# In[ ]:


# Fit the grid search to the data
# grid_search.fit(X_train, y_train['Survived'])


# In[ ]:


#grid_search.best_estimator_


# In[ ]:


final_clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=14, max_features=9, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=80,
            min_weight_fraction_leaf=0.0, n_estimators=20, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


# In[ ]:


final_model = final_clf.fit(X_train, y_train['Survived'])


# In[ ]:


final_prediction = final_model.predict(X_test)


# In[ ]:


final_submission = pd.DataFrame(test.PassengerId)
final_submission['Survived'] = np.array(final_prediction,dtype=int)
# final_submission.to_csv('Random_Forest_1.csv',index=False)


# ### Trying Yet Another Random Forest with more n_estimators
# 

# In[ ]:


final_clf_2 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=14, max_features=9, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=20, min_samples_split=80,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)


# In[ ]:


final_model_2 = final_clf_2.fit(X_train, y_train['Survived'])
final_prediction_2 = final_model_2.predict(X_test)
final_submission_2 = pd.DataFrame(test.PassengerId)
final_submission_2['Survived'] = np.array(final_prediction_2,dtype=int)
#final_submission_2.to_csv('Random_Forest_2.csv',index=False)

