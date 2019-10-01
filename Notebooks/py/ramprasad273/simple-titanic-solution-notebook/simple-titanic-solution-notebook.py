#!/usr/bin/env python
# coding: utf-8

# #  Kaggle - Titanic Solution Notebook

# ### Competition Description
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.
# 
# ### Practice Skills
# 
# * Binary classification
# 
# * Python and R basics
# 

# ### Collecting the Data
# 
# The training and test data are available on Kaggle.
# 
# You can download directly from here [here](https://www.kaggle.com/c/titanic/data) 
# 
# 
# ### Data Dictionary
# 
#  
# | Variable                | Definition                                    |  Key                                            |
# |:------------------------|----------------------------------------------:|------------------------------------------------:|
# | survival                | Survival                                      |  0 = No, 1 = Yes                                |   
# | pclass                  | Ticket class                                  |  1 = 1st, 2 = 2nd, 3 = 3rd                      |
# | sex                     | Sex                                           |                                                 |
# | Age                     | Age in years                                  |                                                 |
# | sibsp                   | # of siblings / spouses aboard the Titanic    |                                                 |
# | parch                   | # of parents / children aboard the Titanic    |                                                 |
# | ticket                  | Ticket number                                 |  0 = No, 1 = Yes                                |
# | fare                    | Passenger fare	                              |  0 = No, 1 = Yes                                |
# | cabin                   | Cabin number                                  |  0 = No, 1 = Yes                                |
# | embarked                | Port of Embarkation                           |  C = Cherbourg, Q = Queenstown, S = Southampton |   
# |-------------------------|-----------------------------------------------|-------------------------------------------------|
# 
# ### Variable Notes
# 
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.
# 

# In[ ]:


#############################################################
#        Step 1: Import libraries                           #
#############################################################

 
import pandas as pd
import numpy as np

# Visual representation libraries
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Machine learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:


#############################################################
#        Step 2: load the datasets                          #
#############################################################

print('Load the datasets...')

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print ('Train dataset: %s'%(str(train.shape)))
print ('Test  dataset: %s'%(str(test.shape)))


train.head()
test.head()


# After loadin the train and test datasets, we can get the information about the total number of rows and coulmns. 
# 
#   #### Training Dataset 
#     Columns: 12
#     Rows   : 981
# 
#   #### Test Dataset
#     Columns: 11
#     Rows   : 418
# 
# 
# 

# In[ ]:


# Combining train and test dataset
train_test_data = [train, test] 


# In[ ]:


#############################################################
#       Step 3: Data Exploration                            #
#############################################################

# describe() method shows values like count, mean, standard deviation, etc. of numeric data types.
train.describe()

# describe(include = ['O']) will show the descriptive statistics of object data types. 
train.describe(include=['O'])


# Get the total missing values present in columns of both train and test dataset
train.isnull().sum()
test.isnull().sum()

# Survival rate from train data
survived = train[train['Survived']==1]
not_survived = train[train['Survived']==0]

print("---------------------------")
print("From the training dataset:")
print("---------------------------")
print("  Total Passengers : %i"      %(len(train)))
print("")
print("  Total Survivors  : %i"      %(len(survived)))

print("  Survival Rate    : %i %% "     % (1.*len(survived)/len(train)*100.0))
print("-------------------------")
print("  Total Casuality  : %i"      %(len(not_survived)))

print("  Fatality Rate    : %i %% "     % (1.*len(not_survived)/len(train)*100.0))
print("-------------------------")

# Check for missing data & list them 
missingData = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Training Dataset', 'Test Dataset']) 
print('')
print('Nan data present in the datasets')
print('')
print(missingData[missingData.sum(axis=1) > 0])


# #### Observations:
# 
# From the training dataset we get the below details:
# 
#   1. There were 891 passengers of which only 38% survived.
#   2. There are some missing data present in both training and test datasets.
#   3. Age column has the highest number of missing data which is present in both the datasets.
#   4. Cabin also has missing data in both the datasets.
#   5. Embarked column has 2 missing data only in the training datasets.
#   6. Fare column has one missing data in the test dataset.
# 
# 
# 
# 
# 

# For the next step we will try to visualize the data.
# 
# Visualizing the data helps us understanding the data better and this will help us in getting the dataset well prepared for 
# the Model training.
# 
# Below is the function "bar_chart" which will take in a feature as a parameter and provide us the bar chart.
# 
# The Bar chart will show how the survival of a passenger varies with different features and its subtypes.
# 

# In[ ]:


#############################################################
#       Step 4: Data Visualization                          #
#############################################################

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    not_survived = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,not_survived])
    df.index = ['Survived','Casuality']
    df.plot(kind='bar',stacked=True, figsize=(7,5))



# In[ ]:


bar_chart('Sex')


# In[ ]:


bar_chart('Pclass')


# In[ ]:


bar_chart('SibSp')


# In[ ]:


bar_chart('Parch')


# In[ ]:


bar_chart('Embarked')


# The Above bar chart just tells us how the survival varies with each features.
# Now in the below section we are going to see how each features impacts the survial of a passenger. 

# In[ ]:


#############################################################
#      Step 5: Relationship Features and Survival           #
#############################################################



# #### Pclass vs. Survival

# In[ ]:


#Pclass vs. Survival

train.Pclass.value_counts()
train.groupby('Pclass').Survived.value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x='Pclass', y='Survived', data=train)


# In[ ]:


# Class vs Survived
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                                   ascending=False))


# #### Observations:
#     1. The passengers in the 1st class have the highest survival rate of 63%.
#     2. The passengers in the 2nd ticket class have survival rate of almost 47%.
#     3. The 3rd class passengers have the lowest survival percentage which is 24%.

# #### Sex vs. Survival

# In[ ]:


#Sex vs. Survival
train.Sex.value_counts()
train.groupby('Sex').Survived.value_counts()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
sns.barplot(x='Sex', y='Survived', data=train)


# In[ ]:


# Class vs Survived
print(train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',
                                                                                                   ascending=False))


# #### Observations
#     1. Female survival percentage is more than that of male.
#     2. 74% of the surviors were female and only 19% men survived.

# #### Pclass & Sex vs. Survival
# Lets check the survival rate when compared to the pclass and sex

# In[ ]:


#Pclass & Sex vs. Survival
tab = pd.crosstab(train['Pclass'], train['Sex'])
print (tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Pclass')
plt.ylabel('Percentage')

sns.factorplot('Sex', 'Survived', hue='Pclass', size=5, aspect=2, data=train)


# #### Observations:
#     1. From the 1st class passengers, 122 male and 94 female survived.
#     2. 76 female and 108 male from the 2nd class survived.
#     3. 144 female and 347 male passengers from the 3rd class survived.  
#     4. 1st class passenger survival rate is higher than other classes.
#     5. 3rd class passenger survival rate is lowest when compared to other classes.
#     

# In[ ]:


#Pclass, Sex & Embarked vs. Survival
sns.factorplot(x='Pclass', y='Survived', hue='Sex', col='Embarked', data=train)


# #### Embarked vs. Survived

# In[ ]:


#Embarked vs. Survived
train.Embarked.value_counts()
train.groupby('Embarked').Survived.value_counts()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=train)

print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                   ascending=False))


# #### Observations
#     1.The passengers embarked from Cherbourg has the highest survival percentage which is 55.
#     2.The passengers embarked from Queenstown has the survival percentage of 39.
#     3.The passengers embarked from Southampton has the lowest survival percentage of 34.
#     

# #### SibSp vs. Survival

# In[ ]:


#SibSp vs. Survival
train.SibSp.value_counts()
train.groupby('SibSp').Survived.value_counts()
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
sns.barplot(x='SibSp', y='Survived', ci=None, data=train) # ci=None will hide the error bar

print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))



# #### Observations:
#     1. The survival rate of a passenger is high when he/she has a siblings / spouse onboard the titanic.
#     2. When there are no siblings / spouse, the survival rate is 34%.
#     3. When there is 1 siblings / spouse onboard the survival rate is highest which is 53%.
#     4. When there are 2 siblings / spouse onboard the survival rate is 46%.
#     5. Survival rate is low when there are more than 2 siblings / spouse.   
#     

# #### Parch vs. Survival

# In[ ]:


#Parch vs. Survival
train.Parch.value_counts()
train.groupby('Parch').Survived.value_counts()
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
sns.barplot(x='Parch', y='Survived', ci=None, data=train)

print(train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# #### Observations
#     1. When there are 5 parents / children members onboard the Titanic the survival rate is very low i.e. 20%.    
#     2. The survival rate is highest (60%) when there are 3 parents / children members.
#     3. The rate dips when the number of parents / children increases.
#     4. When there is no parents / children onboard the survival rate is 34%
#     

# In[ ]:


#Correlating Features
plt.figure(figsize=(15,6))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


# #### Observations
#    1. Correlation index of 1 means perfect correlation and -1 means anti-correlation.
#    2. Positive or negative correlations with the Survived feature are valuable.
#    3. Strong correlations between two other features would suggest that only one of them is necessary for our model.

# In[ ]:


#############################################################
#      Step 5: Feature engineering                          #
#############################################################


# ### Featur Engineering
# 
# #### 1. Name
#      
#      * We create Title from the name feature.
#      * Then we map the title to numeric values
#      * We create a new mapping others which will have the other titles like 'Don', 'Dr', 'Major', 'Rev', 'Sir',etc
#      * So there are in total 5 groupings made from the title

# In[ ]:


#Name

for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()

test['Title'].value_counts()


for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
bar_chart('Title')


# #### 2. Sex
#      
#      *  We map male as 0 and female as 1

# In[ ]:


#Sex
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

bar_chart('Sex')



# #### 3. Age
# 
#      * For the first step we fill the missing value in age field by the median age grouped by title.
#      * We then create a age bands like 0-15  as 0,
#                                        16-32 as 1,
#                                        33-48 as 2,
#                                        49-63 as 3 and 
#                                        64 and above as 4

# In[ ]:


#Age

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

for dataset in train_test_data:    
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 49) & (dataset['Age'] <= 63), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 63, 'Age'] = 4


# #### 4.Embarked
# 
#      * For embark field we map fill the missing values with S (mode)
#      * We then map the ports with numeric values S-->0,C-->1 and Q-->2

# In[ ]:


#Embarked


for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

embarked_mapping = {"S": 0, "C": 1, "Q": 2}

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)
    






# #### 5.Fare
# 
#      * We fill in the missing values in fare band with the median values grouped by the pclass features.
#      * The reason is simple the fare value depends on the Pclass ticket.
#      * We then create a fare band and then we group them

# In[ ]:


#Fare

train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

for dataset in train_test_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
train['FareBand'] = pd.qcut(train['Fare'], 4)

print (train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean())

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


# #### 6. Cabin
#          * Fill the missing values by taking the median value grouped by Pclass and cabin
#          * We then create a cabin mapping and group them together

# In[ ]:


#Cabin

train.Cabin.value_counts()
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts() 
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3]) 
df.index = ['1st class','2nd class', '3rd class'] 
df.plot(kind='bar',stacked=True, figsize=(10,5))
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)




# #### 7. Family size

# In[ ]:


#Family size
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


#8.SibSp & Parch Feature
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1

print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())

print("---------------------------------------------")

for dataset in train_test_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())





# #### 8. Pclass

# In[ ]:


#Pclass

Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))


# In[ ]:


#############################################################
#      Step 6: Feature Selection                            #
#############################################################


# In[ ]:


# delete unnecessary feature from dataset

print("Before", train.shape, test.shape, train_test_data[0].shape, train_test_data[1].shape)
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'FamilySize']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId', 'FareBand'], axis=1)
test_pred  = test.drop("PassengerId", axis=1).copy()

print("After", train.shape, test.shape, train_test_data[0].shape, train_test_data[1].shape)

train_data = train.drop('Survived', axis=1)
target = train['Survived']



# In[ ]:


train_data.shape, target.shape, test_pred.shape


# In[ ]:


#############################################################
#      Step 7: Applying Different Models                    #
#############################################################


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train_data, target)
Y_pred = logreg.predict(test_pred)
logistic_acc = round(logreg.score(train_data, target) * 100, 2)
logistic_acc

coeff_df = pd.DataFrame(train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# Support Vector Machines

svc = SVC()
svc.fit(train_data, target)
Y_pred = svc.predict(test_pred)
svc_acc = round(svc.score(train_data, target) * 100, 2)
svc_acc

# Knn
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_data, target)
Y_pred = knn.predict(test_pred)
knn_acc = round(knn.score(train_data, target) * 100, 2)
knn_acc

# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(train_data, target)
Y_pred = gaussian.predict(test_pred)
gaussian_acc = round(gaussian.score(train_data, target) * 100, 2)
gaussian_acc

# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data, target)
Y_pred = decision_tree.predict(test_pred)
decision_tree_acc = round(decision_tree.score(train_data, target) * 100, 2)
decision_tree_acc

# Random Forest
random_forest = RandomForestClassifier(n_estimators=250)
random_forest.fit(train_data, target)
Y_pred = random_forest.predict(test_pred)
random_forest.score(train_data, target)
random_forest_acc = round(random_forest.score(train_data, target) * 100, 2)
random_forest_acc




# In[ ]:


from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth=1000, n_estimators=1500, learning_rate=0.09)
classifier.fit(train_data,target)
Y_pred = classifier.predict(test_pred)
classifier.score(train_data, target)
acc_xgb = round(classifier.score(train_data, target) * 100, 2)
acc_xgb


# In[ ]:


#############################################################
#      Step 8: Model Selection                              #
#############################################################


# In[ ]:


models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines','KNN',
              'Naive Bayes','Decision Tree','Random Forest','XGBoost'],
    'Score': [ logistic_acc, svc_acc,knn_acc,gaussian_acc
              ,decision_tree_acc,random_forest_acc,acc_xgb ]})
models.sort_values(by='Score')


# In[ ]:


#############################################################
#      Step 8: Result Submission                            #
#############################################################


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')


# The accuracy for random forest is high as per the table. Hence we submit the result 
# of the model applied to Kaggle.

# In[ ]:




