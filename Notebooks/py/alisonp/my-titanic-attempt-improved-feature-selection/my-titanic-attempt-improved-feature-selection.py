#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # Improved feature selection.
# I have already decided my feature selection in my first attempt at the titanic dataset was subpar, so I have decided to try again, this time I have decided to use 3 types of feature selection in order to decide on the important features in determining the survival of a passenger in the titanic.

# In[ ]:


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


#Import the necessary data
test_df = pd.read_csv('../input/test.csv')
train_df = pd.read_csv('../input/train.csv')
combine = [train_df, test_df]


# # Cleaning the data
# Now the data has been upload, it is time to determine which features are missing values and if there is any reason to either replace missing values, or discard the feature completely.

# In[ ]:


#determine the quality of the provided data in order to determine the proceedure for data cleaning.
#missing values
#train_df.info()
print ("percentage of missing values for age: ",(1-((714/891)))*100)
print ("Percentage of missing values for cabin: ",(1-((204/891)))*100)


# The general accepted practice seems to be to fix missing values when they are below 30% missing, based on the above, this means age can be fixed, where as cabin variable really should be dropped

# In[ ]:


print ("The mode of age in train dataset: ",train_df["Age"].mode())
print ("The mode of age in test dataset: ",test_df["Age"].mode())


# As we are going to replace the missing variables, I will replace the NANs with the mode. There are plenty of resources related to this and it will be easier for you to search and come to your own conclusion as to how this is done.

# In[ ]:


train_df['Age'].fillna(24, inplace=True)
test_df['Age'].fillna(21, inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)


# In[ ]:


#remove the cabin column
train_df.drop(['Cabin','Ticket'], axis = 1, inplace = True)
test_df.drop(['Cabin','Ticket'], axis = 1, inplace = True)
combine = [train_df,test_df]


# # Creating binary variables
# It will be easier to run the models if the values for the variables are in the same format type, such as integers. Reviews of other kernals and tutorial suggest that a persons title has some bearing on their ability to survive. I always wondered if this is really the case, so I will split out the titles  and then we can remove the name as this will add no value.
# 
# Additionally, since there are many of titles out there, we will only select the most common as suggested in the tutorials out there, and here on Kaggle

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()



# I am curious to see whether there is a difference between having the age of the passengers, or grouping them into groups. This is the same with the fares. 
# 
# Below I determine the range and then create the bands. 

# In[ ]:


print('Min age: ',train_df['Age'].min())
print('Max age: ',train_df['Age'].max())
print('Min fare: ',train_df['Fare'].min())
print ('Max fare: ',train_df['Fare'].max())


# Based on the age range above, I have decided to split the ages up into 8 groups

# In[ ]:


#split the ages up into groups
train_df['AgeRange'] = pd.cut(train_df['Age'], 8)
train_df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)   


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 10.368, 'AgeBand'] = 0
    dataset.loc[(dataset['Age'] > 10.368) & (dataset['Age'] <= 20.315), 'AgeBand'] = 1
    dataset.loc[(dataset['Age'] > 20.315) & (dataset['Age'] <= 30.263), 'AgeBand'] = 2
    dataset.loc[(dataset['Age'] > 30.263) & (dataset['Age'] <= 40.21), 'AgeBand'] = 3
    dataset.loc[(dataset['Age'] > 40.21) & (dataset['Age'] <= 50.158), 'AgeBand'] = 4
    dataset.loc[(dataset['Age'] > 50.158) & (dataset['Age'] <= 60.105), 'AgeBand'] = 5
    dataset.loc[(dataset['Age'] > 60.105) & (dataset['Age'] <= 70.052), 'AgeBand'] = 6
    dataset.loc[ dataset['Age'] > 70.052, 'AgeBand']=7
    dataset['AgeBand'] = dataset['AgeBand'].astype(int)
#train_df.head()


# Fare proved to be a bit more interesting. The large range suggested there was a great deal of variance in the fares paid, but once I started splitting in to groups, I realised that the max value is actually an outlier when compared to the rest of the data and in order to not have empty groups, the maximum groups I could was 4. 
#  
#  The fare variable was graphed as you can see below and it is very obvious, the max value is an outlier. As this only relates to one record, this passenger can either be dropped, or, the fare value replace with the next highest value.
#  
#  I have decided to drop the passenger in question and that then allows me to spread out the groups which I hope in turn will provide a more accurate representation of the impact the range of fares might have.

# In[ ]:


plt.hist(train_df['Fare'], bins = 'auto')
plt.show()


# In[ ]:


train_df[train_df.Fare != 512.3292]


# In[ ]:


plt.hist(train_df['Fare'], bins = 'auto')
print ('Max fare: ',train_df['Fare'].max())
plt.show()


# In[ ]:


#now banding for fare
train_df['FareRange'] = pd.cut(train_df['Fare'], 6)
train_df[['FareRange', 'Survived']].groupby(['FareRange'], as_index=False).mean().sort_values(by='FareRange', ascending=True)


# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Fare'] <= 43.833, 'FareBand'] = 0
    dataset.loc[(dataset['Fare'] > 43.833) & (dataset['Fare'] <= 87.667), 'FareBand'] = 1
    dataset.loc[(dataset['Fare'] > 87.667) & (dataset['Fare'] <= 131.5), 'FareBand'] = 2
    dataset.loc[(dataset['Fare'] > 131.5) & (dataset['Fare'] <= 175.333), 'FareBand'] = 3
    dataset.loc[(dataset['Fare'] > 175.333) & (dataset['Fare'] <= 219.167), 'FareBand'] = 4
    dataset.loc[ dataset['Fare'] > 219.167, 'FareBand'] = 5
    dataset['FareBand'] = dataset['FareBand'].astype(int)
train_df.head()


# Sex needs to be turned into a binary variable in order to be utilised, so females were converted to 0 and males converted to 1

# In[ ]:


#Turn the sex category into a binary variable
gender_mapping = {"female": 0, "male": 1}
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map(gender_mapping)
    dataset['Sex'] = dataset['Sex'].fillna(0)

train_df.head()


# Again embarked is a categorical variable and needs to be coverted to a numerical value.

# In[ ]:


#Need to turn embarked into a number
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# Other tutorials suggest that the size of the family has an impact on the ability of a person to survive, however, grouping shows there are a large number of family sizes, this is then reduced to a person either being alone ot with their family

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# The unnecessary columns are dropped as they are not considered to be variables that useful, or were created as dummy variables.

# In[ ]:


#drop the 2 band column
train_df = train_df.drop(['FareRange','AgeRange','Name','Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]


# In[ ]:


#set up the data for the models
X_train = train_df.drop(['Survived','PassengerId',], axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop(['PassengerId','Name','Parch', 'SibSp', 'FamilySize'], axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
#print (X_train.head())


# # Feature Selection
# Now the data has been cleaned and the necessary variables created and removed, I will start the process of feature selection. I have decided to use 3 methods and based on the results, I will will use correlation and variance inflation factor (VIF) if necessary to decide on the final variables. 

# In[ ]:


#Feature selection using RFE
model = LogisticRegression()
rfe = RFE(model)
fit = rfe.fit(X_train, Y_train)
print("Num Features: ",fit.n_features_)
print("Selected Features: ",fit.support_)
print("Feature Ranking: ",fit.ranking_)
print(X_train.head())


# In[ ]:


print ("Top 4 features for RFE are: PClass, Sex, Embarked, Title")


# In[ ]:


#Feature selection using SelectKBest
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X_train, Y_train)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X_train)
# summarize selected features
print(features[0:5,:])
print (X_train.head())


# In[ ]:


print ("Top 4 features for SelectKBest are: Sex, Fare Title & FareBand")


# In[ ]:


# feature extraction using extra trees classifier
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)
print(model.feature_importances_)
X_train.head()


# In[ ]:


print ("Top 4 features  for Extra trees Classifier are: Fare, Sex, Age, & Pclass")


# Based on the 3 methods of feature importance, it has been determined the most important features are:
# 
# * Extra Trees Classifier: Fare, Sex, Age, & Pclass
# * SelectKBest: Sex, Fare Title & FareBand
# * RFE: PClass, Sex, Embarked, Title

# In[ ]:


train_df.corr(method = 'pearson')


# In[ ]:


# For each X, calculate VIF and save in dataframe
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns
vif.round(1)


# It is interesting to see how these chosen features rank in relation to correlation with the Survivial variable:
# 
# * Fare with a correlation of 0.25 indicates a weak relationship with the target variable and in a manner similar with sex and title it is hardly surpising to see a strong relationship between Pclass and Fare, suggesting possible colinearity, however, the VIF for fare is higher than fareBand , and it has been suggested as an important feature by 2 of the 3 selection methods, therefore, it will be kept, Where as fareBand was only selected by one and its correlation is weaker than fare.
# 
# * Age, along with AgeBand show a weak, to almost non-existant correlation with the target variable, and considering it only appeared in 1 feature selection method out of 3, additionly the really high VIF suggests there is some colineraity between the Age variables and others. Not surprising as discussed in further detail with other variables.
# 
# * Sex as covered shows a strong correlation with survival, in fact it has the strongest out of all the potential features. It suggests a negative relationship, since a female was labelled 0 and a male was labelled 1, it suggests males haveless of a chance of survival than a female, not unexpected given the time.
# 
# * Title is an interesting feature. It takes into account a persons sex, as well as their age. A female with the Title Mrs is generally expected to be older than a female with the title Miss and perhaps that is why there is a strong relationship between the sex variable as well as the survival variable, whereas age had no correlation at all with the survival variable.
# 
# * PClass shows a moderately strong correlation with the survival variable, as your class level increases, you have a smaller chance for survival. Not unexpected as 3rd class was located lower on the ship than first class.
# 
# * Embarked is an interesting feature, RFE suggests that it has some importance, yet correlation relationship is very weak, since the correlation is weak, and only 1 of the 3 feature selection methods chose this variable, I will remove it from the final model.
# 
# The final chosen variables are: Sex, Title, Fare & PClass.

# In[ ]:


X_train = X_train.drop(['Age','Embarked','AgeBand','Fare','IsAlone'], axis=1)
Y_train = train_df["Survived"]
X_test  = X_test.drop(['Age','Embarked','AgeBand','Fare','IsAlone'], axis=1)
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


X_test.info()


# In[ ]:


#Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_predSVM = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:




knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_predKNN = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn



# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:




# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd



# In[ ]:




# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_predDT = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree



# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_predRF = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_predRF
    })

#submission.head()
submission.to_csv('submission.csv', index=False)

