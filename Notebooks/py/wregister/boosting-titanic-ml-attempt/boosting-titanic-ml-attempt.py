#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# imports
import pandas as pd
import matplotlib.pyplot as mlp
import seaborn as sns
import numpy as np
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#get train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


train.tail()


# In[ ]:


train.info()
print('-'*42)
test.info()


# Age has about 200 nulls and cabin has about 700 it is unlikely that cabin can be useable without causing bias those passengers who cabin was recorded, but age can likely be patched up with averges. So cabin will be dropped from the data in both train and test. Additionally there are 2 passengers with nulls for embarked, if they survive the cabin drops I will decide whether to avg the embarked for their class, age, or ticket, or just drop them.

# In[ ]:


#drop Cabin from both dfs as well as passenger id bc we do not need a second index
train.drop(['Cabin'], axis = 1, inplace = True)
test.drop(['Cabin'], axis = 1, inplace = True)


# In[ ]:


#graph a hist of age
sns.set_style('whitegrid')
train['Age'].hist(bins = 50)
mlp.xlabel('Age')


# In[ ]:


# plot a boxplot of the avg ages of the three classes
mlp.figure(figsize = (12, 7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train)


# In[ ]:


#Create a function that replaces nulls in the age culumn with the average of that passenger's class

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[ ]:


#replace null ages
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)


# In[ ]:


train.info()


# In[ ]:


# drop the embarked nulls
train.dropna(inplace = True)


# In[ ]:


train.info()


# Remaining issues are the categorical columns that still have text like sex, embarked, ticket, and name. Name might be an issue although the titels of the people ma be useful in some way. Ticket is an issue because there are some repeats but most are unique and they all contain letters and numbers.

# In[ ]:


#replace sex and embarked characters with categorical columns using pd.get_dummies
Sex=pd.get_dummies(train[['Sex']], drop_first = True)
Sex2=pd.get_dummies(test[['Sex']], drop_first = True)

def embarked_enumerator(cols):
    embarked = cols[0] 
    
    if embarked == 'Q':
        return 0
    elif embarked == 'S':
        return 1
    else:
        return 2


# In[ ]:


train['Embarked'] = train[['Embarked','Pclass']].apply(embarked_enumerator,axis=1)
test['Embarked'] = test[['Embarked','Pclass']].apply(embarked_enumerator,axis=1)


# In[ ]:


#remove old Sex and Embarked columns
train.drop(['Sex'],axis=1,inplace=True)
test.drop(['Sex'],axis=1,inplace=True)


# In[ ]:


#concatonate the new versions of embarked and sex with train and test
train = pd.concat([train, Sex], axis = 1)
test = pd.concat([test, Sex2], axis = 1)


# In[ ]:


train.head()


# In[ ]:


#extract the prefix of each passenger from their name
Both = [train, test]

for dataframe in Both:
    dataframe['Prefix'] = dataframe.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


train.tail()


# In[ ]:


pd.crosstab(train['Prefix'], train['Sex_male'])


# Many of these preifxed are repeats of the same thing simply abbreviated differently like miss and ms., others are too rare to keep as their own category so it would be best to consolidate names before getting dummies again

# In[ ]:


#replace similar names with a single name, and take uncommon names and replace them with the unusual tag
for dataframe in Both:
    dataframe['Prefix'] = dataframe['Prefix'].replace(['Countess', 'Lady','Capt', 'Don', 	'Col', 'Dr', 'Rev', 'Major', 'Dona', 'Sir', 'Johnkeer'], 'Unusual')
    
    dataframe['Prefix'] = dataframe['Prefix'].replace(['Mlle', 'Ms'], 'Miss')
    dataframe['Prefix'] = dataframe['Prefix'].replace('Mme', 'Mrs')


# In[ ]:


#now that there are fewer categories of names we can get dummies for the prefix column
def prefix_enumerator(cols):
    prefix = cols[0] 
    
    if prefix == 'Master':
        return 0
    elif prefix == 'Miss':
        return 1
    elif prefix=='Mr':
        return 5
    elif prefix=='Mrs':
        return 3
    else:
        return 4


# In[ ]:


train['Prefix'] = train[['Prefix','Pclass']].apply(prefix_enumerator,axis=1)
test['Prefix'] = test[['Prefix','Pclass']].apply(prefix_enumerator,axis=1)


# In[ ]:


#now we can drop name as it is not needed anymore
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


# Parch and SibSp both measure family size, and since equating siblings to
#spouses and parents to children eliminates a lot of the good the columns could do alone we might as well combine them

train['Fammems'] = train['Parch'] + train['SibSp']
test['Fammems'] = test['Parch'] + test['SibSp']


# In[ ]:


train.drop(['SibSp', 'Parch'], axis=1, inplace=True)
test.drop(['SibSp', 'Parch'], axis = 1, inplace = True)


# Age is a continuous feature that might cause correlations to bne found between random age numbers that dont exist, dumbing down the ages to various age groupings would make it much easier

# In[ ]:


#Create a function that replaces age numbers with more categorical age bands

def age_bander(cols):
    Age = cols[0]
    
    if Age < 16:
        return 0
    elif Age >= 16 and Age < 32:
        return 1
    elif Age >= 32 and Age < 48:
        return 2
    elif Age >= 48 and Age < 64:
        return 3
    else:
        return Age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(age_bander,axis=1)
test['Age'] = test[['Age','Pclass']].apply(age_bander,axis=1)


# In[ ]:


train.drop(['Ticket'], axis = 1, inplace = True)
test.drop(['Ticket'], axis = 1, inplace = True)


# In[ ]:


test.head()


# In[ ]:


train['Fare'].describe(percentiles = [.15, .30, .45, .60, .75, .90])


# In[ ]:


#Create a function that replaces fare numbers with more categorical fare bands

def fare_bander(cols):
    Fare = cols[0]
    
    if Fare < 7.76:
        return 0
    elif Fare >= 7.76 and Fare < 8.06:
        return 1
    elif Fare >= 8.06 and Fare < 13:
        return 2
    elif Fare >= 13 and Fare < 14.4543:
        return 3
    elif Fare >= 14.4543 and Fare < 21.076:
        return 4
    elif Fare >= 21.076 and Fare < 31:
        return 5
    else:
        return Fare


# In[ ]:


train['FareBand'] = train[['Fare','Pclass']].apply(fare_bander,axis=1)
test['FareBand'] = test[['Fare','Pclass']].apply(fare_bander,axis=1)
train.drop('Fare', axis = 1, inplace = True)
test.drop('Fare', axis = 1, inplace = True)


# In[ ]:


train.head()


# Now we can do some actual learning with our fixed up data.

# In[ ]:


X = train.drop('Survived', axis = 1) 
y = train['Survived']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived', 'PassengerId'], axis = 1, inplace = False), train['Survived'],
                                                    test_size=0.33, random_state=42)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


# Logistic Regression

lregr = LogisticRegression()
lregr.fit(X_train, y_train)
Y_pred = lregr.predict(X_test)
acc_lregr = round(lregr.score(X_test, y_test) * 100, 2)
acc_lregr


# In[ ]:


#get the correlation of each feature using the logistic regression just done, thanks to Manav Seghal, 
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

#create a dataframe of all of the column titles from train
coeff=pd.DataFrame(train.columns.delete(0))
#rename the column to Feature
coeff.columns=['Feature']
#create a correlation column in coeff which contains the coefficient from the above linear regression for each feature
coeff['Correlation']=pd.Series(lregr.coef_[0])
#sort the coefficients in descending order
coeff.sort_values(by='Correlation', ascending=False)


# Now we can try a whole bunch of methods to see which seem to work best before we try ensembling or boosting

# In[ ]:


from sklearn import svm


# In[ ]:


# Support Vector Classifier
svc = svm.SVC()

svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
SVC_acc = round(svc.score(X_test, y_test) * 100, 2)
SVC_acc


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# KNN classifier

KNNc = KNeighborsClassifier(n_neighbors=5)
KNNc.fit(X_train, y_train)
y_pred=KNNc.predict(X_test)
KNNc_acc = round(KNNc.score(X_test, y_test)*100, 2)
KNNc_acc


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_test, y_test) * 100, 2)
acc_gaussian



# In[ ]:


from sklearn.linear_model import Perceptron


# In[ ]:


# Perceptron

ptron = Perceptron()
ptron.fit(X_train, y_train)
y_pred = ptron.predict(X_test)
acc_perceptron = round(ptron.score(X_test, y_test) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = svm.LinearSVC()
linear_svc.fit(X_train, y_train)
y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_test, y_test) * 100, 2)
acc_linear_svc


# In[ ]:


from sklearn.linear_model import SGDClassifier


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_test, y_test) * 100, 2)
acc_sgd


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_test, y_test) * 100, 2)
acc_decision_tree



# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=131)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 2)
acc_random_forest


# In[ ]:


# now to compare all the models

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [SVC_acc, KNNc_acc, acc_lregr, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


import xgboost as xgb


# In[ ]:


booster = xgb.XGBClassifier(max_depth = 3, n_estimators= 100, learning_rate= 0.1)
booster.fit(X_train, y_train)
booster.predict(X_test)
booster_acc=round(booster.score(X_test, y_test) * 100, 2)
booster_acc


# In[ ]:


# try different learning rates, estimator amounts, and max depths to see how good our booster can get
a = [3,5,10,15,20]
b = [50,100,150,200,250,300]
c = [0.1, 0.33, 0.5,0.75, 1]


# In[ ]:


for num in range(0, len(b)):
    for dig in range(0,len(a)):
        for elem in range(0, len(c)):
            booster = xgb.XGBClassifier(max_depth = a[dig], n_estimators= b[num], learning_rate= c[elem])
            booster.fit(X_train, y_train)
            booster.predict(X_test)
            booster_acc=round(booster.score(X_test, y_test) * 100, 2)
            print(booster_acc, 'depth = {}, estimators = {}, rate = {}'.format(a[dig], b[num], c[elem]))
            print('-'*40)


# In[ ]:


submission_ex = pd.read_csv('../input/gender_submission.csv')
submission_ex.tail()


# Now all we have to do is choose the best couple models and submit with them.

# In[ ]:


booster = xgb.XGBClassifier(max_depth = 5, n_estimators= 50, learning_rate=0.5)
booster.fit(X.drop('PassengerId', axis = 1), y)
boost_preds = booster.predict(test.drop(['PassengerId'], axis = 1))


# In[ ]:


booster_submission = pd.DataFrame(data = test['PassengerId'])


# In[ ]:


booster_submission['Survived'] = boost_preds


# In[ ]:


booster_submission.to_csv(path_or_buf='booster_submission', index = False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




