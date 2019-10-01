#!/usr/bin/env python
# coding: utf-8

# ## First Competition Notebook-- Titanic Data Set

# In[ ]:


#Import Some Basic Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


#Load data as pandas data frames, then check it out
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')   #Data I will predict on later
gender_submission = pd.read_csv('../input/gender_submission.csv')  #--- Example of submission File


# In[ ]:


train.head(2)


# In[ ]:


train.info()


# Since this my first submission and I want to avoid "analysis paralysis" (as I have been prone to in the past), my current plan is to ignore the mostly missing cabin info. I am also most likely to ditch the 3 records with missing embarked info. That leaves the age in somewhat of a bad spot since, at least initially, age would seem to be an important factor in whether someone survived or not. I may end up doing some data imputation on that later. Some things to explore would be distribution of age amongst the passengers, and correlation between age and the survival rate.

# In[ ]:


rd = train.drop('Cabin',axis=1)
rd = rd[rd['Embarked'].notnull()]


# In[ ]:


rd.info()


# In[ ]:


sns.distplot(rd[(rd['Age'].notnull()) & (rd['Survived'] == 1)]['Age'],label='Survive',color='blue',kde=False)
sns.distplot(rd[(rd['Age'].notnull()) & (rd['Survived'] == 0)]['Age'],label='Not Survive',color='red',kde=False)
plt.legend()


# Initial glance it looks like the survival rate from about 10-60 is pretty uniform. There is a definite bump for 5 and under of likeliness to survive, and a bump from 65+ of likeliness to not survive.

# In[ ]:


sns.pairplot(rd[rd['Age'].notnull()],hue='Survived')


# Age seems to be correlcated with pclass, sibsp, parch, and fare, and seems like it should be predictable off those values. Current plan is to build regression model to guess age off of other variables. From there I will will use a random forest to predict survival.

# ## Age Regression Model

# Going to make the executive decision to drop Name and ticket. There may be a chance to get some data from the Name (such as title) and ticket info... again, for now I want to avoid "analysis paralysis"

# In[ ]:


model_data = rd.drop(['Name','Ticket','PassengerId'],axis=1)


# In[ ]:


model_data.head(2)


# In[ ]:


model_data = pd.get_dummies(model_data,columns=['Sex','Embarked'],drop_first=True)


# In[ ]:


age_rd = model_data[model_data['Age'].notnull()].drop('Survived',axis=1)


# In[ ]:


from sklearn.cross_validation import train_test_split


# In[ ]:


X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(age_rd.drop('Age',axis=1), age_rd['Age'], test_size=0.3, random_state=42)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


linear_age_model = LinearRegression()
linear_age_model.fit(X_train_age,y_train_age)


# In[ ]:


preds = linear_age_model.predict(X_test_age)


# In[ ]:


from sklearn import metrics


# In[ ]:


print(metrics.mean_absolute_error(y_test_age,preds))
print(metrics.mean_squared_error(y_test_age,preds))
print(metrics.explained_variance_score(y_test_age,preds))
print(metrics.r2_score(y_test_age,preds))


# Ok, so this model isn't looking too good... I am going to try bucketing the age, then look at logistic regression, or random forest

# In[ ]:


age_rd['Age']=age_rd['Age'].apply(lambda x: 10*(x//10))


# In[ ]:


age_y = pd.DataFrame(age_rd['Age'])
age_x = age_rd.drop('Age',axis=1)


# In[ ]:


age_y = pd.get_dummies(age_y,columns=['Age'])


# In[ ]:


X_train_rfc_age,X_test_rfc_age, y_train_rfc_age, y_test_rfc_age = train_test_split(age_x, age_y, test_size=0.3, random_state=42)


# In[ ]:


X_train_rfc_age.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc_age =RandomForestClassifier(n_estimators=150)


# In[ ]:


rfc_age.fit(X_train_rfc_age,y_train_rfc_age)


# In[ ]:


rfc_age_preds=rfc_age.predict(X_test_rfc_age)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix


# In[ ]:


print(classification_report(y_test_rfc_age,rfc_age_preds))


# Precision Ain't bad for under 10 and over 50, which is where we care the most.... Not the best, but I'm sticking with it for now.

# ## Main Model
# First fill in age values where they're missing. Using the random forest. Then use the Random Forest as a predictor.

# In[ ]:


model_data.info()


# In[ ]:


model_data.head(2)


# In[ ]:


def impute(X):
    X_Age = X[~X['Age'].isnull()]
    X_Age['Age'] = X_Age['Age'].apply(lambda x: 10*(x//10))
    X_Age = pd.get_dummies(X_Age,columns=['Age'])
    
    X_NoAge = X[X['Age'].isnull()]
    no_age_preds = pd.DataFrame(rfc_age.predict(X_NoAge.drop(['Survived','Age'],axis=1)),columns=['Age_0.0','Age_10.0','Age_20.0','Age_30.0','Age_40.0','Age_50.0','Age_60.0','Age_70.0','Age_80.0'])
    X = pd.concat([X_NoAge.reset_index(),no_age_preds.reset_index()],axis=1)
    
    return pd.concat([X_Age,X.drop(['Age','index'],axis=1)],axis=0)  
    #return X.drop(['Age','index'],axis=1)
    


# In[ ]:


imputed_data = impute(model_data)


# In[ ]:


imputed_x = imputed_data.drop('Survived',axis=1)
imputed_y = imputed_data['Survived']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(imputed_x, imputed_y, test_size=0.3, random_state=9)


# In[ ]:


rfc = RandomForestClassifier(n_estimators = 200)


# In[ ]:


rfc.fit(X_train,y_train)


# In[ ]:


rfc_preds = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))


# 81%..... Not Too Bad I suppose. No we apply to competition test data to submit. First I'll retain the model on all the data that we were given

# In[ ]:


rfc.fit(imputed_x,imputed_y)


# In[ ]:


test.head()


# In[ ]:


test2 = test.drop(['Name','Ticket','Cabin'],axis = 1)


# In[ ]:


test2 = pd.get_dummies(test2,columns=['Sex','Embarked'],drop_first=True)


# In[ ]:


test2.info()


# In[ ]:


def impute2(X):
    X_Age = X[~X['Age'].isnull()]
    X_Age['Age'] = X_Age['Age'].apply(lambda x: 10*(x//10))
    X_Age = pd.get_dummies(X_Age,columns=['Age'])
    
    X_NoAge = X[X['Age'].isnull()]
    no_age_preds = pd.DataFrame(rfc_age.predict(X_NoAge.drop(['Age','PassengerId'],axis=1)),columns=['Age_0.0','Age_10.0','Age_20.0','Age_30.0','Age_40.0','Age_50.0','Age_60.0','Age_70.0','Age_80.0'])
    X = pd.concat([X_NoAge.reset_index(),no_age_preds.reset_index()],axis=1)
    
    return pd.concat([X_Age,X.drop(['Age','index'],axis=1)],axis=0)  
    #return X.drop(['Age','index'],axis=1)


# In[ ]:


test3 = impute2(test2)


# In[ ]:


test3['Age_80.0'] = 0


# Quick and Dirty - assign the one missing fare value to be the median

# In[ ]:


test3[test3['Fare'].isnull()] = test3['Fare'].median()


# In[ ]:


test3.head()


# In[ ]:


predictions = pd.DataFrame(rfc.predict(test3.drop('PassengerId',axis=1)),columns=['Survived'])


# In[ ]:


predictions = pd.concat([test3['PassengerId'].reset_index(),predictions['Survived'].reset_index()],axis=1)


# In[ ]:


predictions.drop('index',axis=1,inplace = True)


# In[ ]:


predictions.head()


# In[ ]:


gender_submission.head()


# In[ ]:




