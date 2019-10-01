#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
sns.set_style('whitegrid')

get_ipython().magic(u'matplotlib inline')


# In[ ]:


###Loading the data
titanic_df = pd.read_csv('../input/train.csv')
titanic_df.head()


# In[ ]:


titanic_df.info()


# In[ ]:


###Who were the passengers on the titanic? (What age, gender, class etc)

###Gender Plot
sns.factorplot('Sex',data=titanic_df,kind='count')

### Shows more male passengers than female 


# In[ ]:


### Class plot
sns.factorplot('Pclass',data=titanic_df,kind='count')


# In[ ]:


###Interesting! More passengers are from class Three. Now lets find the gender ration among the classes

sns.factorplot('Pclass',data=titanic_df,hue='Sex',kind='count')


# In[ ]:


##This gives us an insight that there are quite a few males than females in 3rd class. Now lets dig deeper and find the children among the passengers.

def titanic_children(passenger):
    
    age , sex = passenger
    if age <16:
        return 'child'
    else:
        return sex

titanic_df['person'] = titanic_df[['Age','Sex']].apply(titanic_children,axis=1)
        


# In[ ]:


titanic_df.head(10)


# In[ ]:


### Plotting a graph to check the ratio of male,female and children in each category of class

sns.factorplot('Pclass',data=titanic_df,hue='person',kind='count')


# More number of males, females and children in the class three. WIll this insight help us in  making  prediction? let's check it out!
# 

# In[ ]:


###Now let us look at the ages of the passengers

titanic_df['Age'].hist(bins=70)


# In[ ]:


as_fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# In[ ]:


as_fig = sns.FacetGrid(titanic_df,hue='person',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# In[ ]:


as_fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=5)

as_fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

as_fig.set(xlim=(0,oldest))

as_fig.add_legend()


# From the above graphs, we can infer that there are more number of passengers with a age group of 20 to 40 in all the three classes.

# In[ ]:


###Mean age of the passengers
titanic_df['Age'].mean()


# In[ ]:


titanic_df['Age'] = titanic_df['Age'].fillna(titanic_df['Age'].mean())


# In[ ]:


#### Drop the Cabin column as there are many null values and it does not help in making prediction

titanic_df.drop('Cabin',axis=1,inplace=True)


# In[ ]:


## Filling the null values in the Embarked column with S as there are more number of passengers boarded from Southhampton
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')



# In[ ]:


## To check if there are still any null values in the dataset
titanic_df.isnull().values.any()


# In[ ]:


sns.factorplot('Embarked',data=titanic_df,kind='count')


# In[ ]:


sns.factorplot('Embarked',data=titanic_df,hue='Pclass',kind='count')


# It is intereting to see that most of the passengers boarded at Queenstown are from 3rd class. And many passengers boarded at Southhampton. Will this help in making predictions? 

# In[ ]:


## Let's check who are with family and who are alone
## This can be found by adding Parch and Sibsp columns
titanic_df['Alone'] = titanic_df.Parch + titanic_df.SibSp


# In[ ]:


## if Alone value is >0 then they are with family else they are Alone

titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Without Family'


# In[ ]:


#Let us visualise the Alone column

sns.factorplot('Alone',kind='count',data=titanic_df)


# In[ ]:


# let us see who are alone according to class
sns.factorplot('Alone',kind='count',data=titanic_df,hue='Pclass')


# Let's dig deeper into data and find out what factors helped survival.

# In[ ]:


sns.factorplot('Survived',data=titanic_df,kind='count')


# In[ ]:


## checking of the class had any effect in the survival rate
sns.factorplot('Survived',data=titanic_df,kind='count',hue='Pclass')


# In[ ]:


sns.factorplot('Pclass','Survived',data=titanic_df,hue='person')


# The above graph shows that the survival rate for male is very low nevertheless of the class. And, the survival rate is less for the 3rd class passengers.

# In[ ]:


sns.factorplot('Pclass','Survived',data=titanic_df,hue='Alone')


# As expected, the survival rates are higher if they are with family.  Let us check how Age playes a role in the survival rate.

# In[ ]:


sns.lmplot('Age','Survived',data=titanic_df)


# In[ ]:




sns.lmplot('Age','Survived',data=titanic_df,hue='Pclass')


# The above graphs shows that  older the passenger, lesser the chance of survival. 

# In[ ]:


sns.lmplot('Age','Survived',data=titanic_df,hue='Sex')


# In[ ]:


sns.lmplot('Age','Survived',data=titanic_df,hue='Alone')


# In[ ]:


sns.lmplot('Age','Survived',data=titanic_df,hue='Embarked')


# Shockingly, the number of passengers boarded at Southhampton are more compared to Cherbourg and Queenstown but the survival rate is high for Cherbour passengers than Southhampton. So there is a chance that Embarked  helps in prediction. Let us create dummies of the Embarked and drop the Queenstown to avoid multicollinearity(might be caused due to dummies) and there are quiet a few passengers boarded at Queenstown(more are from 3rd class which has less survival rate)

# Now from the analysis we understood the important features for making predictions. 
# Features to be used for Predicting: Age, female,child,with family, C, S, class_1, class_2,Fare
# Now let's drop the other features like PassengerId, Name, Sibsp, Parch, Ticket as these are not much useful in the predictions. Also drop other features like without family, male, class_3,Q as they having a very low survival rate. 

# In[ ]:


person_dummies = pd.get_dummies(titanic_df['person'])
alone_dummies = pd.get_dummies(titanic_df['Alone'])

embarked_dummies = pd.get_dummies(titanic_df['Embarked'])

embarked_dummies.drop('Q',axis=1,inplace=True)


# In[ ]:


pclass_dummies = pd.get_dummies(titanic_df['Pclass'])

pclass_dummies.columns=['class_1','class_2','class_3']


# In[ ]:


import math

titanic_df['Age'] = titanic_df['Age'].apply(math.ceil)
titanic_df['Fare'] = titanic_df['Fare'].apply(math.ceil)


# In[ ]:


titanic_df = pd.concat([titanic_df,pclass_dummies,person_dummies,alone_dummies,embarked_dummies],axis=1)


# In[ ]:


titanic_df.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Embarked'],axis=1,inplace=True)
titanic_df.drop(['Alone','person','Pclass','Without Family','male','class_3'],axis=1,inplace=True)


# In[ ]:


titanic_df.head()


# In[ ]:


titanic_train = titanic_df.drop('Survived',axis=1)
titanic_survived = titanic_df.Survived


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(titanic_train,titanic_survived,test_size=0.2)


# In[ ]:


x_train.head()


# In[ ]:


x_train.head()


# In[ ]:


x_test.head()


# In[ ]:


log_model = LogisticRegression()

log_model.fit(x_train,y_train)

train_survival = log_model.predict(x_test)


# In[ ]:


print("Accuracy Score of logistic model is",metrics.accuracy_score(y_true=y_test,y_pred=train_survival))


# In[ ]:


corr_coeff = list(zip(x_train.columns,np.transpose(log_model.coef_)))


# In[ ]:


print('Correlation coefficients are ',corr_coeff)


# In[ ]:


rand_model = RandomForestClassifier()
rand_model.fit(x_train,y_train)

rand_predict = rand_model.predict(x_test)
#rand_model.score(y_test,rand_predict)


# In[ ]:


print("Accuracy Score of Random Forest model is",metrics.accuracy_score(y_true=y_test,y_pred=rand_predict))


# In[ ]:


## Null error rate

y_train.mean()

## The accuarcy is greater than the 1-y_train.mean() = x < accuracy which means the model is not just guessing the output.


# In[ ]:


## Laoding the test data
titanic_df_test = pd.read_csv('../input/test.csv')


# In[ ]:


titanic_df_test.head()


# In[ ]:


## Storing the PassengerId column for the submission purpose
passenger_id = titanic_df_test.PassengerId


# In[ ]:


embarked_test_dummies = pd.get_dummies(titanic_df_test['Embarked'])

embarked_test_dummies.drop('Q',axis=1,inplace=True)


# In[ ]:


titanic_df_test['Alone'] = titanic_df_test.SibSp + titanic_df_test.Parch

titanic_df_test['Alone'].loc[titanic_df_test['Alone']>0] = 'With Family'
titanic_df_test['Alone'].loc[titanic_df_test['Alone'] == 0] = 'Without Family'

#titanic_df_test.head()


# In[ ]:


alone_test_dummies = pd.get_dummies(titanic_df_test['Alone'])

pclass_test_dummies = pd.get_dummies(titanic_df_test['Pclass'])

pclass_test_dummies.columns = ['class_1','class_2','class_3']



# In[ ]:


titanic_df_test['person'] = titanic_df_test[['Age','Sex']].apply(titanic_children,axis=1)

person_test_dummies = pd.get_dummies(titanic_df_test['person'])


# In[ ]:


titanic_df_test = pd.concat([titanic_df_test,embarked_test_dummies,alone_test_dummies,person_test_dummies,pclass_test_dummies],axis=1)


# In[ ]:


titanic_df_test.drop(['PassengerId','Name','Sex','SibSp','Parch','Ticket','Cabin','Embarked','Alone','person','Pclass','Without Family','male','class_3'],axis=1,inplace=True)


# In[ ]:


titanic_df_test.head()


# In[ ]:


titanic_df_test['Age'] = titanic_df_test['Age'].fillna(titanic_df_test['Age'].mean())


# In[ ]:


titanic_df_test['Fare'] = titanic_df_test['Fare'].fillna(titanic_df_test['Fare'].mean())


# In[ ]:


titanic_df_test['Age'] = titanic_df_test['Age'].apply(math.ceil)
titanic_df_test['Fare'] = titanic_df_test['Fare'].apply(math.ceil)


# In[ ]:


survival_prediction = log_model.predict(titanic_df_test)


# In[ ]:


rand_survival_predictions = rand_model.predict(titanic_df_test)


# In[ ]:


Final_predictions = DataFrame({'passenger_id':passenger_id,'survived':survival_prediction})

#Final_predictions.to_csv('titanic.csv',index=False)


# In[ ]:


Final_predictions.head()


# In[ ]:


### Let's see if our intuitions were correctly predicted by the model

check_model = pd.read_csv('../input/test.csv')


# In[ ]:


check_model['survived'] = rand_survival_predictions


# In[ ]:


check_model


# In[ ]:


check_model['Age'] = check_model['Age'].fillna(check_model['Age'].mean())


# In[ ]:


sns.factorplot('survived',data=check_model,kind='count',hue='Pclass')


# In[ ]:


sns.factorplot('survived',data=check_model,kind='count',hue='Sex')


# In[ ]:





# In[ ]:




