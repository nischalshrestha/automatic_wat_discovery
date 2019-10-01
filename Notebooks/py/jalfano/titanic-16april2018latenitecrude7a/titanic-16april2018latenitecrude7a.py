#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Commentary on Monday April 16 2018, 9:43pm
# This edition is rather simplistic;
# and its score when submitted to Kaggle is approximately 0.69856 .


# In[1]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
## %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[2]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")


# preview the data
print(titanic_df.head())


# In[3]:


#print(titanic_df.isnull().sum())
titanic_df.info()
print("----------------------------")
test_df.info()


test_df["Survived"] = -1

print("============================")
titanicANDtest_df = pd.concat([titanic_df, test_df], keys=['titanic', 'test'])


# In[4]:


#import numpy as np
#import math
#constant_one = 1
#constant_three = 3
#constant_quotient = constant_one/constant_three
#print( apply(lambda x: "%.5f" % x,  constant_quotient) )

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#def myFunction(inputx):
#    print( "%.66f"     % inputx)
#myFunction(math.pi)

#mySquare = lambda x : x*x
#print( mySquare(-7) )

#youPass = lambda hw, qz, test: (0.20*hw + 0.3*qz + 0.5*test) >= 70
#print(youPass(0, 65, 75))

#print(float_formatter(0.333))
#print(float_formatter(0.5))
#print(float_formatter(1/3))
#print(float_formatter(math.pi))
#print(float_formatter(constant_quotient))

#$# Nov 19 edit
print('Here are the NAN counts of titanic_df')
print( titanic_df.isnull().sum(), '\n' )


# In[5]:


print('Pclass and Sex are useful factors.')
print('Here are pivot tables for survivor sum, passenger count, and mean.\n')

table0a = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Pclass'], columns=['Sex'], aggfunc=np.sum)
print( table0a,'\n' )

table0b = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Pclass'], columns=['Sex'], aggfunc='count')
print( table0b,'\n' )

table0c = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Pclass'], columns=['Sex'], aggfunc=np.mean)
print( table0c,'\n' )

print('So we create new columns Female and Male for machine learning.\n')

sex_dummies_titanic  = pd.get_dummies(titanic_df['Sex'])
sex_dummies_titanic.columns = ['Female','Male']
#13April# sex_dummies_titanic.drop(['Male'], axis=1, inplace=True)
titanic_df = titanic_df.join(sex_dummies_titanic)
#13April# titanic_df['Fem'] = titanic_df['Female']
#13April# titanic_df['F'] = titanic_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_test  = pd.get_dummies(test_df['Sex'])
sex_dummies_test.columns = ['Female','Male']
#13April# sex_dummies_test.drop(['Male'], axis=1, inplace=True)
test_df = test_df.join(sex_dummies_test)
#13April# test_df['Fem'] = test_df['Female']
#13April# test_df['F'] = test_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Sex'])
sex_dummies_titanicANDtest.columns = ['Female','Male']
#13April# sex_dummies_titanicANDtest.drop(['Male'], axis=1, inplace=True)
titanicANDtest_df = titanicANDtest_df.join(sex_dummies_titanicANDtest)
#13April# titanicANDtest_df['Fem'] = titanicANDtest_df['Female']
#13April# titanicANDtest_df['F'] = titanicANDtest_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

titanicANDtest_df.head(5)



pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class1','Class2','Class3']
titanic_df    = titanic_df.join(pclass_dummies_titanic)

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class1','Class2','Class3']
test_df    = test_df.join(pclass_dummies_test)

pclass_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Pclass'])
pclass_dummies_titanicANDtest.columns = ['Class1','Class2','Class3']
titanicANDtest_df    = titanicANDtest_df.join(pclass_dummies_titanicANDtest)

titanicANDtest_df.head(5)


# In[6]:


# Here is my python code.

def myTestCode(input):
    return input*input

print(myTestCode(4))

def myCoin(passenger):
    sex, pclass = passenger
    if (sex == 1):
        if (pclass == 1):
            return 0.968085
        elif (pclass == 2):
            return 0.921053
        else:
            return 0.500000
    else:
        if (pclass == 1):
            return 0.368852
        elif (pclass == 2):
            return 0.157407
        else:
            return 0.135447

print(myCoin( [1, 3] ))        
        
test_df.head(5)        

test_df["YourCoin"] = test_df[ ["Female","Pclass"] ].apply(myCoin, axis=1)
test_df["YourCoin"].tail(5)


### THIS IS MONDAY APRIL 16 NEW !!!
from random import randint
from random import random
import random

def flip(p):                                   ## I stole this from StackOverflow.
    return 'H' if random.random() < p else 'T' ## It returns H with probability p.

def myFlip(p):
    return 1 if (random.random() < p) else 0

def myFlip2(p):
    return( (random.random() < p) )

def myFlip3(p):
    return( int(random.random() < p) )

def myFlip4(p):
    return( str(random.random() < p) )

## for i in range(20):
##      print( myFlip4(5/18) == 'True' )

test_df["Survived"] = test_df[ ["YourCoin"] ].apply(myFlip3 , axis=1)
test_df.head(25)

## test_df["Survived"] = test_df["Survived"].astype(bool)
## test_df.tail(10)


# In[ ]:





# In[ ]:


## Logistic Regression
#
#logreg = LogisticRegression()
#
#logreg.fit(X_train, Y_train)
#
#Y_pred = logreg.predict(X_test)
#
#logreg.score(X_train, Y_train)


# In[ ]:


## get Correlation Coefficient for each feature using Logistic Regression
#coeff_df = DataFrame(titanic_df.columns.delete(0))
#coeff_df.columns = ['Features']
#coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
#
## preview
#coeff_df


# In[7]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
#        "Survived": Y_pred
        "Survived": test_df["Survived"]
    })
submission.to_csv('titanic.csv', index=False)

