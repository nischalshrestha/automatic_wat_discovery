#!/usr/bin/env python
# coding: utf-8

# In[147]:


# FINAL SUBMISSION

# For every first/second class Female I changed their "coin" weight to about 1.00 
# For third class Male/Female I kept their "coin" weight to the orginal probability
# I did this becuase I felt like all First/Second Class Females were their own category, sperated from Third Class Male/Female

# I took out the function for myCoin becuase I didn't think it was useful for what I wanted to run
# Also did not include the tickets

# I wanted to run Log Reg becuase thats what we have been using and its Prof. Alfano favorite.
# However, PENG introduced me to the Voting Machine Learning Algorithm so I used that instead.
# The Voting prediction came out to be the best so I decided to stay with it.

# Reason why I took out the fuction myCoin was becuase I figured the Voting Algorithm would take over and give a better score

# FINAL SCORE = 0.80810
# BEST CAREER SCORE
# FINALLY BROKE .8


# In[126]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame
import random
# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# normalizeation
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree,svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# In[127]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")


# preview the data
print(titanic_df.head())


# In[128]:


#print(titanic_df.isnull().sum())
titanic_df.info()
print("----------------------------")
test_df.info()


test_df["Survived"] = -1

print("============================")
titanicANDtest_df = pd.concat([titanic_df, test_df], keys=['titanic', 'test'])


# In[129]:


float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#$# Nov 19 edit
print('Here are the NAN counts of titanic_df')
print( titanic_df.isnull().sum(), '\n' )


# In[130]:


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
titanic_df['Fem'] = titanic_df['Female']
titanic_df['F'] = titanic_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_test  = pd.get_dummies(test_df['Sex'])
sex_dummies_test.columns = ['Female','Male']
#13April# sex_dummies_test.drop(['Male'], axis=1, inplace=True)
test_df = test_df.join(sex_dummies_test)
test_df['Fem'] = test_df['Female']
test_df['F'] = test_df['Female']
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Sex'])
sex_dummies_titanicANDtest.columns = ['Female','Male']
#13April# sex_dummies_titanicANDtest.drop(['Male'], axis=1, inplace=True)
titanicANDtest_df = titanicANDtest_df.join(sex_dummies_titanicANDtest)
titanicANDtest_df['Fem'] = titanicANDtest_df['Female']
titanicANDtest_df['F'] = titanicANDtest_df['Female']
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


# In[131]:


print('Now from the Name we locate the MasterOrMiss passengers.')

def get_masterormiss(passenger):
    name = passenger
    if (   ('Master' in str(name))         or ('Miss'   in str(name))         or ('Mlle'   in str(name)) ):
        return 1
    else:
        return 0

titanic_df['MasterMiss'] =     titanic_df[['Name']].apply( get_masterormiss, axis=1 )
titanic_df['MMs'] = titanic_df['MasterMiss']
titanic_df['Ms'] = titanic_df['MasterMiss']
titanic_df['m'] = titanic_df['MasterMiss']

test_df['MasterMiss'] =     test_df[['Name']].apply( get_masterormiss, axis=1 )
test_df['MMs'] = test_df['MasterMiss']
test_df['Ms'] = test_df['MasterMiss']
test_df['m'] = test_df['MasterMiss']

titanicANDtest_df['MasterMiss'] =     titanicANDtest_df[['Name']].apply( get_masterormiss, axis=1 )
titanicANDtest_df['MMs'] = titanicANDtest_df['MasterMiss']
titanicANDtest_df['Ms'] = titanicANDtest_df['MasterMiss']
titanicANDtest_df['m'] = titanicANDtest_df['MasterMiss']

#$# print(titanicANDtest_df.head())
    
    

print('Here are pivot tables for survival by Sex and MasterMiss, by Pclass.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'], columns=['Pclass'],                     aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'], columns=['Pclass'],                     aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'], columns=['Pclass'],                     aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' ) 


# In[132]:


print('Now Embarked.  Fill the 2 NaNs with S, as ticket-number blocks imply.')

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanicANDtest_df["Embarked"] = titanicANDtest_df["Embarked"].fillna("S")


embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
#$# embark_dummies_titanic.columns = ['3','17','19']
#$# embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
titanic_df = titanic_df.join(embark_dummies_titanic)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
#$# embark_dummies_test.columns = ['3','17','19']
#$# embark_dummies_test.drop(['S'], axis=1, inplace=True)
test_df    = test_df.join(embark_dummies_test)

embark_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Embarked'])
#$# embark_dummies_titanicANDtest.columns = ['3','17','19']
#$# embark_dummies_titanic.drop(['S'], axis=1, inplace=True)
titanicANDtest_df = titanicANDtest_df.join(embark_dummies_titanicANDtest)


print('Pivot tables for survival by Sex + MasterMiss, by Pclass + Embark.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' ) 


# In[133]:


print('Now consider Parch as a binary decision: is the value greater than 0?')

#$# def is_positive(passenger):
#$#     parch = int(passenger)
#$#     return 1 if (parch > 0) else 0

titanic_df['ParchBinary'] =   titanic_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanic_df['Pch'] = titanic_df['ParchBinary']
titanic_df['Pc'] = titanic_df['ParchBinary']
titanic_df['p'] = titanic_df['ParchBinary']
 
test_df['ParchBinary'] =   test_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
test_df['Pch'] = test_df['ParchBinary']
test_df['Pc'] = test_df['ParchBinary']
test_df['p'] = test_df['ParchBinary']
 
titanicANDtest_df['ParchBinary'] =   titanicANDtest_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1) 
titanicANDtest_df['Pch'] = titanicANDtest_df['ParchBinary']
titanicANDtest_df['Pc'] = titanicANDtest_df['ParchBinary']
titanicANDtest_df['p'] = titanicANDtest_df['ParchBinary']


print('Pivot tables: Sex + MasterMiss + ParchBinary, by Pclass + Embark.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Pch', 'Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Pch', 'Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Pch', 'Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' )


# In[134]:


print('Now consider SibSp as a binary decision: is the value greater than 0?')

titanic_df['SibSpBinary'] =   titanic_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanic_df['SbS'] = titanic_df['SibSpBinary']
titanic_df['Sb'] = titanic_df['SibSpBinary']
titanic_df['s'] = titanic_df['SibSpBinary']
 
test_df['SibSpBinary'] =   test_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
test_df['SbS'] = test_df['SibSpBinary']
test_df['Sb'] = test_df['SibSpBinary']
test_df['s'] = test_df['SibSpBinary']

titanicANDtest_df['SibSpBinary'] =   titanicANDtest_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
titanicANDtest_df['SbS'] = titanicANDtest_df['SibSpBinary']
titanicANDtest_df['Sb'] = titanicANDtest_df['SibSpBinary']
titanicANDtest_df['s'] = titanicANDtest_df['SibSpBinary']


print('Pivot tables: ParchBinary + SibSpBinary + Sex + MasterMiss, by Embark + Pclass.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',              index = ['Fem', 'MMs', 'SbS', 'Pch'],              columns=['Pclass', 'Embarked'],              aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',              index = ['Fem', 'MMs', 'SbS', 'Pch'],              columns=['Pclass', 'Embarked'],              aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',              index = ['Fem', 'MMs', 'SbS', 'Pch'],              columns=['Pclass', 'Embarked'],              aggfunc=np.mean )
print( table0f.iloc[::-1].round(2),'\n' )


# In[135]:


# Here is my python code.
def myCoin5(passenger):
    female, mastermiss, sibspbinary, parchbinary, pclass, embarked = passenger
    if   (female == 1 and mastermiss == 1 and sibspbinary == 1 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00              
        elif (pclass == 3 and embarked == 'C'):
            return 0.80     
        elif (pclass == 3 and embarked == 'Q'):
            return 999 # NaN  
        else:
            return 0.18  # This is row 1 of 16 in the probability table.
    elif (female == 1 and mastermiss == 1 and sibspbinary == 1 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 1.0  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return  999 # NaN   
        elif (pclass == 2 and embarked == 'S'):
            return 999 # NaN              
        elif (pclass == 3 and embarked == 'C'):
            return 0.33    
        elif (pclass == 3 and embarked == 'Q'):
            return 1.00  
        else:
            return 0.20  # This is row 2 of 16 in the probability table.
    elif (female == 1 and mastermiss == 1 and sibspbinary == 0 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN
        elif (pclass == 1 and embarked == 'S'):
            return 1.00   
        elif (pclass == 2 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00              
        elif (pclass == 3 and embarked == 'C'):
            return 0.67    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.00  
        else:
            return 0.50  # This is row 3 of 16 in the probability table.
    elif (female == 1 and mastermiss == 1 and sibspbinary == 0 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 1.0  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00             
        elif (pclass == 3 and embarked == 'C'):
            return 0.67    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.76  
        else:
            return 0.44  # This is row 4 of 16 in the probability table.
    elif (female == 1 and mastermiss == 0 and sibspbinary == 1 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00 
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00               
        elif (pclass == 3 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.00  
        else:
            return 0.30  # This is row 5 of 16 in the probability table.
    elif (female == 1 and mastermiss == 0 and sibspbinary == 1 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00              
        elif (pclass == 3 and embarked == 'C'):
            return 0.50    
        elif (pclass == 3 and embarked == 'Q'):
            return 1.00  
        else:
            return 0.50  # This is row 6 of 16 in the probability table.
    elif (female == 1 and mastermiss == 0 and sibspbinary == 0 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00   
        elif (pclass == 2 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00              
        elif (pclass == 3 and embarked == 'C'):
            return 0.60    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.00  
        else:
            return 0.57  # This is row 7 of 16 in the probability table.
    elif (female == 1 and mastermiss == 0 and sibspbinary == 0 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00      
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN    
        elif (pclass == 1 and embarked == 'S'):
            return 1.00     
        elif (pclass == 2 and embarked == 'C'):
            return 1.00      
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN    
        elif (pclass == 2 and embarked == 'S'):
            return 1.00                
        elif (pclass == 3 and embarked == 'C'):
            return 1.00       
        elif (pclass == 3 and embarked == 'Q'):
            return 999 # NaN    
        else:
            return 0.67  # This is row 8 of 16 in the probability table.
    elif (female == 0 and mastermiss == 1 and sibspbinary == 1 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00   
        elif (pclass == 2 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00              
        elif (pclass == 3 and embarked == 'C'):
            return 1.00    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.00  
        else:
            return 0.28  # This is row 9 of 16 in the probability table.
    elif (female == 0 and mastermiss == 1 and sibspbinary == 1 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN   
        elif (pclass == 1 and embarked == 'S'):
            return 999 # NaN   
        elif (pclass == 2 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN   
        elif (pclass == 2 and embarked == 'S'):
            return 999 # NaN              
        elif (pclass == 3 and embarked == 'C'):
            return 1.00     
        elif (pclass == 3 and embarked == 'Q'):
            return 999 # NaN   
        else:
            return 999 # NaN  # This is row 10 of 16 in the probability table.
    elif (female == 0 and mastermiss == 1 and sibspbinary == 0 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 1.00              
        elif (pclass == 3 and embarked == 'C'):
            return 1.00     
        elif (pclass == 3 and embarked == 'Q'):
            return 999 # NaN  
        else:
            return 1.00     # This is row 11 of 16 in the probability table.
    elif (female == 0 and mastermiss == 1 and sibspbinary == 0 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN
        elif (pclass == 1 and embarked == 'S'):
            return 999 # NaN
        elif (pclass == 2 and embarked == 'C'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN
        elif (pclass == 2 and embarked == 'S'):
            return 999 # NaN            
        elif (pclass == 3 and embarked == 'C'):
            return 999 # NaN  
        elif (pclass == 3 and embarked == 'Q'):
            return 999 # NaN
        else:
            return 999 # NaN # This is row 12 of 16 in the probability table.
    elif (female == 0 and mastermiss == 0 and sibspbinary == 1 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 0.50    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 0.33  
        elif (pclass == 2 and embarked == 'C'):
            return 0.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 0.00              
        elif (pclass == 3 and embarked == 'C'):
            return 0.33    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.00  
        else:
            return 0.00      # This is row 13 of 16 in the probability table.
    elif (female == 0 and mastermiss == 0 and sibspbinary == 1 and parchbinary == 0):
        if   (pclass == 1 and embarked == 'C'):
            return 0.62    
        elif (pclass == 1 and embarked == 'Q'):
            return 0.0  
        elif (pclass == 1 and embarked == 'S'):
            return 0.40  
        elif (pclass == 2 and embarked == 'C'):
            return 0.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 0.07              
        elif (pclass == 3 and embarked == 'C'):
            return 0.00    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.20  
        else:
            return 0.09      # This is row 14 of 16 in the probability table.
    elif (female == 0 and mastermiss == 0 and sibspbinary == 0 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 0.33    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 0.00   
        elif (pclass == 2 and embarked == 'C'):
            return 999 # NaN    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 0.00               
        elif (pclass == 3 and embarked == 'C'):
            return 999 # NaN     
        elif (pclass == 3 and embarked == 'Q'):
            return 999 # NaN  
        else:
            return 0.00      # This is row 15 of 16 in the probability table.
    else:
        if   (pclass == 1 and embarked == 'C'):
            return 0.35    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 0.33  
        elif (pclass == 2 and embarked == 'C'):
            return 0.25    
        elif (pclass == 2 and embarked == 'Q'):
            return 0.0  
        elif (pclass == 2 and embarked == 'S'):
            return 0.09              
        elif (pclass == 3 and embarked == 'C'):
            return 0.15    
        elif (pclass == 3 and embarked == 'Q'):
            return 0.07  
        else:
            return 0.12      # This is row 15 of 16 in the probability table.
        
print(myCoin5( [0, 0, 0, 0, 3, 'C'] ))        
        
test_df.head(5)        

test_df["YourCoin5"] = test_df[ ["Female","MasterMiss","SibSpBinary","ParchBinary","Pclass","Embarked"] ].apply(myCoin5, axis=1)
test_df["YourCoin5"].tail(5)


### THIS IS MONDAY APRIL 16 NEW !!!
###from random import randint
###from random import random
#import random

#def flip(p):                                   ## I stole this from StackOverflow.
#    return 'H' if random.random() < p else 'T' ## It returns H with probability p.

#def myFlip(p):
#    return 1 if (random.random() < p) else 0

#def myFlip2(p):
#    return( (random.random() < p) )

#def myFlip3(p):
#    return( int(random.random() < p) )

#def myFlip4(p):
#    return( str(random.random() < p) )

## for i in range(20):
##      print( myFlip4(5/18) == 'True' )

#test_df["Survived"] = test_df[ ["YourCoin5"] ].apply(myFlip3 , axis=1)
#test_df.head(25)

## test_df["Survived"] = test_df["Survived"].astype(bool)
## test_df.tail(10)


# In[136]:


myMean = titanic_df["Fare"].mean()
print(myMean)
test_df["Fare"] = test_df["Fare"].fillna(myMean)

titanic_df.drop(["PassengerId","Name","Sex","Ticket","Cabin","Embarked","Age","MMs","Ms","m","Pch","Pc","p","SbS","Sb","s","Fem","F","Male"], axis=1, inplace=True)

test_df.drop([                 "Name","Sex","Ticket","Cabin","Embarked","Age","MMs","Ms","m","Pch","Pc","p","SbS","Sb","s","Fem","F","Male","YourCoin5", "Survived"], axis=1, inplace=True)
#titanic_df.drop(["PassengerId", "Name"], axis=1, inplace=True)

titanic_df.head()
#test_df.isnull().sum()


# In[137]:


test_df.head()


# In[138]:


X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()


# In[139]:


min_max_scaler = preprocessing.MinMaxScaler()
X_train= min_max_scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train)
min_max_scaler = preprocessing.MinMaxScaler()
X_test= min_max_scaler.fit_transform(X_test)
X_test=pd.DataFrame(X_test)


# In[140]:



model = LogisticRegression()
n_f=14 #number of features
rfe = RFE(model, n_f)#(fitting model, number of features)
fit = rfe.fit(X_train, Y_train)
print(fit.n_features_)#number
print(fit.support_)#selected
print(fit.ranking_)#rank
print(fit.score(X_train, Y_train))#fits score

X_train=fit.fit_transform(X_train,Y_train)
X_test=fit.transform(X_test)
titanic_df.drop('Survived',axis=1,inplace=True)
titanic_df=titanic_df.iloc[:,fit.support_]


# In[141]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[142]:




clf = RandomForestClassifier(n_estimators=20)

param_dist = {"max_depth": [3,4,5,6, None],
              "max_features": sp_randint(1, n_f),
              "min_samples_split": sp_randint(2, 4),
              "min_samples_leaf": sp_randint(1, n_f),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)
#random_search.fit(X_train, Y_train)
#report(random_search.cv_results_)


# In[143]:


clfe = ExtraTreesClassifier(n_estimators=20)

param_dist = {"max_depth": [3,4,5,6, None],
              "max_features": sp_randint(1, n_f),
              "min_samples_split": sp_randint(2, 4),
              "min_samples_leaf": sp_randint(1, n_f),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# run randomized search
n_iter_search = 20
random_search_e = RandomizedSearchCV(clfe, param_distributions=param_dist,
                                   n_iter=n_iter_search)


# In[144]:


#voting

logreg = LogisticRegression()
rf=random_search #from cross-validation
extree = random_search_e
svc=svm.SVC()
xg=XGBClassifier()
knn=KNeighborsClassifier()
gb=GradientBoostingClassifier()

vcr=VotingClassifier(estimators=[('lg',logreg),('rf',rf),('extree',extree),('svc',svc),('xg',xg),('knn',knn),('gb',gb)],
                     voting='hard',weights=[2,1,1,1,1,1,1])

vcr.fit(X_train, Y_train)

Y_pred = vcr.predict(X_test)

print('voting',vcr.score(X_train, Y_train))

logreg.fit(X_train, Y_train)
rf.fit(X_train, Y_train)
xg.fit(X_train, Y_train)
svc.fit(X_train, Y_train)
extree.fit(X_train, Y_train)
knn.fit(X_train, Y_train)
gb.fit(X_train, Y_train)
print('logreg',logreg.score(X_train, Y_train))
print('randforest',rf.score(X_train, Y_train))
print('extree',extree.score(X_train, Y_train))
print('svc',svc.score(X_train, Y_train))
print('xg',xg.score(X_train, Y_train))
print('knn',knn.score(X_train, Y_train))
print('gb',gb.score(X_train, Y_train))
#report(random_search.cv_results_)
#report(random_search_xgb.cv_results_)


# In[145]:


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# In[146]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
#        "Survived": test_df["Survived"]
    })
submission.to_csv('titanic.csv', index=False)

