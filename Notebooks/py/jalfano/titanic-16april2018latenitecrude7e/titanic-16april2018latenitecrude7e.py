#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Commentary on Monday April 16 2018, 9:43pm
# The first edition was rather simplistic: its predictive features were only "Female" and "Pclass". 
# Its score when submitted to Kaggle is approximately 0.69856 .

# Now we update the code for more refinement.
# The predictive features are "Female" and "MasterMiss" and "Pclass".
# Its score when submitted to Kaggle is approximately 0.67464 .

# Again we update the code for more refinement.
# The predictive features are "Female" and "MasterMiss" and "Embarked" and "Pclass".
# Its score when submitted to Kaggle is approximately 0.72248 .

# And again we update the code for more refinement.
# The predictive features are "ParchBinary" and Female" and "MasterMiss" and "Embarked" and "Pclass".
# Its score when submitted to Kaggle is approximately 0.72727 .

# Yet again we update the code for more refinement.
# The predictive features are "Female" and "MasterMiss" and "SibSpBinary" and "ParchBinary" and "Embarked" and "Pclass".
# Its score when submitted to Kaggle is approximately 0.71291.


# In[11]:


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


# In[12]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")


# preview the data
print(titanic_df.head())


# In[13]:


#print(titanic_df.isnull().sum())
titanic_df.info()
print("----------------------------")
test_df.info()


test_df["Survived"] = -1

print("============================")
titanicANDtest_df = pd.concat([titanic_df, test_df], keys=['titanic', 'test'])


# In[14]:


float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#$# Nov 19 edit
print('Here are the NAN counts of titanic_df')
print( titanic_df.isnull().sum(), '\n' )


# In[15]:


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


# In[16]:


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


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[22]:


# Here is my python code.
def myCoin5(passenger):
    female, mastermiss, sibspbinary, parchbinary, pclass, embarked = passenger
    if   (female == 1 and mastermiss == 1 and sibspbinary == 1 and parchbinary == 1):
        if   (pclass == 1 and embarked == 'C'):
            return 1.00    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 0.75  
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
            return 0.93    
        elif (pclass == 1 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 1 and embarked == 'S'):
            return 1.00  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 1.0  
        elif (pclass == 2 and embarked == 'S'):
            return 0.88              
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
            return 0.83  
        elif (pclass == 2 and embarked == 'C'):
            return 1.00    
        elif (pclass == 2 and embarked == 'Q'):
            return 999 # NaN  
        elif (pclass == 2 and embarked == 'S'):
            return 0.91               
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
            return 0.80              
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
            return 0.91                
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

test_df["Survived"] = test_df[ ["YourCoin5"] ].apply(myFlip3 , axis=1)
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


# In[23]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
#        "Survived": Y_pred
        "Survived": test_df["Survived"]
    })
submission.to_csv('titanic.csv', index=False)

