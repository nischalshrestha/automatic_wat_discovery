#!/usr/bin/env python
# coding: utf-8

# In[129]:


# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

# normalizeation
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics


# In[130]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_df.head()


# In[131]:


titanic_df.info()
print("----------------------------")
test_df.info()


# In[132]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Ticket','Cabin','Fare','SibSp'], axis=1)
test_df    = test_df.drop(['Ticket','Cabin','Fare','SibSp'], axis=1)
titanic_df=titanic_df.dropna(axis=0,how='any')


# In[133]:



def get_masterormiss(passenger):
    name = passenger
    if (   ('Master' in str(name))         or ('Miss'   in str(name))         or ('Mlle'   in str(name)) ):
        return 1
    else:
        return 0

titanic_df['MasterMiss'] =     titanic_df[['Name']].apply( get_masterormiss, axis=1 )


test_df['MasterMiss'] =     test_df[['Name']].apply( get_masterormiss, axis=1 )

titanic_df = titanic_df.drop(['Name'], axis=1)
test_df    = test_df.drop(['Name'], axis=1)


# In[134]:


'''
# Family

# Instead of having two columns Parch & SibSp, 
# we can have only one column represent if the passenger had any family member aboard or not,
# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

titanic_df['Family'] =  titanic_df["Parch"] + titanic_df["SibSp"]
titanic_df['Family'].loc[titanic_df['Family'] > 0] = 1
titanic_df['Family'].loc[titanic_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

# drop Parch & SibSp
titanic_df = titanic_df.drop(['SibSp','Parch'], axis=1)
test_df    = test_df.drop(['SibSp','Parch'], axis=1)

# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

# sns.factorplot('Family',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Family', data=titanic_df, order=[1,0], ax=axis1)

# average of survived for those who had/didn't have any family member
family_perc = titanic_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)

'''


# In[135]:



# get average, std, and number of NaN values in test_df
average_age_test   = test_df["Age"].mean()
std_age_test       = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()

# generate random numbers between (mean - std) & (mean + std)
rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)


test_df["Age"][np.isnan(test_df["Age"])] = rand_2


test_df['Age']    = test_df['Age'].astype(int)
        


# In[136]:


# Sex

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
# So, we can classify passengers as males, females, and child
def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex
    
titanic_df['Person'] = titanic_df[['Age','Sex']].apply(get_person,axis=1)
test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

# No need to use Sex column since we created Person column
titanic_df.drop(['Sex'],axis=1,inplace=True)
test_df.drop(['Sex'],axis=1,inplace=True)

# create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
person_dummies_titanic  = pd.get_dummies(titanic_df['Person'])
person_dummies_titanic.columns = ['Child','Female','Male']

person_dummies_test  = pd.get_dummies(test_df['Person'])
person_dummies_test.columns = ['Child','Female','Male']



titanic_df = titanic_df.join(person_dummies_titanic)
test_df    = test_df.join(person_dummies_test)

fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

# sns.factorplot('Person',data=titanic_df,kind='count',ax=axis1)
sns.countplot(x='Person', data=titanic_df, ax=axis1)

# average of survived for each Person(male, female, or child)
person_perc = titanic_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

titanic_df.drop(['Person'],axis=1,inplace=True)
test_df.drop(['Person'],axis=1,inplace=True)


# In[137]:



# Pclass
# sns.factorplot('Pclass',data=titanic_df,kind='count',order=[1,2,3])
'''
sns.factorplot('Pclass','Survived',order=[1,2,3], data=titanic_df,size=5)

# create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
pclass_dummies_titanic  = pd.get_dummies(titanic_df['Pclass'])
pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']

pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']

titanic_df.drop(['Pclass'],axis=1,inplace=True)
test_df.drop(['Pclass'],axis=1,inplace=True)

titanic_df = titanic_df.join(pclass_dummies_titanic)
test_df    = test_df.join(pclass_dummies_test)
'''


# In[138]:



# Embarked

# only in titanic_df, fill the two missing values with the most occurred value, which is "S".

# plot
sns.factorplot('Embarked','Survived', data=titanic_df,size=4,aspect=3)

fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

# sns.factorplot('Embarked',data=titanic_df,kind='count',order=['S','C','Q'],ax=axis1)
# sns.factorplot('Survived',hue="Embarked",data=titanic_df,kind='count',order=[1,0],ax=axis2)
sns.countplot(x='Embarked', data=titanic_df, ax=axis1)
sns.countplot(x='Survived', hue="Embarked", data=titanic_df, order=[1,0], ax=axis2)

# group by embarked, and get the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)

# Either to consider Embarked column in predictions,
# and remove "S" dummy variable, 
# and leave "C" & "Q", since they seem to have a good rate for Survival.

# OR, don't create dummy variables for Embarked column, just drop it, 
# because logically, Embarked doesn't seem to be useful in prediction.

embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
embark_dummies_titanic.drop(['Q','S'], axis=1, inplace=True)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
embark_dummies_test.drop(['Q','S'], axis=1, inplace=True)

titanic_df = titanic_df.join(embark_dummies_titanic)
test_df    = test_df.join(embark_dummies_test)




# In[139]:



def get_malem(passenger):
    pc,male,mm=passenger
    if ( (pc==1 or pc==2 ) and male==1 and mm==0):
        return 1
    else:
        return 0

titanic_df['Pmm'] =     titanic_df[['Pclass','Male','MasterMiss']].apply( get_malem, axis=1 )
test_df['Pmm'] =     test_df[['Pclass','Male','MasterMiss']].apply( get_malem, axis=1 )

    


# In[140]:


titanic_df['ParchBinary'] =   titanic_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
test_df['ParchBinary'] =   test_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)

    
def get_malepm(passenger):
    pch,male,mm=passenger
    if ( pch==1 and male==1 and mm==0):
        return 1
    else:
        return 0

titanic_df['PchM'] =     titanic_df[['ParchBinary','Male','MasterMiss']].apply( get_malepm, axis=1 )
test_df['PchM'] =     test_df[['ParchBinary','Male','MasterMiss']].apply( get_malepm, axis=1 )


# In[141]:


def get_malec(passenger):
    em,male=passenger
    if ( em==1 and male==0):
        return 1
    else:
        return 0

titanic_df['Fc'] =     titanic_df[['C','Male']].apply( get_malec, axis=1 )
test_df['Fc'] =     test_df[['C','Male']].apply( get_malec, axis=1 )

titanic_df.drop(['Male'],axis=1,inplace=True)
test_df.drop(['Male'],axis=1,inplace=True)

titanic_df.drop(['C','Embarked','MasterMiss','Parch','ParchBinary'], axis=1,inplace=True)
test_df.drop(['C','Embarked','MasterMiss','Parch','ParchBinary'], axis=1,inplace=True)


# In[142]:


X_train = titanic_df.drop("Survived",axis=1)
Y_train = titanic_df["Survived"]
X_test  = test_df.drop("PassengerId",axis=1)


# In[143]:


#normalize the data to range (0,1), speed up from about 2000 iterations to 100
#I think the reason was that the age column has too much diffenence with the only 0,1 columns
min_max_scaler = preprocessing.MinMaxScaler()
X_train= min_max_scaler.fit_transform(X_train)
X_train=pd.DataFrame(X_train)
min_max_scaler = preprocessing.MinMaxScaler()
X_test= min_max_scaler.fit_transform(X_test)
X_test=pd.DataFrame(X_test)


# In[144]:


logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[145]:


coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# In[146]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)

