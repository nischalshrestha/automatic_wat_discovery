#!/usr/bin/env python
# coding: utf-8

# In[27]:


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

#Update on May 7, 2018
#Added new features such as separating the men from the women, and further serparting the men and the boys
#Also decided to group people by their sex and Pclass, which gave 6 features that were helpful in my prediction
#Also changed the method of predicting from a singular logistic regression to a voting method
#The vote takes into account predictions from a logistic regression, k-nearest neighbors, random forest, and XGBoost and Support Vector Machines
#As of now the highest score I have been able ot achieve is a 0.78947

#Update on May 10, 2018 by Joe Alfano.
#This submission is from Dan Heston.  I forked a copy and obtained score 0.78468 .
#Note to Daniel Heston: good job!


# In[28]:


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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

#imported xgboost for another machine learning model
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')


# In[29]:


# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")


# preview the data
print(titanic_df.head())


# In[30]:


#see what types of data we are dealing with and how we're going to have to clean or transform this data in order to make our model
titanic_df.info()
print("----------------------------")
test_df.info()


test_df["Survived"] = -1

print("============================")
titanicANDtest_df = pd.concat([titanic_df, test_df], keys=['titanic', 'test'])


# In[31]:


# drop unnecessary columns, these columns won't be useful in analysis and prediction
titanic_df = titanic_df.drop(['PassengerId','Ticket','Cabin','Fare'], axis=1)
test_df    = test_df.drop(['Ticket','Cabin','Fare'], axis=1)


# In[32]:


float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

#$# Nov 19 edit to view the which columns in our dataframe that contain missing values
print('Here are the NAN counts of titanic_df')
#titanic_df['Age'] =  titanic_df['Age'].fillna(999)

print( titanic_df.isnull().sum(), '\n' )


# In[33]:


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
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_test  = pd.get_dummies(test_df['Sex'])
sex_dummies_test.columns = ['Female','Male']
#13April# sex_dummies_test.drop(['Male'], axis=1, inplace=True)
test_df = test_df.join(sex_dummies_test)
#$# titanic_df.drop(['Sex'],axis=1,inplace=True)

sex_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Sex'])
sex_dummies_titanicANDtest.columns = ['Female','Male']
#13April# sex_dummies_titanicANDtest.drop(['Male'], axis=1, inplace=True)
titanicANDtest_df = titanicANDtest_df.join(sex_dummies_titanicANDtest)
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


# In[34]:


print('Now from the Name we locate the MasterOrMiss passengers.')

def get_masterormiss(passenger):
    name = passenger
    if (   ('Master' in str(name))         or ('Miss'   in str(name))         or ('Mlle'   in str(name)) ):
        return 1
    else:
        return 0

titanic_df['MasterMiss'] =     titanic_df[['Name']].apply( get_masterormiss, axis=1 )

test_df['MasterMiss'] =     test_df[['Name']].apply( get_masterormiss, axis=1 )

titanicANDtest_df['MasterMiss'] =     titanicANDtest_df[['Name']].apply( get_masterormiss, axis=1 )

#$# print(titanicANDtest_df.head())
    
    

print('Here are pivot tables for survival by Sex and MasterMiss, by Pclass.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'], columns=['Pclass'],                     aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'], columns=['Pclass'],                     aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'], columns=['Pclass'],                     aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' ) 


# In[35]:


#changed child to include all passengers with the title of master in them
#this is because unlike miss, master was a term only given to male children
#so it would make sense to include all masters as children, especially since
#some passengers with the title master had no age
def get_child(passenger):
    MasterMiss, Age, Sex = passenger
    if((MasterMiss == 1 and Sex == "male") or Age < 16):
        return 1
    else:
        return 0

#applied the get_child function to the titanic and test dataframes
titanic_df["Child"] = titanic_df[["MasterMiss", "Age", "Sex"]].apply(get_child, axis = 1)
test_df["Child"] = test_df[["MasterMiss", "Age", "Sex"]].apply(get_child, axis = 1)
titanic_df.head(5)


# In[36]:


#Here I try to separate the men from the male children
#This is because when classifying males, men had a poor rate of survival
#However, male children had a good rate of survival
#So it's important to distinguish the two
def man(passenger):
    sex, child = passenger
    if(sex == "male" and child == 0):
        return(1)
    else:
        return(0)

#aaplied the man function to the titanic and test dataframes
titanic_df["Man"] = titanic_df[["Sex", "Child"]].apply(man, axis=1)
test_df["Man"] = test_df[["Sex", "Child"]].apply(man, axis=1)
titanic_df.head(5)


# In[37]:


print('Now Embarked.  Fill the 2 NaNs with S, as ticket-number blocks imply.')

titanic_df["Embarked"] = titanic_df["Embarked"].fillna("S")
titanicANDtest_df["Embarked"] = titanicANDtest_df["Embarked"].fillna("S")


embark_dummies_titanic  = pd.get_dummies(titanic_df['Embarked'])
titanic_df = titanic_df.join(embark_dummies_titanic)

embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
test_df    = test_df.join(embark_dummies_test)

embark_dummies_titanicANDtest  = pd.get_dummies(titanicANDtest_df['Embarked'])
titanicANDtest_df = titanicANDtest_df.join(embark_dummies_titanicANDtest)


print('Pivot tables for survival by Sex + MasterMiss, by Pclass + Embark.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' ) 


# In[38]:


print('Now consider Parch as a binary decision: is the value greater than 0?')

#$# def is_positive(passenger):
#$#     parch = int(passenger)
#$#     return 1 if (parch > 0) else 0

titanic_df['ParchBinary'] =   titanic_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
 
test_df['ParchBinary'] =   test_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1)
 
titanicANDtest_df['ParchBinary'] =   titanicANDtest_df[['Parch']].apply( (lambda x: int(int(x) > 0) ), axis=1) 


print('Pivot tables: Sex + MasterMiss + ParchBinary, by Pclass + Embark.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['ParchBinary', 'Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['ParchBinary', 'Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',                     index = ['ParchBinary', 'Female', 'MasterMiss'],                     columns=['Embarked', 'Pclass'],                     aggfunc=np.mean)
print( table0f.iloc[::-1],'\n' )


# In[39]:


print('Now consider SibSp as a binary decision: is the value greater than 0?')

titanic_df['SibSpBinary'] =   titanic_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)
 
test_df['SibSpBinary'] =   test_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)

titanicANDtest_df['SibSpBinary'] =   titanicANDtest_df[['SibSp']].apply( (lambda x: int(int(x) > 0) ), axis=1)


print('Pivot tables: ParchBinary + SibSpBinary + Sex + MasterMiss, by Embark + Pclass.\n')

table0d = pd.pivot_table(titanic_df, values = 'Survived',              index = ['Female', 'MasterMiss', 'SibSpBinary', 'ParchBinary'],              columns=['Pclass', 'Embarked'],              aggfunc=np.sum)
print( table0d.iloc[::-1],'\n' ) #$# This hack reverses the order of the rows.

table0e = pd.pivot_table(titanic_df, values = 'Survived',              index = ['Female', 'MasterMiss', 'SibSpBinary', 'ParchBinary'],              columns=['Pclass', 'Embarked'],              aggfunc='count')
print( table0e.iloc[::-1],'\n' )

table0f = pd.pivot_table(titanic_df, values = 'Survived',              index = ['Female', 'MasterMiss', 'SibSpBinary', 'ParchBinary'],              columns=['Pclass', 'Embarked'],              aggfunc=np.mean )
print( table0f.iloc[::-1].round(2),'\n' )

def get_family(passenger):
    SibSpBinary, ParchBinary = passenger
    return 1 if(SibSpBinary == 1 or ParchBinary ==1) else 0

titanic_df["Family"] = titanic_df[['SibSpBinary', 'ParchBinary']].apply(get_family, axis=1)
test_df["Family"] = test_df[['SibSpBinary', 'ParchBinary']].apply(get_family, axis=1)

def get_family_size(passenger):
    SibSp, Parch = passenger
    return (SibSp + Parch + 1)

titanic_df["FamilySize"] = titanic_df[['SibSp', 'Parch']].apply(get_family_size, axis=1)
test_df["FamilySize"] = test_df[['SibSp', 'Parch']].apply(get_family_size, axis=1)


# In[40]:


#goal of this cell is to combine pclass and sex to make a more accurate predictor
#since the two seem to work better when combined (thanks to Erik Bruin for the idea)

titanic_df["ClassSex"] = titanic_df["Pclass"].map(str) + titanic_df["Sex"]
test_df["ClassSex"] = test_df["Pclass"].map(str) + test_df["Sex"]

classSex_dummies_titanic = pd.get_dummies(titanic_df["ClassSex"])
classSex_dummies_test = pd.get_dummies(test_df["ClassSex"])

titanic_df = titanic_df.join(classSex_dummies_titanic)
test_df = test_df.join(classSex_dummies_test)


# In[41]:


#now to separate the male children from the female children in order to classify each of them
titanic_df["ChildSex"] = titanic_df["Child"].map(str) + titanic_df["Sex"]
test_df["ChildSex"] = test_df["Child"].map(str) + test_df["Sex"]

#separating into adults male and female and girls and boys
child_dummies = pd.get_dummies(titanic_df["ChildSex"])
child_dummies.columns = ["AdultF","AdultM","Girl","Boy"]
titanic_df = titanic_df.join(child_dummies)
#drop the adults and girls because those aren't what we are looking for
titanic_df.drop(["AdultF","AdultM","Girl","ChildSex"], axis=1, inplace = True)

#do the same thing for the test dataframe
child_dummies_test = pd.get_dummies(test_df["ChildSex"])
child_dummies_test.columns = ["AdultF","AdultM","Girl","Boy"]
test_df = test_df.join(child_dummies_test)
test_df.drop(["AdultF","AdultM","Girl","ChildSex"], axis=1, inplace = True)

#now that we have Boys separated, we want to sort them by their individual classes
titanic_df["BoyClass"] = titanic_df["Boy"].map(str) + titanic_df["Pclass"].map(str)
test_df["BoyClass"] = test_df["Boy"].map(str) + test_df["Pclass"].map(str)

#get dummies for people who are boys in classes 1, 2, and 3
boy_dummies_titanic = pd.get_dummies(titanic_df["BoyClass"])
boy_dummies_titanic.columns = ["girl1","girl2","3rdGirls","1stBoys", "2ndBoys", "3rdBoys"]
titanic_df = titanic_df.join(boy_dummies_titanic)
#drop every other column that doesn't involve boys
titanic_df.drop(["girl1","girl2","3rdGirls","BoyClass"], axis = 1, inplace = True)

#do the same thing but on the test dataframe
boy_dummies_test = pd.get_dummies(test_df["BoyClass"])
boy_dummies_test.columns = ["girl1","girl2","3rdGirls","1stBoys", "2ndBoys", "3rdBoys"]
test_df = test_df.join(boy_dummies_test)
test_df.drop(["girl1","girl2","3rdGirls","BoyClass"], axis = 1, inplace = True)


# In[42]:


#myMean = titanic_df["Fare"].mean()
#print(myMean)
#test_df["Fare"] = test_df["Fare"].fillna(myMean)

#here is where we drop all the columns that will not be useful in our prediction, given more time, I would like to do something with age
#I plan on doing that in the future, because I think there is something useful there.
titanic_df.drop(["Name","Sex","Age","Embarked","Male","Parch", "SibSp","ParchBinary","SibSpBinary","Pclass","Class1","Class2","Class3","ClassSex","Female","MasterMiss","S","1male","2male","3male","Boy"], axis=1, inplace=True)

test_df.drop(["Name","Sex","Age","Embarked","Male", "Survived","Parch","SibSp","ParchBinary","SibSpBinary","Pclass","Class1","Class2","Class3","ClassSex","Female","MasterMiss","S","1male","2male","3male","Boy"], axis=1, inplace=True)


titanic_df.head()
# test_df.head()


# Here are where I create the predictive models that will be used to make the prediction. The models chosen are logistic regression, K-nearest neighbors, random forest
# support vector machines and xgboost. I compiled them all together and used a voting prediciton model to make the final prediction.

# In[43]:


X_train = titanic_df.drop("Survived", axis=1)
Y_train = titanic_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()


# In[44]:


# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y1_pred = logreg.predict(X_test)

logreg.score(X_train, Y_train)


# In[45]:


#K Nearest Neighbors with 7 neighbors
knn = KNeighborsClassifier(n_neighbors = 7)

knn.fit(X_train, Y_train)

Y2_pred = knn.predict(X_test)

knn.score(X_train, Y_train)


# In[46]:


#Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y3_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)


# In[47]:


#Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y4_pred = svc.predict(X_test)

svc.score(X_train, Y_train)


# In[48]:


#Naive Bayes wasn't useful in this prediction

#gaussian = GaussianNB()

#gaussian.fit(X_train, Y_train)

#Y5_pred = gaussian.predict(X_test)

#gaussian.score(X_train, Y_train)


# In[49]:


#XGBoost
xgb = XGBClassifier(base_score = 0.5, booster='gbtree')

xgb.fit(X_train, Y_train)

Y6_pred = xgb.predict(X_test)

xgb.score(X_train, Y_train)


# In[50]:


#voting classifier (hard voting)
vcr=VotingClassifier(estimators=[('lg',logreg),('xgb',xgb),('rf',random_forest),('knn',knn),('svc',svc)],voting='hard', weights = [1,2,3,2,1])

vcr.fit(X_train, Y_train)

Y_pred = vcr.predict(X_test)

print("voting score",vcr.score(X_train,Y_train))


# In[51]:


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(titanic_df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df


# In[52]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
#        "Survived": test_df["Survived"]
    })
submission.to_csv('titanic.csv', index=False)

