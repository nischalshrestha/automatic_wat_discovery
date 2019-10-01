#!/usr/bin/env python
# coding: utf-8

# #                       Detailed and In-depth analysis of Titanic [0.85074]

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd
import math 
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().magic(u'matplotlib inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


train.describe()


# In[ ]:


df = train.copy()


# In[ ]:


df.head()


# In[ ]:


# from train.describe() it is evident that only 38.38 % of the population on the ship survived , rest died


# In[ ]:


df.loc[(df.Survived == 1) & (df.Sex == "male") , :].count()


# In[ ]:


# there were 109 males across the ship who survived that accident


# In[ ]:


df.loc[(df.Survived == 1) & (df.Sex == "female") , :].count()


# In[ ]:


# there were 233 females across the ship who survived that accident
# look the following graph


# In[ ]:


sns.factorplot(x="Sex",col="Survived", data=df , kind="count",size=4, aspect=.7);


# In[ ]:


# this gives us the idea that males died more and females survived more


# In[ ]:


# similarly


# In[ ]:


sns.factorplot(x="Sex", hue = "Pclass" , col="Survived", data=df , kind="count",size=6, aspect=.7);


# #### overall the males and females of Pclass 3 died more than others
# #### the males of Pclass 3 showed a remarkable increase in death and shoots the graph up , same goes to the females in
# #### same goes to the females in survived = 0
# #### in survived = 0 , showing increasing trend in death as class shifts down

# #### In survived = 1 females showed a near fall down trend as expected but pclass=2 females survived less than the Pclass=3 females

# #### But the males on contrary showed a dip in between i.e. 
# #### in males who survived , Plass -->  3 > 1 > 2
# 
# 
# #### i.e Survived Pclass=3 males survived more than the survived Pclass=1 males and survived Pclass=2 males
# #### the above is evident from the following inspection
# #### although survived male Plass = 3 is slightly greater than survived male Plass = 1

# In[ ]:


df.loc[(df.Survived == 1) & (df.Sex == "male") & (df.Pclass == 1)].count()


# In[ ]:


df.loc[(df.Survived == 1) & (df.Sex == "male") & (df.Pclass == 2) , :].count()


# In[ ]:


df.loc[(df.Survived == 1) & (df.Sex == "male") & (df.Pclass == 3) , :].count()


# In[ ]:


pd.crosstab(df.Pclass, df.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


# All in all including both the sexes 2nd class survived less than the other two clases


# In[ ]:


df.Survived[df.Pclass == 1].sum()/df[df.Pclass == 1].Survived.count()


# In[ ]:


df.Survived[df.Pclass == 2].sum()/df[df.Pclass == 2].Survived.count()


# In[ ]:


df.Survived[df.Pclass == 3].sum()/df[df.Pclass == 3].Survived.count()


# In[ ]:


# % survived in Pclass = 1  --> 62.96 %  , similarly calculated for others


# In[ ]:


sns.factorplot(x='Pclass',y='Survived', kind="point" ,data=df)


# In[ ]:


sns.factorplot('Pclass','Survived',kind="bar",hue='Sex',data=df)


# In[ ]:


# A cross-tabulation to further inspect


# In[ ]:


pd.crosstab([df.Sex, df.Survived], df.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


# Almost all women in Pclass 1 and 2 survived and nearly all men in Pclass 2 and 3 died


# In[ ]:


# lets see how survivals varies with Embarked


# In[ ]:


sns.factorplot(x="Survived",col="Embarked",data=df ,hue="Pclass", kind="count",size=5, aspect=.7);


# In[ ]:


# this shows that those who were embarked S survived more than those who were survived C and then Q
# Most of the people who died were embarked S


# In[ ]:


# Also , people survived with embarked Q were mostly from Plass 3 females


# In[ ]:


# A more closer look with cross-tab


# In[ ]:


pd.crosstab([df.Survived], [df.Sex, df.Pclass, df.Embarked], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


# can also be viewed like this


# In[ ]:


plt.subplots(figsize = (10,5))
plt.title('Embarked vs Survived wih Sex')
sns.violinplot(x = "Survived", y = "Embarked", hue = "Sex",data = df)
plt.show()


# In[ ]:


# similarly with Pclass

sns.factorplot(x = "Survived", y = "Pclass",col = "Embarked" , hue = "Sex" , kind = "violin",data = df)


# In[ ]:


sns.factorplot(x="Sex", y="Survived",col="Embarked",data=df ,hue="Pclass",kind="bar",size=5, aspect=.7);


# In[ ]:


# Inferences from above graph

# the survived axis shows the % .
# which means embarked Q males in Pclass 1 and 2 were all died

# while embarked females in Pclass 1 and 2 all lived....
# also nearly Pclass 1 and 2 females of all embarked types lived


# In[ ]:


context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df['Sex_bool']=df.Sex.map(context1)
df["Embarked_bool"] = df.Embarked.map(context2)


# In[ ]:


df.head()


# In[ ]:


correlation_map = df[['PassengerId', 'Survived', 'Pclass', 'Sex_bool', 'Age', 'SibSp',
       'Parch', 'Fare' , 'Embarked_bool']].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(12,12)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)


# ### The above heatmap shows the overall picture very clearly 
# 
# ###  PassengerId is a redundant column as its very much less related to all other attributes , we can remove it .
# 
# ###  Also , Survived is related indirectly with Pclass and also we earlier proved that as Pclass value increases Survival decreases
# 
# ###  Pclass and Age are also inversely related and can also be proven by the following cell  that as Pclass decreases , the mean of the Age increases ,  means the much of the older travellers are travelling in high class .
#               
# ###  Pclass and fare are also highly inversely related as the fare of Pclass 1 would obviously be higher than corresponding Pclass 2 and 3 .
# ###  Also , people with lower ages or children are travelling with their sibling and parents more than higher aged people (following an inverse relation) , which is quite a bit obvious .
# ###  Parch and SibSp are also highly directly related
# ###  Sex_bool and Survived people are highly inversely related , i.e. females are more likely to survive than men

# In[ ]:


df.groupby("Pclass").Age.mean()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


for x in [train, test,df]:
    x['Age_bin']=np.nan
    for i in range(8,0,-1):
        x.loc[ x['Age'] <= i*10, 'Age_bin'] = i


# In[ ]:


df[["Age" , "Age_bin"]].head(10)


# In[ ]:


sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex',kind="bar", data=df)


# In[ ]:


sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex', kind="violin", data=df)


# In[ ]:


pd.crosstab([df.Sex, df.Survived], [df.Age_bin, df.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


#  All female in Pclass 3 and Age_bin = 5 died.
#  Males in Age_bin >= 2 and Pclass died more than survived or died greater than 50% .


# In[ ]:


sns.factorplot('SibSp', 'Survived', col='Pclass' , row = 'Sex', data=df )


# In[ ]:


#  Females in Pclass 1 and 2 with siblings upto 3 nearly all survived


# In[ ]:


#  For Pclass 3 , males and females showed a near decreasing trend as number of siblings increased .


# In[ ]:


#  For males, no survival rate above 0.5 for any values of SibSp. (less than 50 %)


# In[ ]:


pd.crosstab([df.Sex, df.Survived], [df.Parch, df.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


#  For males,all survival rates below 0.5 for any values of Parch, except for Parch = 2 and Pclass = 1.


# In[ ]:


sns.factorplot('Parch', 'Survived', col='Pclass' , row = 'Sex', kind="bar", data=df )


# In[ ]:


# the distribution of Age_bin , SibSp and Parch as follows


# In[ ]:


for x in [train, test , df]:
    x['Fare_bin']=np.nan
    for i in range(12,0,-1):
        x.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i


# In[ ]:


fig, axes = plt.subplots(4,1)
fig.set_size_inches(20, 18)
sns.kdeplot(df.SibSp , shade=True, color="red" , ax= axes[0])
sns.kdeplot(df.Parch , shade=True, color="red" , ax= axes[1])
sns.kdeplot(df.Age_bin , shade=True, color="red" , ax= axes[2])
sns.kdeplot(df.Fare , shade=True, color="red" , ax= axes[3])
plt.show()


# ###  Maximum people are with no siblings travelling
# ###  more people were travelling with only their 1 parent rather than 2 
# ###  maximum population on the ship was aged between 15 yrs to 50 yrs.
# ###  most of the people only paid upto 50 as their fare

# In[ ]:


# introducing Fare_bin the same way as done in the Age_bin above but with a gap of 50


# In[ ]:


df[["Fare" , "Fare_bin"]].head(10)


# In[ ]:


pd.crosstab([df.Sex, df.Survived], [df.Fare_bin, df.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


sns.factorplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', data=df)
plt.show()


# In[ ]:


df_test = test.copy()


# In[ ]:


df_test.head()


# In[ ]:


df.drop(['PassengerId','Sex','Embarked','Name','Ticket', 'Cabin', 'Age', 'Fare'],axis=1,inplace=True)
df.head()


# In[ ]:


context1 = {"female":0 , "male":1}
context2 = {"S":0 , "C":1 , "Q":2}
df_test['Sex_bool']=df_test.Sex.map(context1)
df_test["Embarked_bool"] = df_test.Embarked.map(context2)
df_test.drop(['PassengerId','Sex','Embarked','Name','Ticket', 'Cabin', 'Age', 'Fare'],axis=1,inplace=True)
df_test.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


#  Age_bin in both dataframes is still possessing null values


# In[ ]:


df_test.Age_bin.fillna(df_test.Age_bin.mean() , inplace=True)


# In[ ]:


df.Age_bin.fillna(df.Age_bin.mean() , inplace=True)


# In[ ]:


df.Embarked_bool.fillna(df.Embarked_bool.mean() , inplace=True)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), df['Survived'], test_size=0.3, random_state=101)


# In[ ]:


MLA = []
x = [LinearSVC() , DecisionTreeClassifier() , LogisticRegression() , KNeighborsClassifier() , GaussianNB() ,
    RandomForestClassifier() , GradientBoostingClassifier()]

X = ["LinearSVC" , "DecisionTreeClassifier" , "LogisticRegression" , "KNeighborsClassifier" , "GaussianNB" ,
    "RandomForestClassifier" , "GradientBoostingClassifier"]

for i in range(0,len(x)):
    model = x[i]
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    MLA.append(accuracy_score(pred , y_test))


# In[ ]:


MLA


# In[ ]:


sns.kdeplot(MLA , shade=True, color="red")


# In[ ]:


#  this proves that much of the algorithms are giving the accuracy between 77 % to 80 % with some above 80 % .
#  thats a pretty much good estimation 


# In[ ]:


d = { "Accuracy" : MLA , "Algorithm" : X }
dfm = pd.DataFrame(d)


# In[ ]:


# making a dataframe of the list of accuracies calculated above


# In[ ]:


dfm   # a dataframe wilh all accuracies and their corresponding algorithm name


# In[ ]:


sns.barplot(x="Accuracy", y="Algorithm", data=dfm)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived',axis=1), df['Survived'], test_size=0.3, random_state=66)
model = KNeighborsClassifier(n_neighbors=6)
model.fit( X_train , y_train )


# In[ ]:


pred = model.predict(X_test)


# In[ ]:


answer = model.predict(df_test)


# In[ ]:


print (accuracy_score(pred , y_test))


# # So , the accuracy turns out to be 85.074 % with n-neighbors = 6,
# # lets check for other n-neighbors .

# In[ ]:


#  lets check it till 30 neighbours that which has got the maximum accuracy score

KNNaccu = []
Neighbours = []

for neighbour in range(1,31):
    model = KNeighborsClassifier(n_neighbors=neighbour)
    model.fit( X_train , y_train )
    pred = model.predict(X_test)
    KNNaccu.append(accuracy_score(pred , y_test))
    Neighbours.append(neighbour)


# In[ ]:


d = { "Neighbours" : Neighbours , "Accuracy" : KNNaccu }
knndf = pd.DataFrame(d)


# In[ ]:


knndf.head()


# In[ ]:


sns.factorplot(x="Neighbours", y="Accuracy",size = 5 , aspect = 2 , data=knndf)


# ###  This states that for Neighbours = 6 , the accuracy is the maximum  .

# In[ ]:


#  making a csv file of the predictions


# In[ ]:


d = { "PassengerId":test.PassengerId , "Survived":answer }
final = pd.DataFrame(d)
final.to_csv( 'titanic_again.csv' , index = False )


# # Please upvote if you like it...

# In[ ]:




