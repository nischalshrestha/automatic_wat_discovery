#!/usr/bin/env python
# coding: utf-8

# **Table of contents **
# 
# * [1. Introduction](#Introduction)
#     *     [1.1 Preparing my toolkit](#Loading)
#     *     [1.2 Load and explore the datasets](#Explore)
#     *     [1.3 Data visualization](#Visualization)
# * [2. Working on our data](#Data)
#     *     [2.1 Imputing missing Values](#Missing)
#     *     [2.2 Creating new features](#Features)
# * [3. Machine Leaning](#Learning)
#     *    [3.1 Our classifier : K neighbors classifier ](#Classifier)
#     *    [3.2  Hyperparameters tuning and cross validation](#Best)
#     *    [3.3 How good is our model ?](#Result) 
#     *    [3.4 Submitting our Result !](#Submit) 
# 

# <a id="Introduction">1. Introduction</a>
# 
# Hello every one ! Welcome on board ! My name is Baptiste and I am the Cruise ship Captain. 
# During this trip, we will make several stopovers ! We will first analyze the dataset and try to shed light on the correlations between features . Then we will work on our data, building a strategy to fill missing values and transform raw data into new features that better represent the underlying problem. Eventually, we will use a simple predictive model Knclassifier and find the paremeters which achieve the best prediction.
# 
# As this is my first journey through an ocean of data, i am looking forward to hearing your feedback and improving my navigation skills ! 
# 
# 

# <a id="Loading">1.1  Preparing my Toolkit </a>
# 
# In order to sail through this dataset, we need to be well equiped ! From exploratory data analysis to machine learning, we will use the following libraries : 
# - Data Manipulation : numpy, pandas
# - Data visualization : matplotlib.pyplot and seaborn
# - Feature engineering : random ( to generate random numbers) and re ( to match regular expressions)
# - Machine learning : sklearn.kneigbors ( our classifier ), sklearn.preprocess ( to prepare the data for our model ),sklearn.mode_selection ( to find the best parameters), sklearn.metrics ( to evaluate our model)

# In[84]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # Data manipulation
import pandas as pd # Data manipulation

import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Data visualization


import random # generate random numbers
import re # match regular expression
from collections import Counter
sns.set_style('whitegrid') # Set plots' style 
get_ipython().magic(u'matplotlib inline')

# machine learning toolkit
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# <a id="Explore">1.2 Load and explore the Datasets </a>
# 
# We load the train.csv and test.csv file and concatenate them ( we will separate them before training our model ). The more data we have, the better our analys will be ! 
# 

# In[85]:


df_train=pd.read_csv('../input/train.csv')
df_test=pd.read_csv('../input/test.csv')
df_titanic=pd.concat([df_train,df_test]).reset_index(drop=True)
df_titanic.info()


# The dataset contains 1309 entries and 12 features which are : 
# - Age : Age of passengers ( there are some missing values)
# - Cabin : Passengers' cabin ( More than 1000 missing values)
# - Embarked : Port of embarkation  ( 2 missing values)
# - Fare : Tickets price ( 1 missing value )
# - Name : Passengers name
# - Parch : Number of parents and children 
# - Pclass : Passengers social class
# - SibSp: Number of siblings and spouses
# - Survived : Survivor or not ( missing values belong to the test set )
# 
# Let's take a first glimpse to the dataset !

# In[86]:


df_titanic.head()


# We can see that only 38% of the passengers from the train.csv file have survived the sinking of the Titanic. The mean age is 30 years old. 

# In[87]:


df_titanic.describe()


# <a id="Visualization">1.3 Data visualization</a>
# 
# Now that we had a first feeling about the distribution of numerical values, we will visualize our data. 
# It is well known that when Titanic sank, the safety of women and children came first.  As we can see on the following charts, there is an important correlation between the Sex and the survival of passengers. In the train.csv. file, less than 20% of men survived the sinking. However on the right plot, we can see that children where more likely to survive.

# In[88]:


fig, axs = plt.subplots(1,2,figsize=(10,5))
axs[0].set_title("Survival rate vs Sexe")
sns.barplot(x="Sex",y="Survived",data=df_titanic,ax=axs[0])
axs[1].set_title("Survival rate vs Sexe vs Age")
sns.swarmplot(x="Sex",y="Age",data=df_titanic,hue="Survived",ax=axs[1])


# Indeed, the survival rate of men increases when we look at the passengers under 16. As child are more likely to survive, we will create a new value in the Sex feature. The value " Child" will represent passenger under 16. But first, we have to fill the missing values for the Age feature !

# In[89]:


df_child=df_titanic[df_titanic["Age"]<=16]
ax=plt.axes()
ax.set_title('Survival Rate vs Sexe')
sns.barplot(x="Sex",y="Survived",data=df_child)


# In[90]:


sns.factorplot(x="Pclass",y="Survived",data=df_titanic)


#  <a id="Data">2. Working on our data</a>
# *      <a id="Missing">2.1 Imputing missing values</a>
# 
# We will fill missing values of the Age feature by imputing random numbers between the mean and standard deviation. First we will drop missing values to calculate mean and standard deviation.

# In[91]:


df_age=df_titanic["Age"].dropna()
mean_age=int(df_titanic["Age"].dropna().mean())
std_age=int(df_titanic["Age"].dropna().std())

print("Mean Age :{}".format(mean_age))
print("Age standard deviation: {}".format(std_age))


# The random numbers will be generated between 15 and 43 and applied to the Age column using the following lambda function. 

# In[92]:


df_age=df_titanic["Age"].dropna()
mean_age=int(df_titanic["Age"].dropna().mean())
std_age=int(df_titanic["Age"].dropna().std())
df_titanic["Age"]=df_titanic["Age"].apply(lambda x: np.random.randint(int(mean_age-std_age),int(mean_age+std_age)) if pd.isnull(x) else x)


# We visualize the Age's feature new distribution.

# In[93]:


fig,axs=plt.subplots(1,2,figsize=(10,5))
axs[0].set_title("Age Distribution before imputing missing values")
sns.distplot(df_age,ax=axs[0])
axs[1].set_title("Age Distribution after imputing missing values")
sns.distplot(df_titanic["Age"],ax=axs[1])


# 

# In[94]:


f, ax = plt.subplots(1, 1)
sns.distplot(df_titanic[df_titanic["Survived"]==1]["Age"],label='Survivor',hist=False)
sns.distplot(df_titanic[df_titanic["Survived"]==0]["Age"],label='Not Survivor',hist=False)
ax.legend()


# But how will we fill the two missing values for the Embarked feature ? I suppose that the ticket fare is correlated with the port of embarkation. Let's have a look :

# In[95]:


sns.factorplot(x="Embarked",y="Fare",data=df_titanic)


# It seems that the ticket fare was around 30 for Southampton, 60 for Cherbourg and 15 for Queenstown. Now, let's have look at the ticket cost for the two missing values. The Fare is 80 for both ticket, so I supose that they embarked in Cherbourg.

# In[96]:


print(df_titanic[pd.isnull(df_titanic["Embarked"])])


# In[97]:


df_titanic["Embarked"]=df_titanic["Embarked"].fillna("C")


# There is one missing value for the Ticket Fare. The passenger embarked in Southampton, so we will impute a random number between the mean and standard deviation of Southampton's ticket Fare.

# In[98]:


print(df_titanic[pd.isnull(df_titanic["Fare"])])
mean=df_titanic[df_titanic["Embarked"]=='S']["Fare"].mean()
std=df_titanic[df_titanic["Embarked"]=='S']["Fare"].std()
df_titanic["Fare"]=df_titanic["Fare"].fillna(random.randint(int(mean-std),int(mean+std)))


#  <a id="Features">2.2. Creating new features</a>
#  
#  We will now create new features which better represent our problem. 

# Once we have dealt with missing values in the Age feature. We will create two new categories : 
# - "Child" will represent people under 16 
# - "Mother" will represent women over 16 with at least 1 children or Parent on board ( Parch > 0)

# In[99]:


df_titanic.loc[df_titanic["Age"]<=16,"Sex"]="Child"
df_titanic.loc[(df_titanic["Age"]>16) & (df_titanic["Parch"]>0) & (df_titanic["Sex"]=="female"),"Sex"]="Mother"


# In[100]:


sns.barplot(x="Sex",y="Survived",data=df_titanic) 


# Do people with relatives on board have a higher survival rate ? In order to answer this question, we are going to add upp the "Parch" and "SibSp" features.

# In[101]:


fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].set_title('Survival Rate vs Parch')
sns.barplot(x="Parch",y="Survived",data=df_titanic,ax=axs[0])
axs[1].set_title('Survival Rate vs SibSp')
sns.barplot(x="SibSp",y="Survived",data=df_titanic,ax=axs[1])
df_family=df_titanic["Parch"]+df_titanic["SibSp"]
axs[2].set_title('Survival Rate vs Parch + SibSp')
sns.barplot(x=df_family,y=df_titanic["Survived"],ax=axs[2])


# In[102]:


df_titanic["Parch"].value_counts()


# In[103]:


df_titanic["SibSp"].value_counts()


# On the last chart, we can see that people with a number of relatives between 1 and 3 have a higher chance of survival. In order to simplify our model, we create a feature "Family" which contains two categories : 
# - 1 : People with a number of relatives between 1 and 3
# - 0 : Individuals or people with a number of relatives greater than 3

# In[104]:


df_titanic["family"]=df_titanic["Parch"]+df_titanic["SibSp"]
df_titanic.loc[(df_titanic["family"]>=1) & (df_titanic["family"]<=3) ,"family"]=1
df_titanic.loc[df_titanic["family"] >3 | (df_titanic["family"]==0) ,"family"]=0


# The survival rate is around 55% for people with a number of relatives between 1 and 3.

# In[105]:


ax=plt.axes()
ax.set_title('Survival rate vs family size')
sns.barplot(x="family",y="Survived",data=df_titanic)


# Last but not least, we will look a the passengers name ( more precisely at the title before their name). Maybe Doctors or Lords had a better chance to survive. In order to do so, we will extract the title by using the extract method and a regular expression. We can see that they were at least 8 doctors (Dr.) and 1 Countess ( Countess.) on board.

# In[106]:


df_name=pd.DataFrame(df_titanic["Name"].str.extract('([A-Za-z]+\.)'))
print(df_name["Name"].value_counts())


# We will group names in four categories : Mr, Mrs, Miss and VIP. 
# "Miss" means that she is unmarried, "Mrs" means that she is married 
# The VIP category represents people which have a higher title such as "Capt." or "Countess."

# In[107]:


VIP=["Master.","Rev.","Dr.","Col.","Major.","Jonkheer.","Dona.","Capt.","Don.","Sir.","Lady.","Countess."]
df_name.loc[df_name["Name"].isin(VIP),"Name"]="VIP"
df_name.loc[df_name["Name"].isin(["Mlle.","Ms."]),"Name"]="Miss."
df_name.loc[df_name["Name"]=="Mme.","Name"]="Mrs."


# In[108]:


df_titanic["Name"]=df_name["Name"]


# In[109]:


ax=plt.axes()
ax.set_title("Survival rate vs title")
sns.barplot(x="Name",y="Survived",data=df_titanic)


# The first character of the "Cabin" feature is a letter. It may represents the deck where the cabin is located. There is maybe a correlation between this letter and the survival rate but there are too many missing values.

# In[110]:


df_cabin=pd.DataFrame(df_titanic[["Cabin","Pclass","Fare"]].dropna())
df_cabin["Cabin"]=df_cabin["Cabin"].astype(str).str[0]


# In[111]:


print(df_cabin["Cabin"].value_counts())


# In[112]:


sns.factorplot(x="Cabin",y="Fare",data=df_cabin)
sns.factorplot(x="Cabin",y="Pclass",data=df_cabin)


# <a id="Learning">3. Machine Learning</a>
# 
# Let's do some machine learning ! First we have to split the training and the testing set.

# In[113]:


df_train=df_titanic.loc[0:890,]
df_test=df_titanic.loc[891:1308,]


# We drop the "PassengerId", "Cabin","Ticket","Survived","SibSp","Parch" columns and assign the remaining columns in the X variable. The target variable "Survived" is assigned to the y variable. We apply the same transormations on the data set we use to submit our result (Submission set). 

# In[114]:


X=df_train.drop(["PassengerId","Cabin","Ticket","Survived","SibSp","Parch"],axis=1)
y=df_train["Survived"]
Submission_set=df_test.drop(["PassengerId","Cabin","Ticket","Survived","SibSp","Parch"],axis=1)


# <a id="Classifier">3.1 K Neighbors classifier   </a>
# 
# As the KNeighborsClassifier works with distances, we have to convert categorical variables into dummy variables.

# In[115]:


X=pd.get_dummies(X)
Submission_set=pd.get_dummies(Submission_set)
X.info()
X.describe()


# 
# The k-nearest neighbors algorithm will classify ( survived or not survived ) the passenger of our testing set by looking at the most common class assigned to the passenger's k nearest neigbors. We can see that the predictor variable "Fare" ranges from 0 to 512, whereas the predictor variable "family" is a binary variable : 0 or 1. As the k-NN classifier relies on distance between data points, it would focus unfairly on variables with larger ranges such as the "Fare" variable. That's why we have to scale our data before training our model.

# In[116]:


X=scale(X)
Submission_set=scale(Submission_set)


# If we train our model on the training set and evaluate its accuracy on the same data, our model would likely perform better than it actually does on unseen data. In order to avoid overfitting, we will split our train set in two subsets : 80% of the training set will be used for training and 20% will be used for testing our model. We use the train_test_split function to generate those subsets. 

# In[117]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# Then we build our Kneighbors classifier. We will try to find the parameters which achieve the best prediction. We are going to tune two hyperparameters : 
# - Number of neigbors : range from 1 to 50
# - Metric : cityblock or euclidean 
# 

# In[118]:


Neighbors=KNeighborsClassifier()
n_neighbors = np.arange(1,50)
param_grid = {'n_neighbors': n_neighbors, 
             'metric':["cityblock","euclidean"]}


# <a id="B">3.2 Cross Validation and hyperparameters tunning</a>
# 
# We have to try all possible combinations, fit all of them to our training subset and choose the best performing one. This task is done by the GridSearchCV function. The grid search exhaustively generates combinations from a grid of parameter values specified in the param_grid dictionnary. 
# We use cross validation in order to avoid overfitting of the parameters on the test set. For a k folds cross validation, the training set is split into k smallers set. Eventually, the model is trained using k-1 of the folds as training data, and validated on the remaining part of the data. 
# The performance of each combination is the average score for each k-fold validation. The best parameters maximize this performance ! 
# 
# As we can see, the best combination is the metric "cityblock" and 12 neighbors. 
# The score of this combination is 0.83 !

# In[ ]:


Neighbors_cv = GridSearchCV(Neighbors,param_grid, cv=5)
Neighbors_cv.fit(X_train,y_train)

print("Tuned k-NN classifier Parameters: {}".format(Neighbors_cv.best_params_)) 
print("Best score is {}".format(Neighbors_cv.best_score_))


# Once we found the best paremeters,  it's time to test our model on unseen data and evaluate our result !

# In[ ]:


y_pred=Neighbors_cv.predict(X_test)


# <a id="Result">3.3 How good is our model ? </a>
# 
# We calculate the precision and recall  of our model from the confusion matrix.
# 
# Confusion Matrix : 
# <table>
# <tr><th></th><th>Predicted : Not survivor</th><th> Precited : Survivor </th></tr>
# <tr><td> Actual : Not survivor </td> <td> True Positive </td> <td> False negative </td></tr>
# <tr><td> Actual : Survivor </td> <td> False Positive </td> <td> True negative</td></tr>
# </table>
# 
# The precision is :  $ \frac {True positive}{ True positive + False positive } $. A high precision means  that not many survivor were predicted as dead. 
# 
# The recall is : $ \frac {True positive}  {True positive + False negative } $. A high recall means that our classifier predicted most of the passengers death correctly. 
# 
# The f1 score is the harmonic mean of precision and recall. 

# In[ ]:


print("Confusion Matrix :")
print(confusion_matrix(y_pred,y_test))
print("Classification report :")
print(classification_report(y_pred,y_test))


# Our model has an average precision of 82% and recall of 81% ! 

# <a id="Submit">3.4 Submitting our result !</a>
# 
# Eventually, we predict the outcome on the test file and submit our result ! Thank you for reading ! I am looking forward to hearing your feedback ! 

# In[ ]:


Sub_pred=Neighbors_cv.predict(Submission_set).astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": Sub_pred
    })
submission.to_csv('titanic.csv', index=False)

