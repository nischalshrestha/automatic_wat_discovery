#!/usr/bin/env python
# coding: utf-8

# Hello All,
# 
# Titanic dataset is one of the most famous ways to start learning and practicing Data Science. As already said somewhere the best way to learn and to explain the process to others, thus, what better way of doing that, than creating a Kernel of Titanic.
# 
# The main objective is to provide an overview of steps and processes one should have when approaching a machine learning problem, trying to provide all the necessary information and mainly clarifying the doubts I had myself when I started this analytics journey.
# 
# Hope you like it.
# ... AND please **UPVOTE** in case you like it or **COMMENT** for any question of problem.
# 
# GOOD WORK!
# 
# **INDEX:**
# 
# 1) Exploratory Data Analysis:
#     - Analysis of the variables
#     - Perception of relation between variabbles and with to target variable
#     - Dealing with missing values
# 2) Data Cleaning and Features Engineering
#     - Creation of new variables relevant for modelling
#     - Convert continuous values and string into categorical values
#     - Check correlations and Multicollinearity
# 3) Modelling
#     - Creation of the test and train sets
#     - Modelling different statistical and machine learning techniques
# 4) Validation
#     - Model Selection and Cross Validation
#     - Confusion Matrix
#     - Feature Importance for different models

# **PART 1 - EXPLORATORY DATA ANALYSIS**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Importing the data in CSV format
data='../input/train.csv'
data = pd.read_csv(data) 


# In[ ]:


data.head()
#This will provide you brief information about how your data is. By default head and tails presents the 5 first/last rows
#If you whant to see more/less than 5, you just need to put number inside the brackets.


# In[ ]:


data.info()
#This will provide you info about the columns format, number of filled rows, and columns names


# Here you can easily see that there is 891 entries, the dtypes, and that Age, Cabin and Embarked variables have missing values

# PART 1 - EXPLORATORY DATA ANALYSIS

# In[ ]:


# Survived - Target Variable

f,ax=plt.subplots(1,2,figsize=(10,5))
data['Survived'].value_counts().plot(kind='pie',explode =[0,0.1],autopct='%1.1f%%',ax=ax[0])
ax[0].set_title('Pie')
sns.countplot('Survived',data=data,ax=ax[1])
ax[1].set_title('Bar')


# Here we can see that almost 40% of individuals in our dataset did not survived.
# 
# It's now time to explore the dependent variables.

# **PCLASS**

# In[ ]:


data.groupby(['Pclass'])['Survived'].mean()


# In[ ]:


pd.crosstab(data.Pclass,data.Survived,margins=True)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
pd.crosstab(data.Pclass,data.Survived).plot(kind='bar',ax=ax[0])
ax[0].set_title('Survival per Pclass')
sns.factorplot('Pclass','Survived',data=data,ax=ax[1])
ax[1].set_title('Survival per Pclass in %')
plt.close(2)


# Comments:
# 
# -Chance of survival for class 1 with almost 65%, near 50% for class2 and almost none for class 3.
# 
# -Meaning the social status matter in this analysis
# 
# -Most part of poor people didn't survived
# 
# -Important variable for the model

# **SEX**

# In[ ]:


data.groupby(['Sex'])['Survived'].mean()


# In[ ]:


pd.crosstab(data.Sex,data.Survived,margins=True)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(10,5))
pd.crosstab(data.Sex,data.Survived).plot(kind='bar',ax=ax[0])
ax[0].set_title('Survival per Sex')
sns.factorplot('Sex','Survived',data=data,ax=ax[1])
ax[1].set_title('Survival per Sex in %')
plt.close()


# Comments:
# 
# -Chance of survival is quite bigger for woman that for men
# 
# -Important variable for the model
# 
# Lets now try to check result for Pclass and Sex together.

# **SEX AND PCLASS**

# In[ ]:


data.groupby(['Sex','Pclass'])['Survived'].mean()


# In[ ]:


pd.crosstab([data.Sex,data.Pclass],data.Survived,margins=True)


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab([data.Sex,data.Pclass],data.Survived).plot(kind='bar',ax=ax[0])
ax[0].set_title('Survival per Sex and Pclass')
sns.factorplot('Sex','Survived',hue='Pclass',data=data,ax=ax[1])
ax[1].set_title('Survival per Sex/Pclass in %')
plt.close()


# Comments:
# 
# This two variables together are good contribution for the model
# 
# Survival rate for woman in 1&2 class is arround 95%. In the opposite way, for man in 3class is arround 14%
# 
# Woman in 3class still survived more than Men in 1&2 class.

# **AGE**

# In[ ]:


data['Age'].describe()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
sns.violinplot('Survived','Age',data=data,split=True,ax=ax[0],)
ax[0].set_title('Survival per Age')
dataage=data.loc[data.Age.notnull(),] #to check normal distribution
sns.distplot(dataage.Age,ax=ax[1])
ax[1].set_title('Age distribution')


# Most part of people were between 20 and 40 years old.
# 
# Most part of kids below 10 years old were saved.

# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
sns.violinplot('Pclass','Age',hue='Survived',data=data,split=True,ax=ax[0])
ax[0].set_yticks(range(0,110,10))
sns.violinplot('Sex','Age',hue='Survived',data=data,split=True,ax=ax[1])
ax[1].set_yticks(range(0,110,10))


# Still possible to see that the children were all saved
# 
# Survival rate for female passengers between 20 and 40 years old is better

# **SIBSP**

# In[ ]:


data.groupby(['SibSp'])['Survived'].mean()


# In[ ]:


pd.crosstab(data.SibSp,data.Survived).plot(kind='bar')


# In[ ]:


sns.factorplot('SibSp','Survived',hue='Sex',col='Pclass',data=data)


# Comments:
# 
# -Being with 1 or 2 people seems to increase probability to survive, rather to be alone.
# 
# -Being with more than 2 people seems to be fatal. Females with 2 people in class 1 or 2 have almost 100% survival rate.

# **PARCH**

# In[ ]:


data.groupby(['Parch'])['Survived'].mean()


# In[ ]:


pd.crosstab(data.Parch,data.Survived).plot(kind='bar')


# In[ ]:


sns.factorplot('Parch','Survived',hue='Sex',col='Pclass',data=data)


# Being with 1 or 2 person seems to increase probability to survive, rather to be alone.

# **FARE**

# In[ ]:


data['Fare'].describe()


# In[ ]:


#The min is 0 which means there's people who didn't pay the ticket. 
#On the other side for the 3rd quartile to the 4rd one, the fare increases from 31 to 512.


# In[ ]:


sns.jointplot(x="Fare", y="Age", data=data) 


# In[ ]:


#To check any possible relation between age and fare and see variables distribution
#No relation between the 2 variables


# In[ ]:


f,ax=plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=5, reverse=True)
sns.kdeplot(dataage.Fare, dataage.Age, cmap=cmap, n_levels=60, shade=True)


# In[ ]:


# To see how is distributed our data in terms of Fare and Age
# Most part of our data set is between 20 and 40 years old and the paid price for tickets is between 0 and 30.


# In[ ]:


f,ax=plt.subplots(1,3,figsize=(14,5))
sns.distplot(data.loc[data['Pclass']==1,'Fare'],ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(data.loc[data['Pclass']==2,'Fare'],ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(data.loc[data['Pclass']==3,'Fare'],ax=ax[2])
ax[2].set_title('Fares in Pclass 3')


# In Class 1 the distributions vary between 0 and 500, being the maximum distribution between 0 and 200.
# 
# In Class 2 the distributions vary between 0 and 80, being the maximum between 10 and 40.
# In Class 2 the distributions vary between 0 and 60, being the maximum between 0 and 20.
# 
# This shows well a very high relation between Pclass and Fare.

# In[ ]:


f,ax=plt.subplots(figsize=(14,8))
sns.violinplot('Survived','Fare',data=data,split=True)
ax.set_yticks(range(0,540,20))


# Possible to see the expensier the Fare ticket the higher the chances of survival.

# **CABIN**

# In[ ]:


data['Cabin'].value_counts()


# In[ ]:


# There is a lot of information missing. And the remaining data is spread all over multiple cabins.
# Not relevant for the model


# **EMBAREKD**

# In[ ]:


data.groupby(['Embarked'])['Survived'].mean()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab(data.Embarked,data.Survived).plot(kind='bar',ax=ax[0])
ax[0].set_title('Survival per Embarked Place')
sns.factorplot('Embarked','Survived',data=data,ax=ax[1])
ax[1].set_title('Percentage of survival per Embark')
plt.close(2)


# Comments:
# - Most part of people embarked in Southampton
# - Survival rate was bigger for poeple who embark in Cherbourg and lower for Southampton
# - Let's check any relation between Embark place and Pclass/Sex

# In[ ]:


sns.factorplot('Embarked','Survived',hue='Sex',col='Pclass',data=data,)


# Comments:
# - Woman in Pclass 1 & 2 regardless the embark place have huge survival rates, but better for C and Q.
# - Mens tend to have low survival rate in Cherbourg
# - For woman in Pclass 3, to embark in C and Q increased a lot the survival rate.

# **NAME**
# 
# Name is a string and tipically does not add any valuable information to the analysis. Yet, in this case it have the initials of  the name, which may contain some relevant insight for the analysis.

# In[ ]:


data['Initials'] = 0
for i in range(data.shape[0]):
    data['Initials'] = data.Name.str.extract('([A-Za-z]+)\.')
    # To extract the initials
    # str.extract('([A-Za-z]+)\.') looks for strings which lie between A-Z or a-z and followed by a .(dot)


# In[ ]:


data['Initials'].value_counts() # to see the distinct values that were found.


# In[ ]:


# Since there is several distinct values meaning almost the same we are going to unite them in a few.
data['Initials'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)


# In[ ]:


# To validate the initial and the Sex
pd.crosstab(data.Initials,data.Sex)


# In[ ]:


data.groupby(['Initials'])['Survived'].mean()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(14,5))
pd.crosstab(data.Initials,data.Survived).plot(kind='bar',ax=ax[0])
sns.factorplot('Initials','Survived',data=data,ax=ax[1])
plt.close(2)


# Once more, a  bigger chance of survival per female initials, Mrs and Miss.
# 
# **DEALING WITH MISSING VALUES**
# 
# AGE - Wth the Initials we can easily assume a mean age to imput

# In[ ]:


data.groupby(['Initials'])['Age'].mean()


# In[ ]:


data.loc[(data['Age'].isnull()) & (data['Initials']=='Master'),'Age']=5
data.loc[(data['Age'].isnull()) & (data['Initials']=='Miss'),'Age']=22
data.loc[(data['Age'].isnull()) & (data['Initials']=='Mr'),'Age']=33
data.loc[(data['Age'].isnull()) & (data['Initials']=='Mrs'),'Age']=36
data.loc[(data['Age'].isnull()) & (data['Initials']=='Other'),'Age']=46

#To input the new values


# Embarked - Since only 3 values are missing. We will replace them for the most common value, which is S

# In[ ]:


data['Embarked'].fillna('S',inplace=True)


# In[ ]:


data.info() #To verify there is no more missing values for modelling


# **PART 2 - DATA CLEANING AND FEATURES ENGINEERING** 

# In this part, the main purpose is to convert string and numerical values into categorical values, since the models tipically don't accept strings.
# It's also in this part where we will drop variables not relevant to feed the model or in create new features that we might think it's relevant for model.
# 
# **Remember**, this is a process of constant testing, which means we can try to create/eliminate variables and not get the expected result. The goal is always to try, check the results and if not ok, re-doing all the processo of feature engineering all over.

# NEW FEATURES

# In[ ]:


data['companions']=0
for i in range(data.shape[0]):
    data['companions']=data['SibSp']+data['Parch']

data['alone']=0
data.loc[data['companions']>0,'alone']=1 # 1 represent individual not alone


# TRANSFORMING CONTINUOUS VARIABLES

# In[ ]:


#AGE 
#We saw that the older the individual the lower the survival rate. 
#We also checked that the younger people had bigger survival rates. 
#Thus, I will create classes based on kids, adults, midle age, old.

data.loc[data['Age']<20,'Age']=0
data.loc[(data['Age']>=20)&(data['Age']<40),'Age']=1
data.loc[(data['Age']>=40)&(data['Age']<60),'Age']=2
data.loc[data['Age']>=60,'Age']=3

#FARE
#We will replace the values based on quartiles.
data.loc[data['Fare']<8,'Fare']=0
data.loc[(data['Fare']>=8)&(data['Fare']<15),'Fare']=1
data.loc[(data['Fare']>=15)&(data['Fare']<31),'Fare']=2
data.loc[data['Fare']>=31,'Fare']=3


# TRANSFORMING STRINGS INTO CLASS

# In[ ]:


data['Sex'].replace(['male','female'],[0,1],inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'],[0,1,2],inplace=True)
data['Initials'].replace(['Mr', 'Mrs', 'Miss', 'Master', 'Other'],[0,1,2,3,4],inplace=True)


# DROPPING UNNEEDED VARIABLES

# In[ ]:


data.drop(['PassengerId','Name','Ticket','Cabin','SibSp','Parch'],axis=1,inplace=True)


# CHECKING AGAIN THE NEW/MODIFIED FEATURES

# In[ ]:


f,ax=plt.subplots(2,2,figsize=(15,10))
sns.factorplot('Age','Survived',hue='Sex',data=data,ax=ax[0,0])
ax[0,0].set_title('Survival per Age')
sns.factorplot('Fare','Survived',hue='Sex',data=data,ax=ax[0,1])
ax[0,1].set_title('Survival per Fare')
sns.factorplot('companions','Survived',hue='Sex',data=data,ax=ax[1,0])
ax[1,0].set_title('Survival per Companion')
sns.factorplot('alone','Survived',hue='Sex',data=data,ax=ax[1,1])
ax[1,1].set_title('Survival vs Alone')
plt.subplots_adjust(wspace=0.2,hspace=0.5)
plt.close(2)
plt.close(3)
plt.close(4)
plt.close(5)


# In[ ]:


#Since our interpretation of the data was maintained with the data transformation, we can move on.


# CORRELATIONS AND MULTICOLLINEARITY

# In[ ]:


plt.subplots(figsize=(18,8))
sns.heatmap(data.corr(),annot=True)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

X=data.loc[:,'Pclass':'alone']
y=data['Survived']

def calculate_vif(x):
    threeshold=10.0
    output=pd.DataFrame()
    vif=[variance_inflation_factor(np.array(X.values,dtype='float'),j)for j in range (X.shape[1])]
    for i in range (1,X.shape[1]):
        print('Iteration nº.:  ', i)
        print(vif)
        a=np.argmax(vif)
        print('Max vif is for variable.:    ',X.columns[a])
        print('With the value of.:  ',vif[a])
        print('')
        if vif[a] <= threeshold:
            break
        if i == 1:
            output = X.drop(X.columns[a], axis = 1)
            vif=[variance_inflation_factor(np.array(output.values,dtype='float'),j) for j in range (output.shape[1])]
        elif i > 1:
            output = output.drop(output.columns[a], axis = 1)
            vif=[variance_inflation_factor(np.array(output.values,dtype='float'),j) for j in range (output.shape[1])]
        return(output.collumns)            


# In[ ]:


calculate_vif(X)


# The function "calculate_vif" aims to calculate the variance of inflation for each variable of the dataset. Case when, it's detected a variable with VIF above 10, then the variable with max value of VIF is removed and the process re-start all over, detecting once more problems of Multicollinearity.
# 
# In this case, no problems were detected, since the Iteration stopped here:
# 
# if vif[a] <= threeshold:
#             break
# 

# **PART 3 - MODELLING**
# 
# In this part the main objective is to take data already with the variables treated for analysis and run the models to get the results of prediction.
# 
# For this, we will create train and test data. The train data will serve as the place to train and create the model for then to apply on the test data and check wheter the prediction is being well done.
# Concerning the models to apply, we should try several different methods and choose the one with better results.

# In[ ]:


from sklearn.model_selection import train_test_split # to split the data
from sklearn import metrics # to evaluate the model accuracy

x_train, x_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.3)


# In[ ]:


from sklearn import svm #Support Vector Machine
model=svm.SVC(kernel='rbf')
model.fit(x_train,y_train)
prediction1=model.predict(x_test)

model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
prediction2=model.predict(x_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
prediction3=model.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
prediction4=model.predict(x_test)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
prediction5=model.predict(x_test)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train,y_train)
prediction6=model.predict(x_test)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x_train,y_train)
prediction7=model.predict(x_test)

from sklearn.neural_network import MLPClassifier
model=MLPClassifier()
model.fit(x_train,y_train)
prediction8=model.predict(x_test)

from sklearn.ensemble import AdaBoostClassifier
model=AdaBoostClassifier()
model.fit(x_train,y_train)
prediction9=model.predict(x_test)

from sklearn.ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(x_train,y_train)
prediction10=model.predict(x_test)

from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(x_train,y_train)
prediction11=model.predict(x_test)

print('Accuracy for rbf SVM is:  ',metrics.accuracy_score(y_test,prediction1))
print('Accuracy for linear SVM is:  ',metrics.accuracy_score(y_test,prediction2))
print('Accuracy for Logistic Regression is:  ',metrics.accuracy_score(y_test,prediction3))
print('Accuracy for Random Forest is:  ',metrics.accuracy_score(y_test,prediction4))
print('Accuracy for Decision Tree is:  ',metrics.accuracy_score(y_test,prediction5))
print('Accuracy for KNeighbors is:  ',metrics.accuracy_score(y_test,prediction6))
print('Accuracy for Naive Bayes is:  ',metrics.accuracy_score(y_test,prediction7))
print('Accuracy for Neural Network is:  ',metrics.accuracy_score(y_test,prediction8))
print('Accuracy for AdaBoost is: ',metrics.accuracy_score(y_test,prediction9))
print('Accuracy for GradientBoosting is: ',metrics.accuracy_score(y_test,prediction10))
print('Accuracy for XGBoost is: ',metrics.accuracy_score(y_test,prediction11))


# HYPER-PARAMETER -  At this step the objective is to get for the machine learning models, the parameters that can maximize the accuracy of the model. For example, for Random Forest and Decision Trees the estimators number can increase/decrease the power of the model. In KNeighbors, the number of neighbors can also change the accuracy.
# 
# That being said, we will find what the numbers that can maximize this models.

# In[ ]:


from sklearn.model_selection import GridSearchCV

#Support Vector Machine
C=[0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
kernel=['rbf','linear']
hyper={'kernel':kernel,'C':C,'gamma':gamma}
gd1=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)
gd1.fit(x_train,y_train)

#Random Forest
n_estimators=range(100,1100,100)
hyper={'n_estimators':n_estimators}
gd2=GridSearchCV(estimator=RandomForestClassifier(),param_grid=hyper,verbose=True)
gd2.fit(x_train,y_train)

#AdaBoost
n_estimators=list(range(100,1100,100))
learning_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
hyper={'n_estimators':n_estimators,'learning_rate':learning_rate}
gd3=GridSearchCV(estimator=AdaBoostClassifier(),param_grid=hyper,verbose=True)
gd3.fit(x_train,y_train)

#Gradient Boosting
n_estimators=list(range(100,1100,100))
learning_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
hyper={'n_estimators':n_estimators,'learning_rate':learning_rate}
gd4=GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=hyper,verbose=True)
gd4.fit(x_train,y_train)

#XGboost
n_estimators=list(range(100,1100,100))
learning_rate=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
hyper={'n_estimators':n_estimators,'learning_rate':learning_rate}
gd5=GridSearchCV(estimator=XGBClassifier(),param_grid=hyper,verbose=True)
gd5.fit(x_train,y_train)

print('The best score for SVM is: ', gd1.best_score_)
print('With the estimator parameters: \n \n', gd1.best_estimator_)
print('The best score for Random Forest is: ', gd2.best_score_)
print('With the estimator parameters: \n \n', gd2.best_estimator_)
print('The best score for AdaBoost is: ', gd3.best_score_)
print('With the estimator parameters: \n \n', gd3.best_estimator_)
print('The best score for Gradient Boosting is: ', gd4.best_score_)
print('With the estimator parameters: \n \n', gd4.best_estimator_)
print('The best score for XGboost is: ', gd5.best_score_)
print('With the estimator parameters: \n \n', gd5.best_estimator_)


# In[ ]:


#KNNeighbors
N_Neighbors= list(range(5,20))
knn_train=pd.Series()
knn_test=pd.Series()
for i in N_Neighbors:
    model=KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train,y_train)
    prediction_train=model.predict(x_test)
    prediction_test=model.predict(x_train)
    knn_train=knn_train.append(pd.Series(metrics.accuracy_score(y_test,prediction_train)))
    knn_test=knn_test.append(pd.Series(metrics.accuracy_score(y_train,prediction_test)))
plt.subplots(figsize=(15,10))
plt.plot(N_Neighbors,knn_train, label='train set accuracy')
plt.plot(N_Neighbors,knn_test, label='test set accuracy')
plt.xlabel('N_Neighbors')
plt.ylabel('Accuracy')
plt.legend()


# In this case the plot suggest the best number of  neighbors would be 11

# In[ ]:


#Random Forest
max_leaf_nodes=list(range(10,100,10))
randomforest_train=pd.Series()
randomforest_test=pd.Series()
for i in max_leaf_nodes:
    model=RandomForestClassifier(max_leaf_nodes=i, n_estimators=200)
    model.fit(x_train,y_train)
    prediction_train=model.predict(x_train)
    prediction_test=model.predict(x_test)
    randomforest_train=randomforest_train.append(pd.Series(metrics.accuracy_score(y_train,prediction_train)))
    randomforest_test=randomforest_test.append(pd.Series(metrics.accuracy_score(y_test,prediction_test)))
plt.subplots(figsize=(15,10))
plt.plot(max_leaf_nodes,randomforest_train, label='train set accuracy')
plt.plot(max_leaf_nodes,randomforest_test,label='test set accuracy')
plt.xlabel('Max_leaf_nodes')
plt.ylabel('Accuracy')
plt.legend()    


# In this case the plot suggest the best number of for max_leaf_nodes would be 60.

# In[ ]:


#Decision Trees
max_leaf_nodes=list(range(10,100,10))
decisiontrees_train=pd.Series()
decisiontrees_test=pd.Series()
for i in max_leaf_nodes:
    model=DecisionTreeClassifier(max_leaf_nodes=i)
    model.fit(x_train,y_train)
    prediction_train=model.predict(x_train)
    prediction_test=model.predict(x_test)
    decisiontrees_train=decisiontrees_train.append(pd.Series(metrics.accuracy_score(y_train,prediction_train)))
    decisiontrees_test=decisiontrees_test.append(pd.Series(metrics.accuracy_score(y_test,prediction_test)))
plt.subplots(figsize=(15,10))
plt.plot(max_leaf_nodes,decisiontrees_train, label='train set accuracy')
plt.plot(max_leaf_nodes,decisiontrees_test, label='test set accuracy')
plt.xlabel('Max_leaf_nodes')
plt.ylabel('Accuracy')
plt.legend()


# The plot suggest in this case a max_leaf_nodes = 40

# **PART 4 - VALIDATION**
# 
# In this part we are going to check the quality of the model assesing cross validation, confusion matrix, and features importance. 

# **CROSS VALIDATION**

# In[ ]:


from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation

kfold=KFold()
accuracy_mean=[]
accuracy_std=[]
classifiers=['RBF SVM','Linear SVM','Logistic Regression','Decision Trees','Random Forest',
             'KNneighbors','Naive Bayes','Neural Networks','Adaboostclassifier',
             'GradientBoostingClassifier','XGboost']
models=[svm.SVC(C=0.3,gamma=0.3),svm.SVC(kernel='linear'),LogisticRegression(),DecisionTreeClassifier(max_leaf_nodes=40),
        RandomForestClassifier(n_estimators=600,max_leaf_nodes=60),KNeighborsClassifier(n_neighbors=11),
        GaussianNB(),MLPClassifier(),AdaBoostClassifier(learning_rate=0.05, n_estimators=200),
        GradientBoostingClassifier(learning_rate=0.05,),XGBClassifier(learning_rate=0.05)]
        # Already with hyper parameters
for i in models:
    model=i
    cv_result=cross_val_score(model,X,y,cv=kfold,scoring='accuracy')
    accuracy_mean.append(cv_result.mean())
    accuracy_std.append(cv_result.std())

model_frame=pd.DataFrame({'classifiers':classifiers,'Model_Mean':accuracy_mean,'Model_Std':accuracy_std})
model_frame.sort_values(by=['Model_Mean'],ascending=False)


# Based on the results the best 5 models are RBF SVM, Random Forest, Adaboost, Gradient Boosting and XGboost.

# **CONFUSION MATRIX**

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

f,ax=plt.subplots(3,2,figsize=(12,10))
pred=cross_val_predict(svm.SVC(C=0.3,gamma=0.3),X,y,cv=10)
sns.heatmap(confusion_matrix(y,pred),annot=True,fmt='2.0f', ax=ax[0,0])
ax[0,0].set_title('Matrix for RBF_SVM')
pred=cross_val_predict(RandomForestClassifier(n_estimators=600,max_leaf_nodes=60),X,y,cv=10)
sns.heatmap(confusion_matrix(y,pred),annot=True,fmt='2.0f', ax=ax[0,1])
ax[0,1].set_title('Matrix for RandomForest')
pred=cross_val_predict(AdaBoostClassifier(learning_rate=0.05, n_estimators=200),X,y,cv=10)
sns.heatmap(confusion_matrix(y,pred),annot=True,fmt='2.0f', ax=ax[1,0])
ax[1,0].set_title('Matrix for Adaboost')
pred=cross_val_predict(GradientBoostingClassifier(learning_rate=0.05, n_estimators=200),X,y,cv=10)
sns.heatmap(confusion_matrix(y,pred),annot=True,fmt='2.0f', ax=ax[1,1])
ax[1,1].set_title('Matrix for GradientBoosting')
pred=cross_val_predict(XGBClassifier(learning_rate=0.05, n_estimators=100),X,y,cv=10)
sns.heatmap(confusion_matrix(y,pred),annot=True,fmt='2.0f', ax=ax[2,0])
ax[2,0].set_title('Matrix for XGboost')


# The confusion matrix is interpreted in the following way. If we consider the first matrix of rbf-SVM, our model predicted:
# 
# [0,0 ] - 496 people would not survive - And they really didn't survive;
# [1,1] - 247 people would survived - And they really survived;
#     Score_Accuracy = (496+247)/(496+53+95+247)=83%
# ERRORS:
# [0,1] - 57 people would survive - And they died
# [1,0] - 94 people would die - And they survived
# 
# Thus, based on it, we can make assumptions such as:
# -The model who better classified  survival is Adaboost and RandomForest
# -The model who better classified dead pessengers is XGboost.
# -Since, predicting that a pessenger will survived and then the it dies, might be very dangerous, the better model for this case is also XGboost with the number of 49.
# 
# The choice of the model will then depend always on the business case and on your objectives.
# 
# 

# **FEATURES IMPORTANCE**

# In[ ]:


f,ax=plt.subplots(2,2, figsize=(15,10))
model=RandomForestClassifier(n_estimators=600,max_leaf_nodes=60)
model.fit(X,y)  
pd.Series(model.feature_importances_,X.columns).sort_values().plot(kind='barh',width=0.8,ax=ax[0,0])
ax[0,0].set_title('RandomForest')
model=AdaBoostClassifier(learning_rate=0.05, n_estimators=200)
model.fit(X,y)  
pd.Series(model.feature_importances_,X.columns).sort_values().plot(kind='barh',width=0.8,ax=ax[0,1])
ax[0,1].set_title('Adaboost')     
model=GradientBoostingClassifier(learning_rate=0.05, n_estimators=200)
model.fit(X,y)  
pd.Series(model.feature_importances_,X.columns).sort_values().plot(kind='barh',width=0.8,ax=ax[1,0])
ax[1,0].set_title('GradientBoosting')        
model=XGBClassifier(learning_rate=0.05, n_estimators=100)
model.fit(X,y)  
pd.Series(model.feature_importances_,X.columns).sort_values().plot(kind='barh',width=0.8,ax=ax[1,1])
ax[1,1].set_title('XGboost')

# For rbf-SVM the attribute feature_importances_ is not available


# Comments:
# 
# Initials were in general the best predictor of survival rate in Titanic, followed by companions, and Pclass.
# Curiously, sex was not relevant is any of the top models, contrarly to was identified in exploratory data analysis.
# 

# **Thanks a lot for taking a look at this notebook. Please upvote if you liked it!**
# 
# Relevant references for creating this Kernel were taken on (by relevance):
# 
# [Titanic Competitionl from I,Coder on Kaggle](https://www.kaggle.com/ash316/eda-to-prediction-dietanic)
# 
# [Linear Regression from Ekta on ListenData](https://www.listendata.com/2018/01/linear-regression-in-python.html)
# 
# [Diabetes Analysis from Susan Li on Towards Data Science](https://towardsdatascience.com/machine-learning-for-diabetes-562dd7df4d42)
# 
# [Logistic Regression on Python from Susan Li on Towards Data Science](https://www.kaggle.com/learn/machine-learning)
