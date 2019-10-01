#!/usr/bin/env python
# coding: utf-8

# **Dead OR Alive: Titanic Disaster for Beginners**
# 
# 15th April, 1912: Its been more than a century, when the mighty ship met her unfateful sink. Today, we are going to look through the dataset and comeout with a prediction of survival.
# 
# **Path of Approach**
# 1. Import the necessary libraries
# 2. Reading the csv files
# 3. Data structure and missing values
# 4. Cleaning and Transformation data
# 5. EDA through Visualization
# 6. Train the model with training data
# 7. Fit and Predict using different models
# 8. KFold Cross Validation
# 9. Comparing models
# 10. Submission
# 

# **IMPORTING LIBRARIES**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# **READING FILES**
# 
# Here, we are using only train dataset. Our methodology will be, we will split the "train" dataset into "training" and "Validating" data. And after applying all the model (LR, SVM,DT and RF) will choose the best accuracy led model.
# 
# NOTE: I have not used test set for testing my model, will be implementing the same later on, timely.

# In[ ]:


train = pd.read_csv("../input/train.csv")


# PassengerID - UniqueID
# 
# Pclass  - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# 
# **Survival  (0 = No; 1 = Yes)**
# 
# name - Name
# 
# sex -  Sex
# 
# age -  Age
# 
# sibsp  - Number of Siblings/Spouses Aboard
# 
# parch  - Number of Parents/Children Aboard
# 
# ticket  - Ticket Number
# 
# fare  - Passenger Fare (British pound)
# 
# cabin  - 
# 
# embarked  - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
# 

# **DATA STRUCTURE AND MISSING VALUES**
# 
# One of the most interesting part of ML. If you understand your data structure well and tackle all the missing/unwanted entries, you are closer to your destination for prediction.

# In[ ]:


train.head(5)


# In[ ]:


train.info()


# In[ ]:


train.isna().sum()


# So, here we have missing values in Age, Cabin and Embarked variables. We will replace the missing values is Age with "Mean" of Age. Note that missing values in "Cabin" is way large (almost 80% is missing), so this variable can be dropped without and worrying of lossing any kind of valuable information. And "Embarked" will be replaced by "Mode" of Embarked variables.

# In[ ]:


round(np.mean(train["Age"]))
train["Age"].fillna(round(np.mean(train["Age"])), inplace=True)


# In[ ]:


train["Embarked"].value_counts()
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)


# Here, we have 5 object type or categorical type of variable. For now we are dropping the "PassengerID" ,"Name", "Ticket", "Cabin" variables, cause of intuition that they wont be contibuting for model prediction. But after model creation we will look into these variables too. So, now are we left with "Sex" variable which will be converting to numerical category using 'apply'. LabelEncoder  can also be used. 

# In[ ]:


target = {"male":0, "female":1}
train["Sex"] = train["Sex"].apply(lambda x: target[x])


# In[ ]:


train = train.drop(["PassengerId" , "Ticket", "Cabin"], axis=1)


# In[ ]:


train.head(5)          #Cleaned Dataframe


# Data is almost clean for us as of now. But brace yourself, for future, to spend more time in here when more variable and missing values are encountered.

# **EDA THROUGH VISUALIZATION**
# 
# This is where you will extract almost all the information through visual interpretation.

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(train["Survived"], data=train, hue="Pclass", palette="Set1", saturation=0.80)


# Here, we can see that "Pclass-3" has more number of Non-Survivors, followed by "Pclass-2" and "Pclass-1". The reason could be more number of passengers travelliing in Pclass-3, as it is cheap. Note that "Pclass-1" has higher number of Survivors.

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(train["SibSp"], data=train, hue="Survived", palette="Set2", saturation=0.9)


# People without Sibling/Spouse has higher number of deaths. People travelling with single spouse/Sibliing has a bit higher survival rate compared to non-survivors for same category. Also, note that people travelling with more than 5 or greater had no chance of survival.

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(train["Parch"], data=train, hue="Survived", palette="Set1", saturation=0.9)


# A similiar trend could be seen in here. Travellers without Parent/Child has more number of deaths. People with single Parent/Child has more survivors to their complimentry. And more than 4 has very less chance of survival.

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(train["Embarked"], data=train, hue="Survived", palette="Set2", saturation=0.75)


# So, people who boarded at Southampton has large number of deaths in the tragedy. Cherbourg people somehow managed to survive more compared to the deaths. But Queenstown could not do the same.

# In[ ]:


plt.figure(figsize=(8,5))
sns.countplot(train["Sex"], data=train, hue="Survived", palette="Set1", saturation=0.75)


# 1:Female and 0:Male. So, lets see what we have here. We see that Female are the larger survivors compared to Male. The high red bar of Male, shows larger number of deaths (0). 

# Lets look at the Survivors against Age variable.

# In[ ]:


graph = sns.FacetGrid(train, col='Survived')
graph.map(plt.hist, 'Age', bins=20 , color='g')


# Few insights from histograph, Infant below (<=5) survived. Old people age 78-80 years somehow survived. The maximum non-survivors are in the age band of 19-35 years.

# In[ ]:


graph1 = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.5, aspect=1.8)
graph1.map(plt.hist, 'Age', bins=20, color='r')
graph1.add_legend()


# Few insights, those few infants who 'died' were travelling in Pclass-3. And those 'survived' were travelling evenly in Pclass-3 and Pclass-2. Adults aged 18-40 years, who did not survive, were travelling in Pclass-3. People who 'survived' in the same category were travelling in Pclass-1. Strangly, people aged 30 were among the large survivors compared to other class.

# **DUMMY CREATION FOR "EMBARKED COLUMN"**

# In[ ]:


dummy = pd.get_dummies(train["Embarked"], prefix='Embarked').iloc[:,1:]


# In[ ]:


train = pd.concat([train,dummy], axis=1)
train = train.drop(["Embarked"], axis=1)


# The iloc part for selecting only two variable and avoid the dummy trap. Later we have concatenated the dummy variables with our working train dataset and dropping the "Embarked" variable.

# In[ ]:


train.head(5)


# We see two column "SibSp" and "Parch" can be combined under one hood of "Family" and remove the two variables. 

# In[ ]:


train["Family"] = train["SibSp"] + train["Parch"]
train["Family"].loc[train["Family"]>0]=1       #1 indicates --- travelling With Family
train["Family"].loc[train["Family"]==0]=0      #0 indicates --- travelling Alone
train = train.drop(["SibSp","Parch"], axis=1)


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(12,7))                #this wil create the two blank subplots
sns.countplot(x='Family', data=train, order=[1,0], ax=axis1)                      #This will plot in first box, with family and alone count
family_df = train[["Family","Survived"]].groupby(["Family"], as_index=False).mean()   #Grouping by family
sns.barplot(x="Family", y="Survived", data=family_df, order=[1,0], ax=axis2)         #Plotting in second plot the survival


# So, we see that most of the passengers were travelling "Alone". But in contrast to that, Alone people did not survive. "Family" members survived the disaster in more number.

# Categorizing the Fare according to the range.

# In[ ]:


#Fare
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
train.loc[ train['Fare'] <= 7.91, 'Fare'] = 0
train.loc[(train['Fare'] > 7.91) & (train['Fare'] <= 14.454), 'Fare'] = 1
train.loc[(train['Fare'] > 14.454) & (train['Fare'] <= 31), 'Fare']   = 2
train.loc[(train['Fare'] > 31) & (train['Fare'] <= 99), 'Fare']   = 3
train.loc[(train['Fare'] > 99) & (train['Fare'] <= 250), 'Fare']   = 4
train.loc[train['Fare'] > 250, 'Fare'] = 5
train['Fare'] = train['Fare'].astype(int)


# Adding a new column of "Title" using "Name" column in the dataset. Findinf which Title survived more and who could not survive.

# In[ ]:


train["Title"] = train["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
train[["Title"]] = train[["Title"]].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train[["Title"]] = train[["Title"]].replace('Mlle', 'Miss')
train[["Title"]] = train[["Title"]].replace('Ms', 'Miss')
train[["Title"]] = train[["Title"]].replace('Mme', 'Mrs')
train[["Title","Survived"]].groupby(["Title"], as_index=False).mean()
target = {"Master":1, "Miss":2, "Mr":3, "Mrs":4, "Rare":5}
train["Title"] = train["Title"].map(lambda x: target[x])
train = train.drop(["Name"], axis=1)


# In[ ]:


train.head(5)


# **CORRELATION AND HEATMAP**
# 
# We will try to find the correlation among the variables using Heatmap.

# In[ ]:


cr = train.corr()
plt.figure(figsize=(15,12))
sns.heatmap(cr, annot=True, cmap='viridis', fmt='.2g')


# **ALERT **
# 
# As i informed initially the way i have approach the problem. Now we are going read the "test set" and going to predict the output and submit our submission file. 
# Now i will be little fast on cleaning and processing altogether assuming you have gone through the above concepts and clear on code part.

# In[ ]:


test = pd.read_csv("../input/test.csv")


# In[ ]:


testc = test.copy()
test.info()
test.describe()
test.isna().sum()


# In[ ]:


test["Age"].fillna(round(np.mean(test["Age"])), inplace=True)        #filling missing value in Age
test["Fare"].fillna(round(np.mean(test["Fare"])), inplace=True)      #filling missing value in Fare
test = test.drop(["Cabin"], axis=1)                                   #dropping Cabin variable


# In[ ]:


target = {"female":1, "male":0}
test["Sex"] = test["Sex"].apply(lambda x: target[x])             #Categorizing Sex variable
dummy = pd.get_dummies(test["Embarked"], prefix='Embarked').iloc[:,1:]    #Dummy creation
test = pd.concat([test, dummy], axis=1)
test = test.drop(["Embarked", "PassengerId","Ticket"], axis=1)


# In[ ]:


#Combining SibSp and Parch
test["Family"] = test["SibSp"] + test["Parch"]
test["Family"].loc[test["Family"]>0]=1       #With Family
test["Family"].loc[test["Family"]==0]=0      #Alone
test = test.drop(["SibSp","Parch"], axis=1)


# In[ ]:


#Fare
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)
test.loc[test['Fare'] <= 7.91, 'Fare'] = 0
test.loc[(test['Fare'] > 7.91) & (test['Fare'] <= 14.454), 'Fare'] = 1
test.loc[(test['Fare'] > 14.454) & (test['Fare'] <= 31), 'Fare']   = 2
test.loc[(test['Fare'] > 31) & (test['Fare'] <= 99), 'Fare']   = 3
test.loc[(test['Fare'] > 99) & (test['Fare'] <= 250), 'Fare']   = 4
test.loc[test['Fare'] > 250, 'Fare'] = 5
test['Fare'] = test['Fare'].astype(int)


# In[ ]:


#Creating new column "Title"
test["Title"] = test["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
test[["Title"]] = test[["Title"]].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test[["Title"]] = test[["Title"]].replace('Mlle', 'Miss')
test[["Title"]] = test[["Title"]].replace('Ms', 'Miss')
test[["Title"]] = test[["Title"]].replace('Mme', 'Mrs')
target = {"Master":1, "Miss":2, "Mr":3, "Mrs":4, "Rare":5}
test["Title"] = test["Title"].map(lambda x: target[x])
test = test.drop(["Name"], axis=1)


# In[ ]:


test.head(5)


# In[ ]:


X_train = train.iloc[:,1:9]
y_train = train.iloc[:,0:1]
X_test = test.iloc[:,:]


# In[ ]:


from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X_train, y_train)


# In[ ]:


y_pred_log = regressor.predict(X_test)


# In[ ]:


print("Accuracy of LR: ", regressor.score(X_train, y_train)*100)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
regressor2 = DecisionTreeClassifier(max_depth=5, random_state=100)
regressor2.fit(X_train, y_train)


# In[ ]:


y_pred_DT = regressor2.predict(X_test)


# In[ ]:


print("Accuracy of DT: ", regressor2.score(X_train, y_train)*100)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
regressor3 = RandomForestClassifier(n_estimators=200, random_state=100)
regressor3.fit(X_train, y_train)


# In[ ]:


y_pred_RF = regressor3.predict(X_test)


# In[ ]:


print("Accuracy of RF: ", regressor3.score(X_train, y_train)*100)


# In[ ]:


importances=pd.Series(regressor3.feature_importances_, index=X_train.columns)
importances.plot(kind='barh', figsize=(10,8))


# In[ ]:


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
data1 = go.Bar( x= ["LR","DT","RF"],
                y= [regressor.score(X_train, y_train) *100,
                    regressor2.score(X_train, y_train) *100, regressor3.score(X_train, y_train) *100]
                )
data = [data1]
layout = go.Layout(barmode='group')

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='grouped-bar')


# In[ ]:


submission = pd.DataFrame({"PassengerId": testc["PassengerId"],"Survived": y_pred_RF})


# In[ ]:


submission.head(10)


# In[ ]:


submission.to_csv("Submission_Titanic", index=False)


# So, I have come to an end of my kernel. Its my first kernel in Kaggle for ML. Hope you like it, if yes give it a THUMPS UP. :
