#!/usr/bin/env python
# coding: utf-8

# "Learning from disaster" with titanic dataset has become one of the most famous datasets amongst Kaggle community. This is my first kernel, where I am learning a lot of new stuff and exploring more with tools and techniques. Please have a look and consider commenting for further improvisation. 
# 
# ####  Import neecessary packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
get_ipython().magic(u'matplotlib inline')
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.mode.chained_assignment=None
from pandas.plotting import table


# #### Check for input files and load them

# In[ ]:


from subprocess import check_output
check_output(["ls", "../input"]).decode("utf8")


# In[ ]:


titanic_train = pd.read_csv("../input/train.csv", low_memory=False)
titanic_train.head()


# In[ ]:


titanic_test = pd.read_csv("../input/test.csv", low_memory=False)
titanic_test.head()


# #### Basic statistical info about training and test data

# In[ ]:


print("Number of rows in training data = {}".format(len(titanic_train)))
print("Number of columns in training data = {}".format(titanic_train.shape[1]))

print("Information about training data types:");
print(titanic_train.dtypes)

print("Some statistical information:")
print(titanic_train.describe())


# #### Set PassengerId as index

# In[ ]:


titanic_train = titanic_train.set_index('PassengerId')
tianic_test = titanic_test.set_index('PassengerId')


# **What is the total percentage of survival in the given train data?**

# In[ ]:


fig,ax=plt.subplots(figsize=(12,8))
titanic_train['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax,shadow=True);
ax.set_title('Survived');
ax.set_ylabel('');
plt.legend(['Dead','Survived']);


# #### Are there any missing values ?

# In[ ]:


print("Missing values : ")
missing_values=titanic_train.isnull().sum()[titanic_train.isnull().sum()>0]
print(missing_values)
fix, ax = plt.subplots(figsize=(12,8))
sns.barplot(missing_values.values,missing_values.index);
ax.set_xlabel('Missing values');
ax.set_ylabel('Features');
ax.set_title('Features with missing values');


# It is clear that **Survived** is the target parameter.  Hence as a part of data exploration, we want to see how other features affect the target parameter.  
# 
# ### Fare
# From my initial guess, fare is one of the most important features for survival.  Let us explore more about it. 
# 
# ##### How is Fare distributed ?

# In[ ]:


fig, ax = plt.subplots(figsize=(12,8))
sns.distplot(titanic_train.Fare, hist=False, kde=True);
table(ax, np.round(titanic_train['Fare'].to_frame().describe()),loc='upper right', colWidths=[0.2, 0.2, 0.2])
ax.set_xlabel('Fare');
ax.set_ylabel('Density');
ax.set_title('Distribution of Titanic Fares');


# ##### Survival with Fare
# We observe that fare is a continuous parameter and has a very high range from 0 to 500, with mean value around 32. Hence, I converted this continuos value to a categorical range, where each category is not symmetric as:

# In[ ]:


titanic_Fare_range = [(0,8),(9,16),(17,30),(31,60),(61,100),(101,600)]
Fare_range = []
titanic_Fare_int = list(map(int,titanic_train.Fare.values))
for j in range(len(titanic_Fare_int)):
    for i in range(len(titanic_Fare_range)):
        if titanic_Fare_int[j] in range(titanic_Fare_range[i][0],titanic_Fare_range[i][1]):
            fare_range = titanic_Fare_range[i]
        else:
            pass
    Fare_range.append(fare_range)


# In[ ]:


titanic_train['Fare_range']=Fare_range


# In[ ]:


sns.factorplot(x="Fare_range",hue="Survived", data=titanic_train, kind="count",size=6, 
               aspect=1.5, order=titanic_Fare_range);


#  It looks like low fare has low survival.  
#  
# *  **What is actual percentage of distribution of Fare_range ? **
# *  **What is the survival percentage in each Fare_range ?**

# In[ ]:


fig,ax = plt.subplots(figsize=(18,8))
ax1 = plt.subplot(121)
titanic_train['Fare_range'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1,shadow=True, 
                                                    legend=True);
ax1.set_title('Percentage distribution of fare range');
ax1.set_ylabel('');
#plt.legend(['0-8','9-15','16-30','31-60','61-100','101-600']);

survival_percent_fr = 100*(titanic_train.groupby('Fare_range').sum()['Survived']/(titanic_train.groupby('Fare_range').count()['Survived']))
#survival_percent_fr
ax2 = plt.subplot(122)
sns.barplot(survival_percent_fr.index.values, survival_percent_fr.values);
ax2.set_ylabel('survival percentage')
ax2.set_title('Percentage survived in each Fare range');


# ### Sex

# In[ ]:


sns.factorplot(x="Sex",hue='Survived', data=titanic_train, kind="count",
                   palette="BuPu", size=6, aspect=1.5);


# In[ ]:


titanic_train['Sex'].value_counts()


# In[ ]:


survival_percent_sex = 100*(titanic_train.groupby('Sex').sum()['Survived']/(titanic_train.groupby('Sex').count()['Survived']))
survival_percent_sex


# Only 18.9% of males survived while 74.2% of total females survived. 

# ### Age 
# 
# Let us also look at the distribution of age.  First, we want to fill the missing age with the median of the age of all passengers. 

# In[ ]:


titanic_train['Age'] = titanic_train['Age'].fillna(titanic_train['Age'].median())


# In[ ]:


fix, ax = plt.subplots(figsize=(12,8))
sns.distplot(titanic_train.Age);
table(ax, np.round(titanic_train['Age'].to_frame().describe()),loc='upper right', colWidths=[0.2, 0.2, 0.2])
ax.set_xlabel('Age');
ax.set_ylabel('density');
ax.set_title('Distribution of Age');


# In[ ]:


titanic_Age_range = [(0,12),(13,20),(21,30),(31,40),(41,60),(61,80)]
Age_range = []
titanic_Age_int = list(map(int,titanic_train.Age.values))
for j in range(len(titanic_Age_int)):
    for i in range(len(titanic_Age_range)):
        if titanic_Age_int[j] in range(titanic_Age_range[i][0],titanic_Age_range[i][1]):
            age_range = titanic_Age_range[i]
        else:
            pass
    Age_range.append(age_range)


# In[ ]:


titanic_train['Age_range'] = Age_range


# In[ ]:


sns.factorplot(x="Age_range",hue="Survived", data=titanic_train, kind="count",size=6, 
               aspect=1.5, order=titanic_Age_range);


# In[ ]:


fig,ax = plt.subplots(figsize=(18,8))
ax1 = plt.subplot(121)
titanic_train['Age_range'].value_counts().plot.pie(autopct='%1.1f%%',ax=ax1,shadow=True, 
                                                    legend=True);
ax1.set_title('Percentage distribution of age range');
ax1.set_ylabel('');
#plt.legend(['0-8','9-15','16-30','31-60','61-100','101-600']);

survival_percent_ag = 100*(titanic_train.groupby('Age_range').sum()['Survived']/(titanic_train.groupby('Age_range').count()['Survived']))
#survival_percent_fr
ax2 = plt.subplot(122)
sns.barplot(survival_percent_ag.index.values, survival_percent_ag.values);
ax2.set_ylabel('survival percentage')
ax2.set_title('Percentage survived in each Age range');
#survival_percent_ag = 100*(titanic_train.groupby('Age_range').sum()['Survived']/(titanic_train.groupby('Age_range').count()['Survived']))
#survival_percent_ag


# ### Pclass
# 
# What roles do ticket class have ? 

# In[ ]:


titanic_train.Pclass.value_counts()


# In[ ]:


sns.factorplot(x="Pclass",hue="Survived", data=titanic_train, kind="count",size=6, 
               aspect=1.5);


# In[ ]:


survival_percent_pcl = 100*(titanic_train.groupby('Pclass').sum()['Survived']/(titanic_train.groupby('Pclass').count()['Survived']))
survival_percent_pcl


# 
# ### SibSP
# Number  of siblings / spouses aboard the Titanic(SibSp)

# In[ ]:


titanic_train.SibSp.value_counts()


# In[ ]:


sns.factorplot(x="SibSp",hue="Survived", data=titanic_train, kind="count",size=6, 
               aspect=1.5);


# In[ ]:


survival_percent_sibsp = 100*(titanic_train.groupby('SibSp').sum()['Survived']/(titanic_train.groupby('SibSp').count()['Survived']))
survival_percent_sibsp


# 
# ### Parch
# Number  of parents / children aboard the Titanic 

# In[ ]:


titanic_train.Parch.value_counts()


# In[ ]:


sns.factorplot(x="Parch",hue="Survived", data=titanic_train, kind="count",size=6, 
               aspect=1.5);


# In[ ]:


survival_percent_parch = 100*(titanic_train.groupby('Parch').sum()['Survived']/(titanic_train.groupby('Parch').count()['Survived']))
survival_percent_parch


# ### Embarked

# In[ ]:


titanic_train.Embarked.value_counts()


# In[ ]:


sns.factorplot(x="Embarked",hue="Survived", data=titanic_train, kind="count",size=6, 
               aspect=1.5);


# In[ ]:


survival_percent_embk = 100*(titanic_train.groupby('Embarked').sum()['Survived']/(titanic_train.groupby('Embarked').count()['Survived']))
survival_percent_embk


# ### Name 
# 
# **Title is important than name itself ** . The title in the name is the most important part. Let us observe the survival based on the titles.

# In[ ]:


Title=[]
for i in range(len(titanic_train)):
    names = titanic_train.Name.values[i].replace('.',',').split(',')
    title = names[1]
    Title.append(title)    


# In[ ]:


titanic_train['PTitle'] = Title


# In[ ]:


titanic_train.PTitle.value_counts()


# In[ ]:


sns.factorplot(x="PTitle",hue="Survived", data=titanic_train, kind="count",size=10, 
               aspect=1.5);


# In[ ]:


survival_percent_tit = 100*(titanic_train.groupby('PTitle').sum()['Survived']/(titanic_train.groupby('PTitle').count()['Survived']))
survival_percent_tit


# ### Cabin
# 
# What about this parameter ? Most of the values are missing, but let us explore if this makes some sense out of the given data. 

# In[ ]:


titanic_train.Cabin.value_counts().iloc[0:5]


# So many unique values, but each of them have alphabets as the first letter. Let us consider a new column with only first alphabet (Deck name) as the relevant parameters and we replace all not given values with name 'N'.

# In[ ]:


titanic_train['Cabin'] = titanic_train.Cabin.fillna('N')
titanic_decks = []
for i in range(len(titanic_train['Cabin'])):
    titanic_deck = titanic_train.Cabin.values[i][0]
    titanic_decks.append(titanic_deck)
titanic_train['Deck'] = titanic_decks


# In[ ]:


titanic_train.Deck.value_counts()


# 687 values are missing, and let us observe the survival for each Deck.

# In[ ]:


sns.factorplot(x="Deck",hue="Survived", data=titanic_train, kind="count",size=8, 
               aspect=1.5);


# Deck, however didn't give us much knowledge because there are equally survival and dead in almost all decks, except the one whose values were empty (N). Thus, we neglect this parameter.
# 
# Let us create a df with all relevant columns as:

# In[ ]:


titanic_train_relevant_cols = ['Survived','Pclass','Sex','SibSp','Parch','Embarked',
                              'PTitle','Age_range','Fare_range']
titanic_train_relevant=titanic_train[titanic_train_relevant_cols]


# We want to perform similar analysis for test  data.
# 
# **Analysis on test data**

# In[ ]:


titanic_test = titanic_test.set_index('PassengerId')
titanic_test = titanic_test.drop(['Cabin','Ticket'], axis=1)
titanic_test.head()


# In[ ]:


titanic_test['Fare']=titanic_test['Fare'].fillna(titanic_test['Fare'].median())
fix, ax = plt.subplots(figsize=(12,8))
sns.distplot(titanic_test.Fare);
table(ax, np.round(titanic_test['Fare'].to_frame().describe()),loc='upper right', colWidths=[0.2, 0.2, 0.2])
ax.set_xlabel('fare');
ax.set_ylabel('density');
ax.set_title('Distribution of titanic_test fares');


# In[ ]:


titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].median())
Age_range_test = []
titanic_Age_int_test = list(map(int,titanic_test.Age.values))
for j in range(len(titanic_Age_int_test)):
    for i in range(len(titanic_Age_range)):
        if titanic_Age_int_test[j] in range(titanic_Age_range[i][0],titanic_Age_range[i][1]):
            age_range_test = titanic_Age_range[i]
        else:
            pass
    Age_range_test.append(age_range_test)
titanic_test['Age_range']=Age_range_test


# In[ ]:


titanic_test['Fare'] = titanic_test['Fare'].fillna(titanic_test['Fare'].median())
Fare_range_test = []
titanic_Fare_int_test = list(map(int,titanic_test.Fare.values))
for j in range(len(titanic_Fare_int_test)):
    for i in range(len(titanic_Fare_range)):
        if titanic_Fare_int_test[j] in range(titanic_Fare_range[i][0],titanic_Fare_range[i][1]):
            fare_range_test = titanic_Fare_range[i]
        else:
            pass
    Fare_range_test.append(fare_range_test)
titanic_test['Fare_range']=Fare_range_test


# In[ ]:


Title_test=[]
for i in range(len(titanic_test)):
    names_test = titanic_test.Name.values[i].replace('.',',').split(',')
    title_test = names_test[1]
    Title_test.append(title_test)    
titanic_test['PTitle'] = Title_test


# In[ ]:


titanic_test_relevant = titanic_test.drop(['Name','Age','Fare'],axis=1)
titanic_test_relevant.head()


# We want to encode some of the categorical values in both train and test set. Hence, it is a good idea to merge train_relevant and test_relevant as:

# In[ ]:


titanic_train_test_merged = titanic_train_relevant.append(titanic_test_relevant)
titanic_train_test_merged.shape


# **Encoding the categorical values**

# In[ ]:


titanic_train_test_merged['Sex']=titanic_train_test_merged['Sex'].astype('category')
sex_code = dict( enumerate(titanic_train_test_merged['Sex'].cat.categories) )
print(sex_code)
titanic_train_test_merged['Sex_cat']=titanic_train_test_merged['Sex'].cat.codes
titanic_train_test_merged.shape


# In[ ]:


titanic_train_test_merged['Embarked']=titanic_train_test_merged['Embarked'].astype('category')
embark_code = dict( enumerate(titanic_train_test_merged['Embarked'].cat.categories) )
print(embark_code)
titanic_train_test_merged['Embarked_cat']=titanic_train_test_merged['Embarked'].cat.codes
titanic_train_test_merged.shape


# In[ ]:


titanic_train_test_merged['PTitle']=titanic_train_test_merged['PTitle'].astype('category')
title_code = dict( enumerate(titanic_train_test_merged['PTitle'].cat.categories) )
print(title_code)
titanic_train_test_merged['PTitle_cat']=titanic_train_test_merged['PTitle'].cat.codes
titanic_train_test_merged['Age_range']=titanic_train_test_merged['Age_range'].astype('category')
age_range_code = dict( enumerate(titanic_train_test_merged['Age_range'].cat.categories) )
print(age_range_code)
titanic_train_test_merged['Age_range_cat']=titanic_train_test_merged['Age_range'].cat.codes
titanic_train_test_merged['Fare_range']=titanic_train_test_merged['Fare_range'].astype('category')
fare_range_code = dict( enumerate(titanic_train_test_merged['Fare_range'].cat.categories) )
print(fare_range_code)
titanic_train_test_merged['Fare_range_cat']=titanic_train_test_merged['Fare_range'].cat.codes
titanic_train_test_merged.shape


# In[ ]:


titanic_train_test_merged = titanic_train_test_merged.drop(['Sex','Embarked','PTitle','Age_range',
                                                    'Fare_range'], axis=1)


# In[ ]:


titanic_train_final=titanic_train_test_merged[~titanic_train_test_merged['Survived'].isnull()]
titanic_test_final=titanic_train_test_merged[titanic_train_test_merged['Survived'].isnull()]


# In[ ]:


titanic_test_final = titanic_test_final.drop(['Survived'], axis=1)


# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
corr_mat = titanic_train_final.corr()
sns.heatmap(corr_mat,vmax=.8, linewidths=0.01,square=True,annot=True,cmap='YlGnBu',linecolor="white");


# ### Prediction and Machine Learning

# In[ ]:


titanic_train_final.head()


# In[ ]:


titanic_test_final.head()


# 
# We have titanic_test dataframe with, survival values not given. Hence, for now, we do nothing with this dataframe. 
# But we perform train/test splitting on the given training dataset, to create a new train set and a validation set; and we  perform KFold cross-validation.  
# 
# We observe different machine learning algorithms for this data set and observe the accuracy in each case.
# 1. Linear Regression
# 2. Logistic Regression
# 3. Random Forest
# 4. SVM
# 5. KNN
# 6. Naive Bayes

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import metrics
linreg = LinearRegression()
logreg = LogisticRegression()
rfclf = RandomForestClassifier()
svc = SVC()
linsvc=LinearSVC()
knn = KNeighborsClassifier()
gnb = GaussianNB()


# In[ ]:


X_train = titanic_train_final.drop('Survived',axis=1)
y_train = titanic_train_final['Survived']


# Let us create a function that calculates the predicted values, using given machine learning algorithms as:

# In[ ]:


def find_predicted_values(X_train, y_train,n, ml_algo):
    #n = number of cross validation folds
    #ml_algo = machine learning algorithms
    predictions = []
    kf = KFold(n_splits=n)
    
    for train, test in kf.split(X_train,y_train):
        train_data = X_train.iloc[train,:]
        target_data = y_train.iloc[train]
        ml_algo.fit(train_data,target_data)
        test_predict = ml_algo.predict(X_train.iloc[test,:])
        predictions.append(test_predict)
    predictions=np.concatenate(predictions)
    return predictions


# In[ ]:


linreg_predictions = find_predicted_values(X_train,y_train,4,linreg)
logreg_predictions = find_predicted_values(X_train,y_train,4,logreg)
rfclassifier_predictions = find_predicted_values(X_train,y_train,4,rfclf)
svc_predictions = find_predicted_values(X_train,y_train,4,svc)
linsvc_predictions = find_predicted_values(X_train,y_train,4,linsvc)
knn_predictions = find_predicted_values(X_train,y_train,4,knn)
gnb_predictions = find_predicted_values(X_train,y_train,4,gnb)


# In[ ]:


linreg_predictions[linreg_predictions > 0.5]=1
linreg_predictions[linreg_predictions <= 0.5]=0
accuracy_linreg = metrics.accuracy_score(y_train,linreg_predictions)
accuracy_linreg


# In[ ]:


accuracy_logreg = metrics.accuracy_score(y_train, logreg_predictions)
accuracy_logreg


# In[ ]:


accuracy_rfclf = metrics.accuracy_score(y_train,rfclassifier_predictions)
accuracy_rfclf


# In[ ]:


accuracy_svc = metrics.accuracy_score(y_train,svc_predictions)
accuracy_svc


# In[ ]:


accuracy_linsvc = metrics.accuracy_score(y_train,linsvc_predictions)
accuracy_linsvc


# In[ ]:


accuracy_knn = metrics.accuracy_score(y_train,knn_predictions)
accuracy_knn


# In[ ]:


accuracy_gnb = metrics.accuracy_score(y_train,gnb_predictions)
accuracy_gnb


# Let us observe, how the accuracy varies with the number of cross validation folds for logistic regression and linear regression.

# In[ ]:


N = 10  # number of folds varies from 1 to 10
accuracy_linreg_kfolds = []
accuracy_logreg_kfolds = []
accuracy_rfclf_kfolds = []
accuracy_svc_kfolds = []
accuracy_linsvc_kfolds = []
accuracy_knn_kfolds = []
accuracy_gnb_kfolds = []
for n in range(2,N+2):
    prediction_linreg_kfold = find_predicted_values(X_train,y_train,n,linreg)
    prediction_logreg_kfold = find_predicted_values(X_train,y_train,n,logreg)
    prediction_rfclf_kfold = find_predicted_values(X_train,y_train,n,rfclf)
    prediction_svc_kfold = find_predicted_values(X_train,y_train,n,svc)
    prediction_linsvc_kfold = find_predicted_values(X_train,y_train,n,linsvc)
    prediction_knn_kfold = find_predicted_values(X_train,y_train,n,knn)
    prediction_gnb_kfold = find_predicted_values(X_train,y_train,n,gnb)
    prediction_linreg_kfold[prediction_linreg_kfold > 0.5]=1
    prediction_linreg_kfold[prediction_linreg_kfold <= 0.5]=0
    
    accuracy_linreg_kfold = metrics.accuracy_score(y_train,prediction_linreg_kfold)
    accuracy_logreg_kfold = metrics.accuracy_score(y_train,prediction_logreg_kfold)
    accuracy_rfclf_kfold = metrics.accuracy_score(y_train,prediction_rfclf_kfold)
    accuracy_svc_kfold = metrics.accuracy_score(y_train,prediction_svc_kfold)
    accuracy_linsvc_kfold = metrics.accuracy_score(y_train,prediction_linsvc_kfold)
    accuracy_knn_kfold = metrics.accuracy_score(y_train,prediction_knn_kfold)
    accuracy_gnb_kfold = metrics.accuracy_score(y_train,prediction_gnb_kfold)
    
    accuracy_linreg_kfolds.append(accuracy_linreg_kfold)
    accuracy_logreg_kfolds.append(accuracy_logreg_kfold)
    accuracy_rfclf_kfolds.append(accuracy_rfclf_kfold)
    accuracy_svc_kfolds.append(accuracy_svc_kfold)
    accuracy_linsvc_kfolds.append(accuracy_linsvc_kfold)
    accuracy_knn_kfolds.append(accuracy_knn_kfold)
    accuracy_gnb_kfolds.append(accuracy_gnb_kfold)


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
sns.set_style("darkgrid")
plt.plot(list(range(2,N+2)),accuracy_linreg_kfolds, label='Linear Regression');
plt.plot(list(range(2,N+2)),accuracy_logreg_kfolds, label='Logistic Regression');
plt.plot(list(range(2,N+2)),accuracy_rfclf_kfolds, label='Random Forest Classifier');
plt.plot(list(range(2,N+2)),accuracy_svc_kfolds, label='Support Vector');
plt.plot(list(range(2,N+2)),accuracy_linsvc_kfolds, label='Linear SVC');
plt.plot(list(range(2,N+2)),accuracy_knn_kfolds, label='K Nearest Neighbors');
plt.plot(list(range(2,N+2)),accuracy_gnb_kfolds, label='Gaussian Naive Bayes');
ax.set_xlabel('Number of cross validation folds')
ax.set_ylabel('accuracy')
handles, labels = ax.get_legend_handles_labels();
ax.legend(handles, labels);


# Support Vector Machine outperforms all other algorithms here.  Let us print the accuracies observed for all algorithms, for cross validation folds 2:11.

# In[ ]:


accuracy_dict={'Linear Reg':accuracy_linreg_kfolds, 'Logistic Reg': accuracy_logreg_kfolds,
               'Random Forest':accuracy_rfclf_kfolds, 'Support Vector': accuracy_svc_kfolds,
              'Linear SVC': accuracy_linsvc_kfolds, 'KNN':accuracy_knn_kfolds, 'GNB':accuracy_gnb_kfolds}
accuracy_table = pd.DataFrame(accuracy_dict)
accuracy_table.index=range(2,N+2)
accuracy_table.index.name='CV Folds'
accuracy_table


# We choose SVC with CV fold 4 for training whole set of train data.

# In[ ]:




