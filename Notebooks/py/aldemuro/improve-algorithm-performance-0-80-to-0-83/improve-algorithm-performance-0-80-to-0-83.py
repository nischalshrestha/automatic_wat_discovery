#!/usr/bin/env python
# coding: utf-8

# # Introduction

# I have deided to work with the Titanic dataset again. this kernel is focusing on comparing the performance of several machine learning algorithms, find the best fit algorithm, and tune its parameter to get better score. 
# I am hoping to learn a lot from this site, so feedback is very welcome! This kernel is always improving because of your feedback!!!
# 
# There are three parts to my script as follows:
# 
# 1. Load the library and data
# 2. Data cleaning
# 3. Data spliting
# 4. Training,testing, and Peformance comparison
# 5. Tuning the algorithm
# 
# If you like this work and want to see my other works, you can check it here:
# 
# https://www.kaggle.com/aldemuro
# 

# # 1. Load the library and data
# 
# In this section the library and the data used are loaded into the sytem
# 
# ## 1.1 Load the library

# In[ ]:


#sklearn
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#load package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from math import sqrt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# ## 1.2 Load the data

# In[ ]:


## Read in file
train_original = pd.read_csv('../input/train.csv')
test_original = pd.read_csv('../input/test.csv')
train_original.sample(10)
total = [train_original,test_original]


# In[ ]:


#exploration
train_original.info()
print("----------------------------")
test_original.info()


# # 2. Data Cleaning
# ## 2.1 Retrive the salutation and Eliminating unused variable
# 
# 'Salutation' variable can be retrieved from 'Name' column by taking the string between space string and '.' string.

# In[ ]:


#Retrive the salutation from 'Name' column
for dataset in total:
    dataset['Salutation'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)    


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x="Salutation", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Survive or not in each salutation')

plt.show()


# afterwards, 'Salutation' column should be factorized to be fit in our future model. before factorizing them, we can put Mll and Ms into Miss. Mme can be combined with Mr Based on the graph above, we can see a significantly different number of count. It changes dramatically after Master plot. so all of the low-value counts of salutation will be categorized into the same category.

# In[ ]:


#grouping the low-value data
for dataset in total:
    dataset['Salutation'] = dataset['Salutation'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Salutation'] = dataset['Salutation'].replace('Mlle', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Ms', 'Miss')
    dataset['Salutation'] = dataset['Salutation'].replace('Mme', 'Mrs')
    #dataset['Salutation'] = pd.factorize(dataset['Salutation'])[0]
    


#total.Salutation = pd.factorize(total.Salutation)[0]   


# In[ ]:


plt.subplots(figsize=(15,6))
sns.countplot(x="Salutation", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Survive or not in each salutation after')

plt.show()


# the plot above shows the comparison between survival or not in every salutation. as we can see the number of Mr who are survive is way bigger than those who not. while category Mrs is another way around.

# In[ ]:


#Factorize the salutation
for dataset in total:    
    dataset['Salutation'] = pd.factorize(dataset['Salutation'])[0]


# In[ ]:


plt.subplots(figsize=(8,6))
sns.countplot(x="Sex", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Survive or not in every gender')
plt.show()


# The next step is deletin column that will not be used in our models.

# In[ ]:


#clean unused variable
train=train_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test=test_original.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
total = [train,test]

train.shape, test.shape


# ## 2.2 Detect and fill the missing data

# In[ ]:


#Detect the missing data in 'train' dataset
train.isnull().sum()


# As it is shown above, there are 2 columns which have missing data. the way I'm handling missing 'Age' column is by filling them by the median of age in every salutation group. there are only two data missing in 'Embarked' column. Considering Sex=female and Fare=80, Ports of Embarkation (Embarked) for two missing cases can be assumed to be Cherbourg (C).

# In[ ]:


## Create function to replace missing data with the median value
def fill_missing_age(dataset):
    for i in range(1,8):
        median_age=dataset[dataset["Salutation"]==i]["Age"].median()
        dataset["Age"]=dataset["Age"].fillna(median_age)
        return dataset

train = fill_missing_age(train)


# In[ ]:


plt.subplots(figsize=(8,6))
sns.distplot(train.Age)
plt.xticks(rotation=90)
plt.title('Distribution of Passenger Age')
plt.show()


# In[ ]:


plt.subplots(figsize=(8,6))
sns.distplot(train.Fare)
plt.xticks(rotation=90)
plt.title('Distribution of fare')

plt.show()


# In[ ]:


## Embarked missing cases 
train[train['Embarked'].isnull()]


# In[ ]:


train["Embarked"] = train["Embarked"].fillna('C')


# In[ ]:


plt.subplots(figsize=(8,6))
sns.countplot(x="Embarked", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Survive or not in ship point of embarkation')
plt.show()


# In[ ]:


plt.subplots(figsize=(8,6))
sns.countplot(x="Pclass", hue='Survived', data=train_original,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('Survive or not in pasenger class')
plt.show()


# Detecting the missing data in 'test' dataset is done to get the insight which column consist missing data. as it is shown below, there are 2 column which have missing value. they are 'Age' and 'Fare' column. The same function is used in order to filled the missing 'Age' value. missing 'Fare' value is filled by finding the median of 'Fare' value in the 'Pclass' = 3 and 'Embarked' = S.

# In[ ]:


test.isnull().sum()


# In[ ]:


#apply the missing age method to test dataset
test = fill_missing_age(test)


# In[ ]:


#filling the missing 'Fare' data with the  median
def fill_missing_fare(dataset):
    median_fare=dataset[(dataset["Pclass"]==3) & (dataset["Embarked"]=="S")]["Fare"].median()
    dataset["Fare"]=dataset["Fare"].fillna(median_fare)
    return dataset

test = fill_missing_fare(test)


# ## 2.3 Re-Check for missing data

# In[ ]:


## Re-Check for missing data
train.isnull().any()


# In[ ]:


## Re-Check for missing data
test.isnull().any()


# discretize Age feature

# In[ ]:


pd.qcut(train["Age"], 6).value_counts()


# In[ ]:



for dataset in total:
    dataset.loc[dataset["Age"] <= 19, "Age"] = 0
    dataset.loc[(dataset["Age"] > 19) & (dataset["Age"] <= 25), "Age"] = 1
    dataset.loc[(dataset["Age"] > 25) & (dataset["Age"] <= 32), "Age"] = 2
    dataset.loc[(dataset["Age"] > 32) & (dataset["Age"] <= 35), "Age"] = 3
    dataset.loc[(dataset["Age"] > 35) & (dataset["Age"] <= 40.5), "Age"] = 4
    dataset.loc[dataset["Age"] > 40.59, "Age"] = 5


# Discretize Fare

# In[ ]:


pd.qcut(train["Fare"], 8).value_counts()


# In[ ]:


for dataset in total:
    dataset.loc[dataset["Fare"] <= 7.75, "Fare"] = 0
    dataset.loc[(dataset["Fare"] > 7.75) & (dataset["Fare"] <= 7.91), "Fare"] = 1
    dataset.loc[(dataset["Fare"] > 7.91) & (dataset["Fare"] <= 9.841), "Fare"] = 2
    dataset.loc[(dataset["Fare"] > 9.841) & (dataset["Fare"] <= 14.454), "Fare"] = 3   
    dataset.loc[(dataset["Fare"] > 14.454) & (dataset["Fare"] <= 24.479), "Fare"] = 4
    dataset.loc[(dataset["Fare"] >24.479) & (dataset["Fare"] <= 31), "Fare"] = 5   
    dataset.loc[(dataset["Fare"] > 31) & (dataset["Fare"] <= 69.487), "Fare"] = 6
    dataset.loc[dataset["Fare"] > 69.487, "Fare"] = 7


# Factorized 2 of the column whic are 'Sex' and 'Embarked'

# In[ ]:


for dataset in total:
    dataset['Sex'] = pd.factorize(dataset['Sex'])[0]
    dataset['Embarked']= pd.factorize(dataset['Embarked'])[0]
train.head()


# Checking the correlation between features

# # 3. Spliting the data

# Seperate input features from target feature

# In[ ]:


x = train.drop("Survived", axis=1)
y = train["Survived"]


# Split the data into training and validation sets

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=1)


# # 4. Performance Comparison

# List of Machine Learning Algorithm (MLA) used

# In[ ]:



MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model. RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
   #tree.ExtraTreeClassifier(),
    
    ]


# Train the data into the model and calculate the performance

# In[ ]:


MLA_columns = []
MLA_compare = pd.DataFrame(columns = MLA_columns)


row_index = 0
for alg in MLA:
    
    
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index,'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(x_train, y_train), 4)
    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(x_test, y_test), 4)
    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(y_test, predicted)
    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)





    row_index+=1
    
MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    
MLA_compare


# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Train Accuracy",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Train Accuracy Comparison')
plt.show()


# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Test Accuracy",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Test Accuracy Comparison')
plt.show()


# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Precission",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Precission Comparison')
plt.show()


# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA Recall",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA Recall Comparison')
plt.show()


# In[ ]:


plt.subplots(figsize=(15,6))
sns.barplot(x="MLA Name", y="MLA AUC",data=MLA_compare,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('MLA AUC Comparison')
plt.show()


# In[ ]:


index = 1
for alg in MLA:
    
    
    predicted = alg.fit(x_train, y_train).predict(x_test)
    fp, tp, th = roc_curve(y_test, predicted)
    roc_auc_mla = auc(fp, tp)
    MLA_name = alg.__class__.__name__
    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))
   
    index+=1

plt.title('ROC Curve comparison')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')    
plt.show()


# # 5. Tuning the algorithm

# In[ ]:


tunealg = ensemble.AdaBoostClassifier() #Select the algorithm to be tuned
tunealg.fit(x_train, y_train)

print('BEFORE tuning Parameters: ', tunealg.get_params())
print("BEFORE tuning Training w/bin set score: {:.2f}". format(tunealg.score(x_train, y_train))) 
print("BEFORE tuning Test w/bin set score: {:.2f}". format(tunealg.score(x_test, y_test)))
print('-'*10)



# In[ ]:


#tune parameters
param_grid = {'n_estimators': [10,15,25,35,45,50,55,60,65], 
              'learning_rate': [0.1,0.2,0.3,0.4,0.5,1.0],
              'algorithm': ['SAMME','SAMME.R'],                
              'random_state':  [1,2,3,4,5,50, None], 
              
             }
# So, what this GridSearchCV function do is finding the best combination of parameters value that is set above.
tune_model = model_selection.GridSearchCV(ensemble.AdaBoostClassifier(), param_grid=param_grid, scoring = 'roc_auc')
tune_model.fit (x_train, y_train)

print('AFTER tuning Parameters: ', tune_model.best_params_)
print("AFTER tuning Training w/bin set score: {:.2f}". format(tune_model.score(x_train, y_train))) 
print("AFTER tuning Test w/bin set score: {:.2f}". format(tune_model.score(x_test, y_test)))
print('-'*10)


# # 6. Submission

# In[ ]:


#re-train the model with un-split data
y_pred = tune_model.fit(x, y).predict(test)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_original["PassengerId"],
        "Survived": y_pred
    })
submission.to_csv('titanic.csv', index=False)


# In[ ]:




