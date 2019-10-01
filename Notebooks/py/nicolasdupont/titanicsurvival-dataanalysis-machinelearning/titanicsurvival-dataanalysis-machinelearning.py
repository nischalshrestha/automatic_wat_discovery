#!/usr/bin/env python
# coding: utf-8

# # Titanic-Survival : Exploratory Data Analysis & Machine Learning
# 
# ## Summary :
# **1 | Introduction**
# *     1.1 Introduction to the dataset and the topic
# *     1.2 First datasets observation
# 
# ** 2 | Data Analysis and Visualisation**
# * 2.1 Missing values
# * 2.2 Individual features
# * 2.3 Relation
# * 2.4 Feature engineering
#     * Age
#     * Children
#     * Single/Alone
#     * Family size
#     * Title  
# * 2.5 Correlation
# 
# ** 3 | Machine Learning**
# * 3.1 Impute missing value
#     * Age
#     * Cabin
#     * Embarked
# * 3.2 Encode Categorical features
# * 3.3 Scaling numerical features
# * 3.4 Logistic Regression
# * 3.5 Decision Tree
# * 3.6 Random Forest
# * 3.7 Knn
# * 3.8 Svm
# * 3.9 Svc
# 
# ** 4 | Conlusion **  
# * How to improve our model
# * What is actualy our best model
# * Our model is good for generalisation ?
# 

# In[ ]:


import numpy as np
import pandas as pd
import missingno as msno
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
from collections import OrderedDict #order python dictionnary

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
pal = sns.color_palette("Set2", 10)
sns.set_palette(pal)
#sns.palplot(pal)


# In[ ]:


#import datasets
#import datasets
#TitanicTrain = pd.read_csv("/home/nicolas/Notebook/Data/train.csv")
TitanicTrain = pd.read_csv("../input/train.csv")
TitanicTrain['Type'] = 'Train'
#TitanicSubmission = pd.read_csv("/home/nicolas/Notebook/Data/test.csv")
TitanicSubmission = pd.read_csv("../input/test.csv")
TitanicSubmission['Type'] = 'Test'
TitanicSubmission['Survived'] = np.NaN
Titanic = pd.concat([TitanicTrain,TitanicSubmission], ignore_index=True)


# # 1 | Introduction

# **About this Dataset and the competition**  
# 
# Overview  
# The data has been split into two groups:  
# 
# training set (train.csv)  
# test set (test.csv)  
# 
# The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.  
# 
# The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.  
# 
# We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, as an example of what a submission file should look like.  
# 
# **Data Dictionary** 
# 
# VariableDefinitionKey survival Survival 0 = No, 1 = Yes pclass Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd sex Sex Age Age in years sibsp # of siblings / spouses aboard the Titanic parch # of parents / children aboard the Titanic ticket Ticket number fare Passenger fare cabin Cabin number embarked Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton  
# 
# **Variable Notes**
# 
# pclass: A proxy for socio-economic status (SES)    
# * 1st = Upper
# * 2nd = Middle
# * 3rd = Lower
# 
# age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# sibsp: The dataset defines family relations in this way...  
# Sibling = brother, sister, stepbrother, stepsister  
# Spouse = husband, wife (mistresses and fiancés were ignored)  
# 
# parch: The dataset defines family relations in this way...  
# Parent = mother, father  
# Child = daughter, son, stepdaughter, stepson  
# Some children travelled only with a nanny, therefore parch=0 for them.  

# **Basic informations**

# In[ ]:


TitanicTrain.head(10)


# In[ ]:


#TitanicSubmission.head()
TitanicTrain.info()
TitanicTrain.describe()


# # 2 | Data Analysis and Visualisation
# 
# **2.1 Missing values**

# In[ ]:


msno.matrix(TitanicTrain)
TitanicTrain.isnull().sum()
#msno.bar(TitanicTrain)
#msno.heatmap(TitanicTrain)


# It looks there are a lot of missing values for Age and Cabin and only 2 for Embarked.   
# This is interesting information to impute these missing values later to may be improve our prediction model.
# 

# **2.2 Individual features**

# In[ ]:


columns = TitanicTrain.select_dtypes(include=[np.number]).drop(['PassengerId','Age','Fare'], axis=1).columns.tolist()
#print(columns)


# In[ ]:


#Distribution
fig = plt.figure(figsize=(13, 10))
for i in range (0,len(columns)):
    fig.add_subplot(2,2,i+1)
    sns.countplot(x=columns[i], data=TitanicTrain);            
plt.show()
fig.clear()


# In[ ]:


# Age and Fare distribution
i = 1
fig = plt.figure(figsize=(13, 5))
for col in ['Age','Fare']:
    fig.add_subplot(1,2,i)
    sns.distplot(TitanicTrain[col].dropna(),kde=False);
    i += 1
plt.show()
fig.clear()


# In[ ]:


fig = plt.figure(figsize=(13, 5))
#fig.add_subplot(1,2,1)
#sns.countplot(x='Ticket', data=TitanicTrain);
#plt.show()
fig.add_subplot(1,2,1)
sns.countplot(x='Embarked', data=TitanicTrain);
fig.add_subplot(1,2,2)
sns.countplot(x='Sex', data=TitanicTrain);
plt.show()
fig.clear()


# **2.3 Relation**

# In[ ]:


fig = plt.figure(figsize=(13, 10))
i = 1
for col in columns:
    if col != 'Survived':
        fig.add_subplot(2,2,i)
        sns.countplot(x=col, data=TitanicTrain,hue='Survived');
        i += 1
plt.show()
fig.clear()


# In[ ]:


fig = plt.figure(figsize=(13, 5))
fig.add_subplot(1,2,1)
sns.countplot(x='Embarked', data=TitanicTrain,hue='Survived');
fig.add_subplot(1,2,2)
sns.countplot(x='Sex', data=TitanicTrain,hue='Survived');
plt.show()
fig.clear()


# In[ ]:


fig = plt.figure(figsize=(13, 10))
# Age and Survived
fig.add_subplot(2,2,1)
sns.swarmplot(x="Survived", y="Age", hue="Sex", data=TitanicTrain);
fig.add_subplot(2,2,2)
sns.boxplot(x="Survived", y="Age", data=TitanicTrain)
# fare and Survived
fig.add_subplot(2,2,3)
sns.violinplot(x="Survived", y="Fare", data=TitanicTrain)
plt.show()
fig.clear()


# ** 2.4 Feature engineering**

# We going to create some new features from the dataset to see if they can more explain why a passenger survived or not

# In[ ]:


# Age group
def AgeGroup(age):
    ag = ""
    if age <= 10:
        ag = ":10"
    elif age <= 20:
        ag = "11:20"
    elif age <= 30:
        ag = "21:30"
    elif age <= 40:
        ag = "31:40"
    elif age <= 50:
        ag = "41:50"
    elif age <= 60:
        ag = "51:60"
    elif age <= 60:
        ag = "61:70"
    else:
        ag = "71:"
    return ag
# Mjor or not (I assume the legal age is 18..)
def IsMajor(age):
    if age < 18:
        return 0
    else:
        return 1

Titanic["AgeGroup"] = Titanic.apply(lambda row: AgeGroup(row["Age"]), axis=1)
Titanic["Major"] = Titanic.apply(lambda row: IsMajor(row["Age"]), axis=1)
TitanicTrain = Titanic[Titanic.Type=='Train']
#TitanicTrain.head(1)


# In[ ]:


fig = plt.figure(figsize=(13, 5))
# Plot AgeGroup and Major vs Survivedl
fig.add_subplot(1,2,1)
sns.countplot(x='AgeGroup', data=TitanicTrain,hue='Survived');
fig.add_subplot(1,2,2)
sns.countplot(x='Major', data=TitanicTrain,hue='Survived');
plt.show()
fig.clear()


# In[ ]:


# is Alone or not (boolean value if a passenger travel alone or not)
Titanic["Alone"] = Titanic.apply(lambda obs: 1 if np.sum(obs['SibSp']+obs['Parch']) == 0 else 0, axis=1)
TitanicTrain = Titanic[Titanic.Type=='Train']
#TitanicTrain.Alone.value_counts()
# Plot Alone vs Survived
sns.countplot(x='Alone', data=TitanicTrain,hue='Survived');
plt.show()


# It is interesting to see that when you are alone, you are less likely to survive.

# In[ ]:


# Extract the title of a passenger (Mr,Miss,Mme,Mrs,etc)

title = ['Mlle','Mrs', 'Mr', 'Miss','Master','Don','Rev','Dr','Mme','Ms','Major','Col','Capt','Countess']
def ExtractTitle(name):
    tit = 'missing'
    for item in title :
        if item in name:
            tit = item
    if tit == 'missing':
        tit = 'Mr'
    return tit
"""
title = ['Mrs','Mr', 'Miss' ,'Master']
def ExtractTitle(name):
    tit = 'other'
    for item in title:
        if item in name:
            tit = item
    return tit
"""
Titanic["Title"] = Titanic.apply(lambda row: ExtractTitle(row["Name"]),axis=1)
TitanicTrain = Titanic[Titanic.Type=='Train']
# in 1 ligne but not perfect..
#TitanicTrain["Title"] = TitanicTrain.Name.apply(lambda obs: obs.split(' ')[1])
#TitanicTrain.head(1)
TitanicTrain.Title.value_counts()
#TitanicTrain[TitanicTrain["Title"].isnull()].head(4)


# In[ ]:


plt.figure(figsize=(13, 5))
sns.countplot(x='Title', data=TitanicTrain,hue='Survived');
plt.show()


# Actualy I'm not sure that the "Title" feature give me more information than the sex feature !

# In[ ]:


#Family size
Titanic["Fsize"] = Titanic['SibSp']+Titanic['Parch']+1
TitanicTrain = Titanic[Titanic.Type=='Train']


# In[ ]:


fig = plt.figure(figsize=(13, 5))


# ** 2.5 Correlation**

# In[ ]:


# correlations with the new features
corr = TitanicTrain.drop(['PassengerId'], axis=1).corr()
#sns.set(style="white")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(13, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,annot=True, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})


# # 3 | Machine Learning
# 
# **3.1 Impute missing values**

# In[ ]:


# Age
MeanAge = TitanicTrain.Age.mean()
#print(MeanAge)
MedianAge = TitanicTrain.Age.median()
#print(MedianAge)
Titanic.Age = Titanic.Age.fillna(value=MedianAge)

Titanic["AgeGroup"] = Titanic.apply(lambda row: AgeGroup(row["Age"]), axis=1)
Titanic["Major"] = Titanic.apply(lambda row: IsMajor(row["Age"]), axis=1)


# In[ ]:


# Cabin
Titanic["Cabin"] = Titanic.apply(lambda obs: "No" if pd.isnull(obs['Cabin']) else "Yes", axis=1)
TitanicTrain = Titanic[Titanic.Type=='Train']
plt.figure(figsize=(9, 5))
sns.countplot(x='Cabin', data=TitanicTrain,hue='Survived');
plt.show()


# In[ ]:


#TitanicTrain.isnull().sum()


# In[ ]:


# Embarked
# replace NaN with the mode value
ModeEmbarked = TitanicTrain.Embarked.mode()[0]
#print(type(ModeEmbarked))
#print(ModeEmbarked)
Titanic.Embarked = Titanic.Embarked.fillna(value=ModeEmbarked)
print(Titanic.Embarked.value_counts())


# In[ ]:


#Titanic.isnull().sum()


# In[ ]:


# Fare have 1 NaN missing value on the Submission dataset
MedianFare = TitanicTrain.Fare.median()
Titanic.Fare = Titanic.Fare.fillna(value=MedianFare)
#Titanic.isnull().sum()


# **3.2 Encode Categorical Features**

# In[ ]:


print(Titanic.columns)


# In[ ]:


SubmissionPassengerId = Titanic[Titanic.Type=='Test']['PassengerId']
Titanic = pd.get_dummies(Titanic.drop(['PassengerId','Name','Ticket'],axis=1),drop_first=True,columns=['AgeGroup','Sex','Title','Cabin','Embarked'])
print(Titanic.columns.tolist())


# In[ ]:


Titanic.head(2)


# In[ ]:


#Titanic.isnull().sum()


# **3.3 Scaling numerical features**

# In[ ]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler().fit(Titanic[['Age', 'Fare']])
Titanic[['Age', 'Fare']] = scale.transform(Titanic[['Age', 'Fare']])


# In[ ]:


Titanic.head(4)


# In[ ]:


# plot sorting correlations 
TitanicTrain = Titanic[Titanic.Type=='Train']
dfcor = TitanicTrain.corr()["Survived"].to_frame().sort_values(by=['Survived'],ascending=False)
dfcor.reset_index(level=0, inplace=True)
dfcor.columns = ['Feature', 'Coeff']
dfcor = dfcor[dfcor.Feature != "Survived"]
#print(dfcor.head())
plt.figure(figsize=(15, 5))
sns.barplot(x='Feature', y='Coeff', data= dfcor);
plt.xticks(rotation=90) 
plt.show()


# In[ ]:


#print(TitanicTrain.corr()["Survived"])
Target = TitanicTrain.Survived
Features = TitanicTrain.drop(['Survived','Type'],axis=1)


# In[ ]:


# Create training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size = 0.3, random_state=42)
Target = TitanicTrain.Survived
"""
X_train = TitanicTrain.drop(['Survived','Type'],axis=1)
X_test = TitanicTrain.drop(['Survived','Type'],axis=1)
y_train = TitanicTrain.Survived
y_test = TitanicTrain.Survived
"""


# In[ ]:


MlRes= {}
def MlResult(model,score):
    MlRes[model] = score
    print(MlRes)
#MlResult('AlgoName',accuracyscore)


# In[ ]:


roc_curve_data = {}
def ConcatRocData(algoname, fpr, tpr, auc):
    data = [fpr, tpr, auc]
    roc_curve_data[algoname] = data


# In[ ]:


"""
#We need to have same columns on the training dataset and the test dataset
def AlignShape(dfok,dfko):
    colok = dfok.columns.tolist()
    colko = dfko.columns.tolist()
    for c in colok:
        if c not in colko:
            dfko[c] = 0
    return dfko
TitanicSubmission= AlignShape(TitanicTrain,TitanicSubmission)
"""
TitanicSubmission = Titanic[Titanic.Type=='Test'].drop(['Survived','Type'],axis=1)


# **3.4 Logistic Regression Classifier**

# In[ ]:


# Import Logistic Regression from scikit learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve


# In[ ]:


# Logistic Regression : 
logi_reg = LogisticRegression()
# Fit the regressor to the training data
logi_reg.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = logi_reg.predict(X_test)


# In[ ]:


# Score / Metrics
accuracy = logi_reg.score(X_test, y_test) # = accuracy
print("Score Logistic Regression : {}".format(accuracy))
print("Accuracy Logistic Regression : {}".format(accuracy_score(y_test, y_pred)))
auc = roc_auc_score(y_test, y_pred)
print("Roc auc score Logistic Regression: {}".format(auc))


# In[ ]:


MlResult('Logistic Regression',accuracy)


# In[ ]:


# Confusion Matrix with sklearn :
print("Confusion matrix and Classification Report for the Logistic Regression model :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
cla = classification_report(y_test, y_pred)
print(cla)


# 85% of not survival (0) prediction are good  => 152 / (152+27)  
# 80% of survival (1)  prediction are good => 93 / (93+23)

# In[ ]:


# Compute predicted probabilities with the logistic regression model: y_pred_prob
y_pred_prob = logi_reg.predict_proba(X_test)[:,1]
# Generate ROC curve values:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
ConcatRocData('Logistic Regression', fpr, tpr, auc)


# In[ ]:


# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Logistic Regression')
plt.show()


# In[ ]:


#submission
logi_reg_prediction = logi_reg.predict(TitanicSubmission)
submission = pd.DataFrame({
     "PassengerId": SubmissionPassengerId,  #id from the test dataset
        "Survived": logi_reg_prediction.astype(int)      #prediction compute from the test dataset
    })
#submission.to_csv('logi_reg_submission.csv', index=False)


# **3.5 Decision Tree Classifier**

# In[ ]:


# Import Decision Tree Classifier from scikit learn
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc = DecisionTreeClassifier()
# Fit the regressor to the training data
dtc.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = dtc.predict(X_test)


# In[ ]:


# Score / Metrics
accuracy = dtc.score(X_test, y_test) # = accuracy
print("Score Decision Tree : {}".format(accuracy))
print("Accuracy Decision Tree : {}".format(accuracy_score(y_test, y_pred)))
auc = roc_auc_score(y_test, y_pred)
print("Roc auc score Decision Tree : {}".format(auc))


# In[ ]:


MlResult('Decision Tree',accuracy)


# In[ ]:


# Confusion Matrix with sklearn :
print("Confusion matrix and Classification Report for the Decision Tree model :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
cla = classification_report(y_test, y_pred)
print(cla)


# In[ ]:


# Compute predicted probabilities with the Decision tree model: y_pred_prob
y_pred_prob = dtc.predict_proba(X_test)[:,1]
#print(y_pred_prob.shape)
#print(y_pred_prob[5:])
# Generate ROC curve values:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
ConcatRocData('Decision Tree', fpr, tpr, auc)


# In[ ]:


#Features importance
def FeaturesImportance(data,model):
    features = data.columns.tolist()
    fi = model.feature_importances_
    sorted_features = {}
    for feature, imp in zip(features, fi):
        sorted_features[feature] = round(imp,3)

    # sort the dictionnary by value
    sorted_features = OrderedDict(sorted(sorted_features.items(),reverse=True, key=lambda t: t[1]))

    #for feature, imp in sorted_features.items():
        #print(feature+" : ",imp)

    dfvi = pd.DataFrame(list(sorted_features.items()), columns=['Features', 'Importance'])
    #dfvi.head()
    plt.figure(figsize=(15, 5))
    sns.barplot(x='Features', y='Importance', data=dfvi);
    plt.xticks(rotation=90) 
    plt.show()

FeaturesImportance(TitanicTrain,dtc)


# In[ ]:


#submission
dtc_prediction = dtc.predict(TitanicSubmission)
submission = pd.DataFrame({
     "PassengerId": SubmissionPassengerId,  #id from the test dataset
        "Survived": dtc_prediction.astype(int)      #prediction compute from the test dataset
    })
#submission.to_csv('dtc_submission.csv', index=False)


# In[ ]:


# compare roc curve from multiple models
def PlotMultipleRocCurve(roc_curve_data): 
    plt.figure(figsize=(10, 10))
    for algo,rocdata in roc_curve_data.items():
        plt.plot(rocdata[0], rocdata[1], label=str(format(rocdata[2], '.3f'))+" : "+algo)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve and Auc Score for each Model')
    plt.legend(borderaxespad=0.5,loc=4)
    plt.show()


# **3.6 Random Forest Classifier**

# In[ ]:


# Import Random Forest Classifier scikit learn
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# Create a random Forest Classifier instance
rfc = RandomForestClassifier(n_estimators=100)
# Fit to the training data
rfc.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = rfc.predict(X_test)


# In[ ]:


# Score / Metrics
accuracy = rfc.score(X_test, y_test) # = accuracy
print("Score Random Forest on test Data : {}".format(accuracy))
print("Score Random Forest on training Data : {}".format(rfc.score(X_train, y_train)))
print("Accuracy Random Forest : {}".format(accuracy_score(y_test, y_pred)))
auc = roc_auc_score(y_test, y_pred)
print("Roc auc score Random Forest : {}".format(auc))


# In[ ]:


MlResult('Random Forest',accuracy)


# In[ ]:


# Confusion Matrix with sklearn :
print("Confusion matrix and Classification Report for the Random Forest model :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
cla = classification_report(y_test, y_pred)
print(cla)


# In[ ]:


#Features importance
FeaturesImportance(TitanicTrain,rfc)


# In[ ]:


# Compute predicted probabilities with the Random Forest model: y_pred_prob
y_pred_prob = rfc.predict_proba(X_test)[:,1]
#print(y_pred_prob.shape)
#print(y_pred_prob[5:])
# Generate ROC curve values:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
ConcatRocData('Random Forest', fpr, tpr, auc)


# In[ ]:


# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Random Forest')
plt.show()


# In[ ]:


#submission
rfc_prediction = rfc.predict(TitanicSubmission)
submission = pd.DataFrame({
     "PassengerId": SubmissionPassengerId,  #id from the test dataset
        "Survived": rfc_prediction.astype(int)      #prediction compute from the test dataset
    })
#submission.to_csv('rfc_submission.csv', index=False)


# **3.7 Knn Classifier**

# In[ ]:


# Import KNeighbors Classifier from scikit learn
from sklearn.neighbors import KNeighborsClassifier 


# In[ ]:


# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier()
# Fit to the training data
knn.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = knn.predict(X_test)


# In[ ]:


# Score / Metrics
accuracy = knn.score(X_test, y_test) # = accuracy
print("Score KNeighbors on test data : {}".format(accuracy))
print("Score KNeighbors on train data : {}".format(knn.score(X_train, y_train)))
print("Accuracy KNeighbors on test data : {}".format(accuracy_score(y_test, y_pred)))
auc = roc_auc_score(y_test, y_pred)
print("Roc auc score KNeighbors on test data : {}".format(auc))


# In[ ]:


MlResult('KNeighbors',accuracy)


# In[ ]:


# Confusion Matrix with sklearn :
print("Confusion matrix and Classification Report for the KNeighbors model :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
cla = classification_report(y_test, y_pred)
print(cla)


# In[ ]:


# Compute predicted probabilities with the Random Forest model: y_pred_prob
y_pred_prob = knn.predict_proba(X_test)[:,1]
#print(y_pred_prob.shape)
#print(y_pred_prob[5:])
# Generate ROC curve values:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
ConcatRocData('KNeighbors', fpr, tpr, auc)


# In[ ]:


# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve KNeighbors')
plt.show()


# In[ ]:


#submission
knn_prediction = knn.predict(TitanicSubmission)
submission = pd.DataFrame({
     "PassengerId": SubmissionPassengerId,  #id from the test dataset
        "Survived": knn_prediction.astype(int)      #prediction compute from the test dataset
    })
#submission.to_csv('knn_submission.csv', index=False)


# **3.8 Support Vector Machines (SVM)**

# In[ ]:


# Import svm Classifier from scikit learn
from sklearn.svm import SVC, LinearSVC


# In[ ]:


# Create a k-NN classifier with 6 neighbors: knn
svm =SVC(probability=True)
# Fit to the training data
svm.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = svm.predict(X_test)


# In[ ]:


# Score / Metrics
accuracy = svm.score(X_test, y_test) # = accuracy
print("Score Support Vector Machines on test data : {}".format(accuracy))
print("Score Support Vector Machines on train data : {}".format(svm.score(X_train, y_train)))
print("Accuracy Support Vector Machines on test data : {}".format(accuracy_score(y_test, y_pred)))
auc = roc_auc_score(y_test, y_pred)
print("Roc auc score Support Vector Machines on test data : {}".format(auc))


# In[ ]:


MlResult('Support Vector Machines',accuracy)


# In[ ]:


# Confusion Matrix with sklearn :
print("Confusion matrix and Classification Report for the Support Vector Machines model :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
cla = classification_report(y_test, y_pred)
print(cla)


# In[ ]:


# Compute predicted probabilities with the Support Vector Machines model: y_pred_prob
y_pred_prob = svm.predict_proba(X_test)[:,1]
#print(y_pred_prob.shape)
#print(y_pred_prob[5:])
# Generate ROC curve values:
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
ConcatRocData('Support Vector Machines', fpr, tpr, auc)


# In[ ]:


# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Support Vector Machines')
plt.show()


# In[ ]:


#submission
svm_prediction = svm.predict(TitanicSubmission)
submission = pd.DataFrame({
     "PassengerId": SubmissionPassengerId,  #id from the test dataset
        "Survived": svm_prediction.astype(int)      #prediction compute from the test dataset
    })
#submission.to_csv('svm_submission.csv', index=False)


# **3.9 Linear SVC**

# In[ ]:


# Create a k-NN classifier with 6 neighbors: knn
svc = LinearSVC()
# Fit to the training data
svc.fit(X_train, y_train)
# Predict on the test data: y_pred
y_pred = svc.predict(X_test)


# In[ ]:


# Score / Metrics
accuracy = svc.score(X_test, y_test) # = accuracy
print("Score Linear SVC on test data : {}".format(accuracy))
print("Score Linear SVC on train data : {}".format(svc.score(X_train, y_train)))
print("Accuracy Linear SVC on test data : {}".format(accuracy_score(y_test, y_pred)))
auc = roc_auc_score(y_test, y_pred)
print("Roc auc score Linear SVC on test data : {}".format(auc))


# In[ ]:


MlResult('Linear SVC',accuracy)


# In[ ]:


# Confusion Matrix with sklearn :
print("Confusion matrix and Classification Report for the Linear SVC model :")
cm = confusion_matrix(y_test,y_pred)
print(cm)
cla = classification_report(y_test, y_pred)
print(cla)


# In[ ]:


#submission
svc_prediction = svc.predict(TitanicSubmission)
submission = pd.DataFrame({
     "PassengerId": SubmissionPassengerId,  #id from the test dataset
        "Survived": svc_prediction.astype(int)      #prediction compute from the test dataset
    })
#submission.to_csv('svc_submission.csv', index=False)


# **Conclusion :**

# In[ ]:


# Roc curve plot with all models
PlotMultipleRocCurve(roc_curve_data)


# In[ ]:


# print score for each model
#for algo,score in MlRes.items():
#    print(algo+" : {}".format(round(score,3)))
res = pd.DataFrame(list(MlRes.items()), columns=['Model', 'Score']).sort_values("Score", ascending=False)
print(res)


# In[ ]:


plt.figure(figsize=(8, 5))
sns.barplot(x='Score', y='Model', data=res);
plt.xticks(rotation=90) 
plt.xlim([0.6, 0.9])
plt.show()


# Decision three and Random forest seems to give us our best score when I use the all training dataset to fit model but it's actualty clearly overfitted.  
# When I use train test split whith 20,25,30%... for testing model on a test dataset not used for training the model, the decision three and the random forest become our badest models..  
# Svm seems for to be the betterone and give me the best predictions for the competition.

# To improve our code it will be good to use cross validation because our training dataset have very low number of observations to training and to testing at the same time with a classic test train split and it's not good to use all our data to fit a model.  
# It would be nice to reduce the number of features. As you can see some features have a zero importance.  
# We can also tune parameters for each model.

# **This is my first attemp on machine learning so be indulgent ! 
# Give me feedback if you want to improve the notebook !**
