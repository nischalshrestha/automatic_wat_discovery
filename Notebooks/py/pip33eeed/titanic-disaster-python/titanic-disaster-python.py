#!/usr/bin/env python
# coding: utf-8

# **TITANIC Dataset**
# 
# Objective: To predict if a person survived the titanic disaster or not, provided information about him/her.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# **Basic overview of data**
# 
# Notice that some missing values specifically in Age, Cabin and Embarked are observed. We have some data that is in 'text' form and needs to be encoded numerically (most importantly the 'Sex' or Gender column that is binary).

# In[ ]:


data_train = pd.read_csv('../input/train.csv')
display(data_train.head())
display(data_train.info())
display(data_train.describe())


# **Frequency distribution of the labels (Survived column)**
# 
# We see that we are not dealing with a skewed classes as there is an appreciable amount of data from both labels (0 and 1).

# In[ ]:


import matplotlib.pyplot as plt
plt.hist(data_train['Survived'] )
print("Number of survived people: {} \nNumber of not survived people: {}".format(len(data_train[data_train['Survived'] == 1]),len(data_train[data_train['Survived'] == 0])))


# **Visualising the relationship between Survival and few features such as Age, Gender and PassengerClass**
# 
# In first plot we notice that the first class passengers of any age had good chance to survive as compared to the third class passengers where only few younger people survived.

# In[ ]:


plt.figure(figsize = (12,6))
s = plt.scatter(data_train['Age'], data_train['Pclass'], c = data_train['Survived'], alpha = 0.5, cmap = 'binary')
plt.colorbar(s)
plt.xlabel('Age')
plt.ylabel('Passenger Class')
plt.legend('Survived',loc='best')


# In the second plot we observe that women, irrespective of their age had more chance of survival than men. Only children in Male category seems to survive significantly.

# In[ ]:


plt.figure(figsize = (12,6))
s = plt.scatter(data_train['Age'], data_train['Sex'], c = data_train['Survived'], alpha = 0.5, cmap = 'binary')
plt.colorbar(s)
plt.xlabel('Age')
plt.ylabel('Gender')
plt.legend('Survived',loc='best')


# **Transforming the dataset, Dealing with Age feature having missing data**
# 
# We use regular expression to gather the salutation such as 'Mr.', 'Ms.', 'Mrs.' from the Name feature and group them. To predict the age, we don't take the mean of every non-NaN example but we look at the salutation and then we compute the mean age for each salutation and assign it to the respective NaN value.

# In[ ]:


def transformer(data_train = data_train):
    from numpy import array

    names = ''
    for i in data_train['Name']:
        names += ' '+i

    import re
    pattern = re.compile(r',\s\w+')
    matches = pattern.finditer(names)
    salutations = []
    for match in matches:
        if(match.group().lstrip(', ') not in salutations):
            salutations.append(match.group().lstrip(', '))

    print(salutations)

    salutationDF = []
    for sal in data_train['Name']:
        salutationDF.append(re.search(r',\s\w+',sal).group().lstrip(', '))
 
    print(len(salutationDF))


    ageDF = pd.DataFrame(salutationDF)
    ageDF['Age'] = data_train['Age']
    ageDF.columns = ['Salutation','Age']
    a = ageDF['Salutation'].value_counts()

    means = []
    for sal in salutations:
        means.append(array(ageDF[ageDF['Salutation'] == sal]['Age'].dropna()).mean())
    print(means)

    dict_sal = {}
    c = 0
    for sal in salutations:
        dict_sal[sal] = means[c]
        c+=1
    dict_sal    

    for sal in salutations:
        ageDF.ix[ageDF.Salutation == sal, 'Age'] = ageDF[ageDF['Salutation'] == sal]['Age'].fillna(dict_sal[sal])

    ageDF.describe()

    data_train['Age'] = ageDF['Age']
    display(data_train.describe())
    display(data_train.info())
    dataTrain = data_train.drop(columns = ['PassengerId','SibSp','Name','Ticket','Cabin','Embarked'])
    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    dataTrain['Sex'] = encoder.fit_transform(dataTrain['Sex'])
    
    return dataTrain


# In[ ]:


dataTrain = transformer(data_train)
labels = dataTrain['Survived']
dataTrain = dataTrain.drop(columns = ['Survived'])
display(dataTrain.info())


# **Scaling the data**
# 
# We use standard scaler to scale the data

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
dataTrainScaled = scaler.fit_transform(dataTrain)
dataTrainScaled = pd.DataFrame(dataTrainScaled)
display(dataTrainScaled.head())


# **Splitting the training and test set**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataTrainScaled, labels, test_size = 0.2)


# **Support Vector Classifier**

# In[ ]:


from sklearn.svm import SVC
svc = SVC(C = 10)
svc.fit(X_train, y_train)
predictions_train = svc.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix, roc_auc_score


# **Standard metrics to evaluate the results**

# In[ ]:


def metricsPrint(predictions_train, y_test):
    accuracyTrain = accuracy_score(y_test, predictions_train)
    precisionTrain = precision_score(y_test, predictions_train)
    recallTrain = recall_score(y_test, predictions_train)
    print("Accuracy {} Precision {} Recall {}".format(accuracyTrain,precisionTrain, recallTrain))


# **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(X_train, y_train)
predictions_train2 = forest.predict(X_test)


# In[ ]:


metricsPrint(predictions_train, y_test)
metricsPrint(predictions_train2, y_test)


# ***Hyperparameter tuning using GreadSearchCV to find best parameters*** (SVC)

# In[ ]:


#from sklearn.model_selection import GridSearchCV
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2],
                     #'C': [100, 500, 1000, 2000]},
                    #{'kernel': ['linear'], 'C': [100, 500, 1000, 2000]}]
#gcvSVM = GridSearchCV(svc, param_grid = tuned_parameters)
#gcvSVM.fit(X_train, y_train)
#params = gcvSVM.best_params_
#print(params)


# In[ ]:


svcT = SVC(C = 2000, gamma = 0.01, kernel = 'rbf')
svcT.fit(X_train, y_train)
predNew = svcT.predict(X_test)
metricsPrint(y_test, predNew)


# ***Hyperparameter tuning using GreadSearchCV to find best parameters*** (Random Forest)

# In[ ]:


#tuned_parameters2 = {'n_estimators': [10,100,500,1000], 'max_depth': [1,10,100,1000],
                     #'min_samples_split': [2, 10, 100, 1000]}
                   
#gcvDT = GridSearchCV(forest, param_grid = tuned_parameters2)
#gcvDT.fit(X_train, y_train)
#print(gcvDT.best_params_)


# In[ ]:


forestT = RandomForestClassifier(max_depth = 10, min_samples_split = 10, n_estimators = 100)
forestT.fit(X_train, y_train)
predNew2 = forestT.predict(X_test)
metricsPrint(y_test, predNew2)


# **Transforming the test just like training set**

# In[ ]:


from numpy import array
data_test = pd.read_csv('../input/test.csv')
data_testTransformed = transformer(data_test)
display(data_test.head())
data_testTransformed['Age'] = data_testTransformed['Age'].fillna((array(data_testTransformed.loc[data_testTransformed['Age'].isna() == False, 'Age'])).mean())
data_testTransformed['Fare'] = data_testTransformed['Fare'].fillna((array(data_testTransformed.loc[data_testTransformed['Fare'].isna() == False, 'Fare'])).mean())
display(data_testTransformed.info())


# In[ ]:


display(data_testTransformed.head())


# In[ ]:


dataTestScaled = scaler.fit_transform(data_testTransformed)
dataTestScaled = pd.DataFrame(dataTestScaled)
predictionsTest = forestT.predict(dataTestScaled)
predictionsTest2 = svcT.predict(dataTestScaled)


# In[ ]:


results = pd.DataFrame(predictionsTest)
results2 = pd.DataFrame(predictionsTest2)
results['PassengerId'] = data_test['PassengerId']
results2['PassengerId'] = data_test['PassengerId']
results.columns = ['Survived','PassengerId']
results2.columns = ['Survived','PassengerId']
results = results[['PassengerId','Survived']]
results2 = results2[['PassengerId','Survived']]
results.to_csv('SUBMISSION11.CSV', index = False)
results2.to_csv('SUBMISSION22.CSV', index = False)


# **ROC curve score (Area under curve)**

# In[ ]:


def rocCurve(fpr,tpr,fpr2,tpr2):
    plt.plot(fpr,tpr,'b-', label = 'svc')
    plt.plot(fpr2,tpr2,'r-', label = 'tree')
    plt.plot([0,1],[0,1], 'g--')
    plt.legend(loc = 'best')
from sklearn.model_selection import cross_val_predict
confidenceValues = cross_val_predict(svcT, X_test, y_test, cv = 3, method = 'decision_function')
confidenceValues2 = cross_val_predict(forestT, X_test, y_test, cv = 3, method = 'predict_proba')

fpr, tpr, thresholds = roc_curve(y_test, confidenceValues)
fpr2, tpr2, thresholds2 = roc_curve(y_test, confidenceValues2[:,1])

rocCurve(fpr,tpr,fpr2,tpr2)

rocScore1 = roc_auc_score(y_test, confidenceValues)
rocScore2 = roc_auc_score(y_test, confidenceValues2[:,1])
print("ROC AUC Score for 1. SVC {}\n2. TREE {}".format(rocScore1,rocScore2))


# In[ ]:


conMat = confusion_matrix(y_test,predictions_train)
import seaborn as sn
df_cm = pd.DataFrame(conMat, index = [i for i in [0,1]],
                  columns = [i for i in [0,1]])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)

