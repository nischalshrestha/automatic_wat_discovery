#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv("../input/train.csv")
submission_test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


round( (train.isnull().sum() / len(train)) * 100, 2)


# In[ ]:


round( (submission_test.isnull().sum() / len(submission_test)) * 100, 2)


# In[ ]:


len(train[train.Age.isna()])


# In[ ]:


train['Age'].mean()


# In[ ]:


submission_test['Age'].mean()


# In[ ]:


train.Age.fillna(29.5, inplace=True)
submission_test.Age.fillna(30.5, inplace=True)


# In[ ]:


train.isnull().sum()


# In[ ]:


submission_test.isnull().sum()


# In[ ]:


train[['PassengerId', 'Embarked']].groupby("Embarked").count()


# In[ ]:


train.Embarked.fillna('S', inplace=True)


# In[ ]:


train.isnull().sum()


# In[ ]:


submission_test.Fare.fillna(submission_test.Fare.mean(), inplace=True)


# In[ ]:


submission_test.isnull().sum()


# In[ ]:


train['family_size'] = train.SibSp + train.Parch
submission_test['family_size'] = submission_test.SibSp + submission_test.Parch


# In[ ]:


train.drop(['Ticket', 'Name', 'Cabin'], axis=1, inplace=True)
submission_test.drop(['Ticket', 'Name', 'Cabin'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


submission_test.head()


# In[ ]:


plt.figure(num=1, figsize=(10, 4), dpi=100, facecolor='w', edgecolor='k')


plt.subplot(1, 3, 1)
sns.boxplot(y="Age", data=train)

plt.subplot(1, 3, 2)
sns.distplot(train.Age, bins=10)

plt.subplot(1, 3, 3)
plt.hist("Age", data=train) 

plt.show()


# In[ ]:


plt.figure(num=2, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')


plt.subplot(1, 3, 1)
sns.barplot(x="Sex", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival based on Gender")

plt.subplot(1, 3, 2)
sns.barplot(x="Pclass", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Class")

plt.subplot(1, 3, 3)
sns.barplot(x="Embarked", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Embarked")


plt.show()


# In[ ]:


plt.figure(num=3, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 2, 1)
sns.barplot(x="Pclass", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival based on Gender & PClass")

plt.subplot(1, 2, 2)
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Gender and Class")


plt.show()


# In[ ]:


plt.figure(num=4, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 3, 1)
sns.barplot(x="family_size", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on FamilySize")

plt.subplot(1, 3, 2)
sns.barplot(x="Parch", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on Parch")

plt.subplot(1, 3, 3)
sns.barplot(x="SibSp", y="Survived", data=train)
plt.ylabel("Survival Rate")
plt.title("Survival Based on SibSp")


plt.show()


# In[ ]:


plt.figure(num=5, figsize=(16, 4), dpi=80, facecolor='w', edgecolor='k')


plt.subplot(1, 3, 1)
sns.distplot(train.Fare)

plt.subplot(1, 3, 2)
sns.swarmplot(x='Pclass', y='Fare', hue='Survived',data=train)

plt.subplot(1, 3, 3)
sns.boxplot(x='Pclass', y='Fare', hue="Survived", data=train)

plt.show()


# In[ ]:


train.head()


# In[ ]:


submission_test.head()


# In[ ]:





# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[ ]:


X=train.iloc[:, 2:].values
y=train.iloc[:, 1].values


# In[ ]:


submission_X = submission_test.iloc[:, 1:].values


# In[ ]:


le_x = LabelEncoder()
X[:, 1] = le_x.fit_transform( X[:, 1] )
X[:, 6] = le_x.fit_transform(X[:, 6])

submission_X[:, 1] = le_x.fit_transform( submission_X[:, 1] )
submission_X[:, 6] = le_x.fit_transform( submission_X[:, 6] )


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features = [0, 1, 6])


# In[ ]:


X = onehotencoder.fit_transform(X).toarray()

submission_X = onehotencoder.fit_transform(submission_X).toarray()


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


X = scaler.fit_transform(X)

submission_X = scaler.fit_transform(submission_X)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


# In[ ]:


from sklearn.metrics import confusion_matrix


# # SVM - Linear

# In[ ]:


from sklearn.svm import SVC
classifier1 = SVC(kernel = 'linear')
classifier1.fit(X_train, y_train)


# In[ ]:


y_pred = classifier1.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])


# In[ ]:


acc_svm_linear = round(classifier1.score(X_test, y_test) * 100, 2)


# # SVM - Kernel RBF

# In[ ]:


classifier2 = SVC(kernel = 'rbf')
classifier2.fit(X_train, y_train)


# In[ ]:


y_pred = classifier2.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])


# In[ ]:


acc_svm_rbf = round(classifier2.score(X_test, y_test) * 100, 2)


# # SVM - Kernel Poly

# In[ ]:


classifier3 = SVC(kernel = 'poly')
classifier3.fit(X_train, y_train)


# In[ ]:


y_pred = classifier3.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])


# In[ ]:


acc_svm_poly = round(classifier3.score(X_test, y_test) * 100, 2)


# # SVM - Kernel Sigmoid

# In[ ]:


classifier4 = SVC(kernel = 'sigmoid')
classifier4.fit(X_train, y_train)


# In[ ]:


y_pred = classifier4.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])


# In[ ]:


acc_svm_sigmoid = round(classifier4.score(X_test, y_test) * 100, 2)


# # Ramdom Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_rf.fit(X_train, y_train)


# In[ ]:


y_pred = classifier_rf.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])


# In[ ]:


acc_svm_rf = round(classifier_rf.score(X_test, y_test) * 100, 2)


# # SVM - Logistics Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression()
classifier_lr.fit(X_train, y_train)


# In[ ]:


y_pred = classifier_rf.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


pd.DataFrame(cm, ['Actual: NOT', 'Actual: SURVIVED'], ['Predicted: NO', 'Predicted: SURVIVED'])


# In[ ]:


acc_svm_lr = round(classifier_lr.score(X_test, y_test) * 100, 2)


# In[ ]:





# ## Compraing accuracies

# In[ ]:


accs = {'Classifiers':['SVM-L', 'SVM-RBF', 'SVM-Poly', 'SVM-Sigmoid', 'Random_Forest', 'Logistic_Regression'],
       'Accuracy':[acc_svm_linear, acc_svm_rbf, acc_svm_poly, acc_svm_sigmoid, acc_svm_rf, acc_svm_lr]}


# In[ ]:


acc_df = pd.DataFrame.from_dict(accs)


# In[ ]:


plt.figure(num=6, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

sns.barplot(y="Accuracy", x="Classifiers",  data=acc_df)
plt.show()


# In[ ]:


submission_X_predict = classifier2.predict(submission_X)
submission_X_survived = pd.Series(submission_X_predict, name="Survived")
Submission = pd.concat([submission_test.PassengerId, submission_X_survived],axis=1)


# In[ ]:


Submission.head()


# In[ ]:


Submission.to_csv('submission.csv', index=False)


# In[ ]:




