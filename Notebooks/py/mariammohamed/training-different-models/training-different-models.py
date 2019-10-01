#!/usr/bin/env python
# coding: utf-8

# *This is my first solution, please send me if there is an error*

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# gender_submission = pd.read_csv('../input/gender_submission.csv')


# In[ ]:


#drop unnecessary columns
train_data.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True, axis=1)


# In[ ]:


train_data.head()


# In[ ]:


train_data.describe()


# In[ ]:


plt.hist(train_data['Pclass'], color='lightblue')
plt.tick_params(top='off', bottom='on', left='off', right='off', labelleft='on', labelbottom='on')
plt.xlim([0, 4])
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.set_xticks([1, 2, 3])
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()


# In[ ]:


plt.hist(train_data['Survived'], color='lightblue')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
plt.grid(True)
plt.xlim([-1, 2])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not survived', 'Survived'], rotation='vertical')
plt.ylabel('Count')
plt.tight_layout()


# In[ ]:


train_data['Pclass'].unique()


# In[ ]:


y_surv = [len(train_data[((train_data['Survived'] == 1) & (train_data['Pclass'] == 1))]['Pclass'].tolist()), len(train_data[((train_data['Survived'] == 1) & (train_data['Pclass'] == 2))]['Pclass'].tolist()), len(train_data[((train_data['Survived'] == 1) & (train_data['Pclass'] == 3))]['Pclass'].tolist())]
y_not_surv = [len(train_data[((train_data['Survived'] == 0) & (train_data['Pclass'] == 1))]['Pclass'].tolist()), len(train_data[((train_data['Survived'] == 0) & (train_data['Pclass'] == 2))]['Pclass'].tolist()), len(train_data[((train_data['Survived'] == 0) & (train_data['Pclass'] == 3))]['Pclass'].tolist())]
y_surv , y_not_surv


# In[ ]:


x = np.array([1, 2, 3])
width=0.3
fig, ax = plt.subplots()
bar1 = ax.bar(x - width, y_surv, width, color='lightblue', label='Survived')
bar2 = ax.bar(x, y_not_surv, width, color='pink', label='Not survived')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.set_xticks([1, 2, 3])
plt.xlim([0, 4])
plt.ylabel('Count')
plt.grid(True)
plt.legend(loc='upper left')


# In[ ]:


sum(train_data['Age'].isnull()) / len(train_data)


# In[ ]:


sum(train_data[train_data['Survived']==1]['Age'].isnull()) / len(train_data)


# In[ ]:


sum(train_data[train_data['Survived']==0]['Age'].isnull()) / len(train_data)


# In[ ]:


mean_age = np.mean(train_data['Age'])
train_data['Age'] = train_data['Age'].fillna(mean_age)


# In[ ]:


train_data['Age_group'] = pd.cut(train_data['Age'], 10)


# In[ ]:


counts = train_data.groupby(['Age_group', 'Survived']).Age_group.count().unstack()
# plt.bar(counts['', stacked=True, color=['lightblue', 'pink'])
counts.plot(kind='bar', stacked=True, color=['lightblue', 'pink'])
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
plt.grid(True)


# In[ ]:


sum(train_data['Embarked'].isnull())


# In[ ]:


train_data['Embarked'].value_counts()


# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna('S')


# In[ ]:


y_surv_2 = [len(train_data[((train_data['Survived'] == 1) & (train_data['Embarked'] == 'S'))]['Embarked'].tolist()), len(train_data[((train_data['Survived'] == 1) & (train_data['Embarked'] == 'C'))]['Embarked'].tolist()), len(train_data[((train_data['Survived'] == 1) & (train_data['Embarked'] == 'Q'))]['Embarked'].tolist())]
y_not_surv_2 = [len(train_data[((train_data['Survived'] == 0) & (train_data['Embarked'] == 'S'))]['Embarked'].tolist()), len(train_data[((train_data['Survived'] == 0) & (train_data['Embarked'] == 'C'))]['Embarked'].tolist()), len(train_data[((train_data['Survived'] == 0) & (train_data['Embarked'] == 'Q'))]['Embarked'].tolist())]
y_surv_2 , y_not_surv_2


# In[ ]:


x = np.array([1, 2, 3])
width=0.3
fig, ax = plt.subplots()
bar1 = ax.bar(x - width, y_surv_2, width, color='lightblue', label='Survived')
bar2 = ax.bar(x, y_not_surv_2, width, color='pink', label='Not survived')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['S', 'C', 'Q'])
plt.xlim([0, 4])
plt.ylabel('Count')
plt.grid(True)
plt.legend(loc='upper right')


# In[ ]:


# This makes the model worse
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# train_data[['Fare', 'Age']] = sc.fit_transform(train_data[['Fare', 'Age']])


# In[ ]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[ ]:


labelEncoder_sex = LabelEncoder()
train_data['Sex'] = labelEncoder_sex.fit_transform(train_data['Sex'])


# In[ ]:


labelEncoder_embarked = LabelEncoder()
train_data['Embarked'] = labelEncoder_embarked.fit_transform(train_data['Embarked'])


# In[ ]:


train_data.drop(['Age_group'], inplace=True, axis=1)


# In[ ]:


train_data.head(10)


# In[ ]:


X = train_data.iloc[:, 1:8].values
y = train_data['Survived'].values


# In[ ]:


oneHotEncoder = OneHotEncoder(categorical_features=[0, 6])
X = oneHotEncoder.fit_transform(X).toarray()


# In[ ]:


# avoiding the dummy variable trap
X = X[:, [1, 2, 3, 4, 6, 7, 8, 9, 10]]


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)


# In[ ]:


#prepare the test data
# test_data_cp = test_data.copy()
# test_data_cp.drop(['PassengerId', 'Ticket', 'Cabin', 'Name'], inplace=True, axis=1)
# test_data_cp['Age'] = test_data_cp['Age'].fillna(mean_age)
# test_data_cp['Embarked'] = test_data_cp['Embarked'].fillna('S')


# In[ ]:


# test_data_cp.describe()


# In[ ]:


# mean_fare = np.mean(train_data['Fare'])
# test_data_cp['Fare'] = test_data_cp['Fare'].fillna(mean_fare)
# train_data[['Fare', 'Age']] = sc.fit_transform(train_data[['Fare', 'Age']])


# In[ ]:


# test_data_cp['Sex'] = labelEncoder_sex.transform(test_data_cp['Sex'])
# test_data_cp['Embarked'] = labelEncoder_embarked.transform(test_data_cp['Embarked'])

# X_test = test_data_cp.iloc[:, :].values
# y_test = gender_submission['Survived'].values

# X_test = oneHotEncoder.transform(X_test).toarray()

# X_test = X_test[:, [1, 2, 3, 4, 6, 7, 8, 9, 10]]


# In[ ]:


y_pred = classifier.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm


# In[ ]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm


# In[ ]:


#logistic regression seems the best


# In[ ]:




