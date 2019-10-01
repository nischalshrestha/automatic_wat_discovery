#!/usr/bin/env python
# coding: utf-8

# # Practicing feature selection with the Titanic database
# # modified from https://www.kaggle.com/sinakhorami/titanic/titanic-best-working-classifier

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import re as re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[ ]:


# GET DATA
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]

# ADD TITLE DESCRIPTOR
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
# FILL WITH MOST LIKELY VALUE (MOST LIKELY FOR ALL ORIGINS)    
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')    
    
# INTEGERIZE DATA
for dataset in full_data:
    # Mapping Name
    dataset["Name"]=dataset["Name"].map(lambda x: len(x))
    
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 2, 'male': 1} ).astype(int)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # MAPPING EMBARKED
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    t=dataset.Ticket.values
    for i in range(len(t)):
        try:
            t[i]=int(t[i].strip().split()[-1])
        except:
            t[i]=0
passengers=test["PassengerId"]


# In[ ]:


# FILL GAPS AND REGULARIZE
def fillgaps(column1,column2,train,test):
    """FILL COLUMN2 WITH MOST LIKELY VALUES BASED ON COLUMN1"""
    ddict={}
    d1=test[[column1,column2]].dropna().values
    d2=train[[column1,column2]].dropna().values
    c1=np.array(d1[:,0].tolist()+d2[:,0].tolist())
    c2=np.array(d1[:,1].tolist()+d2[:,1].tolist())
    for ic1 in np.unique(c1):
        ddict[ic1]=(c2[c1==ic1].mean(),c2[c1==ic1].std())
    full_data = [train, test]
    for dataset in full_data:
        for missing in np.where(np.isnan(dataset[column2]))[0]:
            m,s=ddict[dataset[column1][missing]]
            if s<=0:
                dataset[column2][missing]=m
            else:
                dataset[column2][missing]=np.random.normal(loc=m,scale=s,size=1)
    return (train,test)
train,test=fillgaps("SibSp","Age",train,test)
train,test=fillgaps("Pclass","Fare",train,test)
print(train.info())
print(test.info())
full_data=[train,test]
tm=max(np.max(train.Ticket.values),np.max(test.Ticket.values))
for dataset in full_data:
    w=np.where(dataset.Ticket==0)[0]    
    for i in w:
        dataset.Ticket[w]=dataset.Ticket.median()
    dataset.Ticket=dataset.Ticket/tm
    dataset.Ticket=dataset.Ticket.map(lambda x: float(100*x)).astype(float)


# In[ ]:


# GENERATE BETTER DESCRIPTORS
full_data = [train, test]
for dataset in full_data:
    dataset["Familial Uniqueness"]= np.exp(-dataset.Age/5.)*dataset.Pclass/(dataset['SibSp'] + dataset['Parch'] + 1)
    dataset["Familial Uniqueness"]=dataset["Familial Uniqueness"].map(lambda x: float(20*x))
    dataset["Detail oriented nature"]=dataset.Name/dataset.Sex
    dataset["Detail oriented nature"]=dataset["Detail oriented nature"].map(lambda x: float(x))
    ms=np.array([dataset.Fare[dataset.Pclass==i].mean() for i in np.unique(dataset.Pclass)])
    dataset.Fare=1.*dataset.Fare//ms[dataset.Pclass-1]
    dataset.Fare=dataset.Fare.map(lambda x: float(x))
    dataset.Age=dataset.Age.map(lambda x: float(x))
print(train.info())
print(test.info())


# In[ ]:


#DROP OLD PREDICTORS

# Feature Selection
drop_elements = ["Cabin","PassengerId"]
#drop_elements = ['SibSp','Parch',"Cabin","PassengerId","Ticket"]
dtrain = train.drop(drop_elements, axis = 1)
dtest  = test.drop(drop_elements, axis = 1)

print(dtrain.info())
print(dtest.info())

dtrain = dtrain.values
dtest  = dtest.values


# In[ ]:


classifiers = [
    KNeighborsClassifier(5),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression()]

log_cols = ["Classifier", "Accuracy"]
log  = pd.DataFrame(columns=log_cols)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = dtrain[0::, 1::]
y = dtrain[0::, 0]

acc_dict = {}
for train_index, test_index in sss.split(X, y):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	
	for clf in classifiers:
		name = clf.__class__.__name__
		clf.fit(X_train, y_train)
		train_predictions = clf.predict(X_test)
		acc = accuracy_score(y_test, train_predictions)
		if name in acc_dict:
			acc_dict[name] += acc
		else:
			acc_dict[name] = acc

for clf in acc_dict:
	acc_dict[clf] = acc_dict[clf] / 10.0
	log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
	log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')

sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="r")


# In[ ]:


candidate_classifier=GradientBoostingClassifier()
candidate_classifier.fit(dtrain[0::, 1::], dtrain[0::, 0])
result = candidate_classifier.predict(dtest)
output=open("results.csv",'w')
output.write("PassengerId,Survived\n")
for p,r in zip(passengers,result):
    output.write("{:},{:}\n".format(p,int(r)))
output.close()


# In[ ]:




