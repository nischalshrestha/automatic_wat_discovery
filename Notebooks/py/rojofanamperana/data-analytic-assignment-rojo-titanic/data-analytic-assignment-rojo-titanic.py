#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib as plt
import pandas as pd
import seaborn as sns


# In[ ]:



# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


#read the data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


data=[train,test]


# In[ ]:


train.head()


# In[ ]:


#check missing data
train.info()


# In[ ]:


test.head()


# In[ ]:


test.info()


# In[ ]:





# In[ ]:


#sex : boolean = (female) for train and test
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} )


# In[ ]:


#Engineering information from Name, I need the title.
for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# In[ ]:


#new feature: alone or not
for dataset in data:
    dataset["Alone"] = (dataset['SibSp']+dataset['Parch'] >=1)


# In[ ]:





# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


for dataset in data:
    dataset['Alone']= dataset['Alone'].map({True:1 , False: 0})


# In[ ]:





# In[ ]:


#training information.
pd.crosstab(train['Title'],train['Survived'])


# In[ ]:





# In[ ]:





# In[ ]:


#how the survival depends on the title
meantitle = train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[ ]:


meantitle 


# In[ ]:





# In[ ]:





# In[ ]:





# I change the titles into the mean of survived

# In[ ]:


for i in range(17):
    train['Title']= train['Title'].replace(meantitle['Title'][i], meantitle['Survived'][i])


# In[ ]:


train.head()


# In[ ]:


test['Title']


# In[ ]:


#Dona is not a title in the training set
test['Title']= test['Title'].replace('Dona', 'Miss')


# In[ ]:


#changing type of titles in the dataset test
for i in range(418):
    for j in range(17):
        if test['Title'][i]== meantitle['Title'][j]:
            test['Title'][i]=meantitle['Survived'][j]
     
    
test['Title']=test['Title'].astype(float)


# In[ ]:


test.info()


# In[ ]:





# In[ ]:


#fill the missing data Fare with the mean from the training data
test['Fare'].fillna(train['Fare'].mean(),inplace=True)


# In[ ]:


test.info()


# In[ ]:





# In[ ]:


meanage= train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
meanage


# In[ ]:


meanparch=  train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
meanparch


# In[ ]:


train[['Embarked','Survived']].groupby(['Embarked'], as_index=False).mean()


# In[ ]:


#embarked : 1,2,3
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map( {'C': 1, 'Q': 2, 'S':3} )


# In[ ]:





# In[ ]:


train['Embarked'].fillna(3,inplace=True)


# In[ ]:





# In[ ]:





# In[ ]:


# new data: exploring the relation sex-title-survived (strongly)
nwdata1= train.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

nwdata1.head()


# In[ ]:


nwdata1.describe()


# In[ ]:


nwdata1.info()


# In[ ]:


#shuffle data
from sklearn.utils import shuffle
nwdata1 = shuffle(nwdata1)


# In[ ]:


#training and confirmation sets
train_data= nwdata1[:850]
confirm_data= nwdata1[850:]


# In[ ]:


#input-output
X_train= train_data.drop(["Survived", "Age"], axis= 1)
X_confirm= confirm_data.drop(["Survived","Age"], axis = 1)
Y_train= train_data["Survived"]
X_test= test.drop(["PassengerId","Name","Ticket","Cabin",'Age'], axis=1)


# In[ ]:





# In[ ]:





# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_lr = logreg.predict(X_confirm)
Y_test_lr= logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# accuracy for the logistic regression
sum(Y_pred_lr == confirm_data["Survived"])/len(Y_pred_lr)


# In[ ]:


#prediction of the test
output_lr = pd.DataFrame({'PassengerId': test['PassengerId'] , 'Survived': Y_test_lr})
output_lr


# In[ ]:





# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_confirm)
Y_test_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


#accuracy prediction for decision tree

sum(Y_pred_dt == confirm_data["Survived"])/len(Y_pred_dt)


# In[ ]:


#prediction of the test
output_dt = pd.DataFrame({'PassengerId': test['PassengerId'] , 'Survived': Y_test_dt})
output_dt


# In[ ]:





# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_confirm)
Y_test_rf = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


# prediction accuracy for random forest
sum(Y_pred_rf == confirm_data["Survived"])/len(Y_pred_rf)


# In[ ]:


#prediction of the test
output_rf = pd.DataFrame({'PassengerId': test['PassengerId'] , 'Survived': Y_test_rf})
output_rf


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svm = svc.predict(X_confirm)
Y_test_svm = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


# prediction accuracy for support vector machines
sum(Y_pred_svm == confirm_data["Survived"])/len(Y_pred_svm)


# In[ ]:


#prediction of the test
output_svm = pd.DataFrame({'PassengerId': test['PassengerId'] , 'Survived': Y_test_svm})
output_svm


# In[ ]:


import keras 
from keras.models import Sequential # intitialize the ANN
from keras.layers import Dense      # create layers


# In[ ]:


keras.__version__


# In[ ]:


# Initialising the NN
model = Sequential()

#X_train = nwdata1.drop(['Survived'], axis=1).values
#y_train = nwdata1['Survived'].values

# layers
model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 8))
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, Y_train, epochs = 150)


# In[ ]:


y_pred = model.predict(X_confirm)
Y_test_ann_step = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_confirm.shape[0])
Y_test_ann = (Y_test_ann_step > 0.5).astype(int).reshape(X_test.shape[0])
output = pd.DataFrame({'Predicted': y_final, 'Survived': confirm_data['Survived']})


# In[ ]:


sum(output.Predicted.values == output.Survived.values)/len(output)


# In[ ]:


#prediction of the test
output_ann = pd.DataFrame({'PassengerId': test['PassengerId'] , 'Survived': Y_test_ann})
output_ann


# In[ ]:


#consider ages
nwdata1.head()


# In[ ]:


nwdata1.describe()


# In[ ]:


nwdata1[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()


# In[ ]:


nwdata1[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


for dataset in data:
    for i in range(len(dataset['Age'])):
            if np.isnan( dataset['Age'][i]):
                if dataset['SibSp'][i]==1:
                    dataset['Age'][i] = 1
                elif dataset['SibSp'][i]>1:
                    dataset['Age'][i] = 0
                elif dataset['SibSp'][i]== 0:
                    if dataset['Parch'][i]== 0:
                        dataset['Age'][i]= 2
                    elif dataset['Parch'][i] >= 4:
                        dataset['Age'][i] = 4
                    else:
                        dataset['Age'][i] = 3
            else:
                if 0 <= dataset['Age'][i] < 16:
                       dataset['Age'][i]= 0
                elif 16 <= dataset['Age'][i] < 32:
                      dataset['Age'][i]= 1
                elif 32 <= dataset['Age'][i] < 48:
                      dataset['Age'][i]= 2
                elif 48 <= dataset['Age'][i] < 64:
                      dataset['Age'][i]=3
                else:
                      dataset['Age'][i]= 4


# In[ ]:


nwdata2= train.drop(['PassengerId','Name','Ticket','Cabin'], axis= 1)


# In[ ]:


#nwdata2= shuffle(nwdata2)


# In[ ]:


nwdata2.head(10)


# In[ ]:


test.info()


# In[ ]:


#training and confirmation sets
train_data= nwdata2[:800]
confirm_data= nwdata2[800:]


# In[ ]:


#input-output
X_train= train_data.drop(["Survived"], axis= 1)
X_confirm= confirm_data.drop(["Survived"], axis = 1)
Y_train= train_data["Survived"]
X_test= test.drop(['PassengerId','Name','Ticket', 'Cabin'], axis = 1)


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_lr = logreg.predict(X_confirm)
Y_test_lr = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


# accuracy for the logistic regression
sum(Y_pred_lr == confirm_data["Survived"])/len(Y_pred_lr)


# In[ ]:


# prediction for the test data
output_lr = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': Y_test_lr })
output_lr


# In[ ]:


output_lr.to_csv('logreg2.csv', index= False)


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_dt = decision_tree.predict(X_confirm)
Y_test_dt = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


#accuracy prediction for decision tree

sum(Y_pred_dt == confirm_data["Survived"])/len(Y_pred_dt)


# In[ ]:


#prediction for the test data
output_dt = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': Y_test_dt})
output_dt


# In[ ]:


#output_dt.to_csv('decision_tree_result97.csv')


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_confirm)
Y_test_rf = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


# prediction accuracy for random forest
sum(Y_pred_rf == confirm_data["Survived"])/len(Y_pred_rf)


# In[ ]:


#prediction for the test data
output_rf = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': Y_test_rf})
output_rf


# In[ ]:





# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svm = svc.predict(X_confirm)
Y_test_svm = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:





# In[ ]:


# prediction accuracy for support vector machines
sum(Y_pred_svm == confirm_data["Survived"])/len(Y_pred_svm)


# In[ ]:


#prediction for the test data
output_svm = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': Y_test_svm})
output_svm


# In[ ]:


output_svm.to_csv('svm2.csv', index= False)


# In[ ]:


# Initialising the NN
model = Sequential()

#X_train = nwdata1.drop(['Survived'], axis=1).values
#y_train = nwdata1['Survived'].values

# layers
model.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9))
model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'sigmoid'))
model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Train the ANN
model.fit(X_train, Y_train, epochs = 200)


# In[ ]:


y_pred = model.predict(X_confirm)
Y_test_ann_step = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_confirm.shape[0])
Y_test_ann = (Y_test_ann_step > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'Predicted': y_final, 'Survived': confirm_data['Survived']})


# In[ ]:


sum(output.Predicted.values == output.Survived.values)/len(output)


# In[ ]:


#prediction for the test data
output_ann = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived': Y_test_ann})
output_ann


# In[ ]:


output_ann.to_csv('seventh_ann.csv', index= False)


# In[ ]:


get_ipython().system(u'ls')

