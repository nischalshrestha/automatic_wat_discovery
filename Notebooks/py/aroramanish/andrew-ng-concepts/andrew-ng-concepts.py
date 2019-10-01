#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


path = os.getcwd()+'../input/train.csv'


# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


print(data.head())


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


# percent of missing "Age" 
print('Percent of missing "Age" records is %.2f%%' %((data['Age'].isnull().sum()/data.shape[0])*100))


# In[ ]:


ax = data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
data["Age"].plot(kind='density', color='teal')
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[ ]:


# mean age
print('The mean of "Age" is %.2f' %(data["Age"].mean(skipna=True)))
# median age
print('The median of "Age" is %.2f' %(data["Age"].median(skipna=True)))


# In[ ]:


print('Percent of missing "Cabin" records is %.2f%%' %((data['Cabin'].isnull().sum()/data.shape[0])*100))


# 
# ## 2.3. Embarked - Missing Values

# In[ ]:


# percent of missing "Embarked" 
print('Percent of missing "Embarked" records is %.2f%%' %((data['Embarked'].isnull().sum()/data.shape[0])*100))


# In[ ]:


import seaborn as sns
print('Boarded passengers grouped by port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton):')
print(data['Embarked'].value_counts())
sns.countplot(x='Embarked', data=data, palette='Set2')
plt.show()


# In[ ]:


print('The most common boarding port of embarkation is %s.' %data['Embarked'].value_counts().idxmax())


# ## 2.4. Final Adjustments to Data (Train & Test)

# In[ ]:



data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)
data["Embarked"].fillna(data['Embarked'].value_counts().idxmax(), inplace=True)
data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(15,8))
ax = data["Age"].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)
data["Age"].plot(kind='density', color='teal')
ax.legend(['Raw Age'])
ax.set(xlabel='Age')
plt.xlim(-10,85)
plt.show()


# In[ ]:


## Create categorical variable for traveling alone
data['TravelAlone']=np.where((data["SibSp"]+data["Parch"])>0, 0, 1)
data.drop('SibSp', axis=1, inplace=True)
data.drop('Parch', axis=1, inplace=True)


# Also create categorical variables for Passenger Class ("Pclass"), Gender ("Sex"), and Port Embarked ("Embarked"). 

# In[ ]:


#create categorical variables and drop some variables
data=pd.get_dummies(data, columns=["Pclass","Embarked","Sex"])
data.drop('Sex_female', axis=1, inplace=True)
data.drop('PassengerId', axis=1, inplace=True)
data.drop('Name', axis=1, inplace=True)
data.drop('Ticket', axis=1, inplace=True)


data.head()


# ### Now, apply the same changes to the test data. <br>
# I will apply to same imputation for "Age" in the Test data as I did for my Training data (if missing, Age = 28).  <br> I'll also remove the "Cabin" variable from the test data, as I've decided not to include it in my analysis. <br> There were no missing values in the "Embarked" port variable. <br> I'll add the dummy variables to finalize the test set.  <br> Finally, I'll impute the 1 missing value for "Fare" with the median, 14.45.

# In[ ]:


path = os.getcwd()+'../input/test.csv'
test_df = pd.read_csv('../input/test.csv')
test_data = test_df.copy()
test_data["Age"].fillna(data["Age"].median(skipna=True), inplace=True)
test_data["Fare"].fillna(data["Fare"].median(skipna=True), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)

test_data['TravelAlone']=np.where((test_data["SibSp"]+test_data["Parch"])>0, 0, 1)

test_data.drop('SibSp', axis=1, inplace=True)
test_data.drop('Parch', axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass","Embarked","Sex"])
testing.drop('Sex_female', axis=1, inplace=True)
testing.drop('PassengerId', axis=1, inplace=True)
testing.drop('Name', axis=1, inplace=True)
testing.drop('Ticket', axis=1, inplace=True)

final_test = testing
final_test.head()


# In[ ]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[ ]:


def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# In[ ]:


cols = list(data.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Survived')) #Remove b from list
data = data[cols+['Survived']] #Create new dataframe with columns in the order you want
data.head()


# In[ ]:


# add a ones column - this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)




# convert to numpy arrays and initalize the parameter array theta


# In[ ]:


cols = data.shape[1]
cols


# In[ ]:


X = data.iloc[:,0:cols-1]


# In[ ]:


X


# In[ ]:


Y = data.iloc[:,cols-1:cols]



# In[ ]:


Y.shape


# In[ ]:


X.shape


# In[ ]:


theta = np.zeros(11)


# In[ ]:


theta.shape


# In[ ]:


X.shape, theta.shape, Y.shape


# In[ ]:


cost(theta, X, Y)


# In[ ]:


def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    
    error = sigmoid(X * theta.T) - y
    
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)
    
    return grad


# In[ ]:


theta.shape


# In[ ]:


import scipy.optimize as opt
result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(X, Y))
cost(result[0], X, Y)


# In[ ]:



theta_min = np.matrix(result[0])
X= np.matrix(X)
Y=np.matrix(Y)
X.shape,theta.shape, result[0].shape,theta_min.shape,theta_min.T.shape


# In[ ]:


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

theta_min = np.matrix(result[0])
predictions = predict(theta_min, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, Y)]
temp = sum(map(int,correct))
accuracy_test = temp/ len(correct)
print ('accuracy_test = {0}%'.format(accuracy_test*100))


# In[ ]:


Y.shape


# In[ ]:


final_test


# In[ ]:


final_test.insert(0, 'Ones', 1)


# In[ ]:


final_test.insert(11, 'Survived', 1)


# In[ ]:


X_test = final_test.iloc[:,0:cols-1]


# In[ ]:


X_test = np.matrix(X_test)


# In[ ]:


Y_test = final_test.iloc[:,cols-1:cols]


# In[ ]:


Y_test = np.matrix(Y_test)


# In[ ]:


final_test


# In[ ]:


predict(theta_min,X_test)


# In[ ]:


Survived_test = predict(theta_min,X_test)


# In[ ]:


Survived_test= pd.DataFrame(Survived_test)


# In[ ]:


Survived_test.head()


# In[ ]:


final_test['Survived'] = Survived_test


# In[ ]:


final_test['Survived'].count()


# In[ ]:


test_df['PassengerId'].count()


# In[ ]:


final_test['Survived'].head()


# In[ ]:


df1 = pd.DataFrame(test_df['PassengerId'])
df2= pd.DataFrame(final_test['Survived'])
concat = pd.merge(df1,df2, left_index=True, right_index = True)
concat.head()


# In[ ]:


concat.head()


# In[ ]:


concat.to_csv('concat.csv',index=False)


# In[ ]:





# In[ ]:




