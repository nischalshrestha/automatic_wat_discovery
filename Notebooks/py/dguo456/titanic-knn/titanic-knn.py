#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# **Load data**

# In[ ]:


dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')


# **Observe the dataset**

# In[ ]:


dataset.columns


# In[ ]:


print(dataset.shape, testset.shape)
dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


dataset.dtypes


# In[ ]:


dataset.describe()


# **Feature selection**

# In[ ]:


print(dataset.Age[1:10]) # observe feature 'Age'
dataset.Age.hist()
plt.xlabel('Age')
plt.ylabel('Number')
plt.show()


# In[ ]:


dataset[dataset.Survived==0]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of not Survived')
plt.show()

dataset[dataset.Survived==1]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of Survived')
plt.show()


# In[ ]:


# feature 'Sex'
Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("sex") 
plt.ylabel("count")
plt.show()


# In[ ]:


dataset['Fare'].hist()  # feature 'Fare'
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution')
plt.show()


# In[ ]:


# feature 'Pclass'
Survived_p1 = dataset.Survived[dataset['Pclass'] == 1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass'] == 2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass'] == 3].value_counts()

df=pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()


# In[ ]:


# feature 'Embarked'
Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("Embarked") 
plt.ylabel("count")
plt.show()


# **Based on above we are gonna keep those features:  Age, Pclass, Sex, Fare, Embarked**

# In[ ]:


# split label and features
label=dataset.loc[:,'Survived']
featured_data=dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat=testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(featured_data.shape)
print(featured_data)


# **Processing empty data**

# In[ ]:


def fill_NAN(data):  
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

featured_data_no_nan = fill_NAN(featured_data)
testdat_no_nan = fill_NAN(testdat)

print(featured_data.isnull().values.any())   
print(featured_data_no_nan.isnull().values.any())
print(testdat.isnull().values.any())    
print(testdat_no_nan.isnull().values.any())    

print(featured_data_no_nan)


# **Deal with feature_columns**

# In[ ]:


print(featured_data_no_nan['Sex'].isnull().values.any())

def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

featured_data_after_sex = transfer_sex(featured_data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(featured_data_after_sex)


# In[ ]:


def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

featured_data_after_embarked = transfer_embark(featured_data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)
print(featured_data_after_embarked)


# **After processing our training data, we begin to train our model**
# 
# **First we do a train_test_split**

# In[ ]:


training_data_final = featured_data_after_embarked
test_data_final = testdat_after_embarked
from sklearn.model_selection import train_test_split

train_data,test_data,train_labels,test_labels=train_test_split(training_data_final,label,random_state=0,test_size=0.2)
print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)


# **Training our model**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# compare and select the optimal k value
k_range = range(1, 51)
k_scores = []
for K in k_range:
    clf=KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data,train_labels)
    print('K=', K)
    predictions=clf.predict(test_data)
    score = accuracy_score(test_labels,predictions)
    print(score)
    k_scores.append(score)


# In[ ]:


plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())


# **Validating our model on test dataset**

# In[ ]:


clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(training_data_final,label)
result=clf.predict(test_data_final)
print(result)


# **Print out the final submission csv file**

# In[ ]:


df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True)

