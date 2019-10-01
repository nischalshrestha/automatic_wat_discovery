#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt # 画图常用库


# In[ ]:


dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')


# In[ ]:


dataset.columns


# 0, PassengerId：乘客的数字id
# 
# 1, Survived：幸存(1)、死亡(0)
# 
# 2, Pclass：乘客船层—1st = Upper，2nd = Middle， 3rd = Lower
# 
# 3, Name：名字。
# 
# 4, Sex：性别
# 
# 5, Age：年龄
# 
# 6, SibSp：兄弟姐妹和配偶的数量。
# 
# 7, Parch：父母和孩子的数量。
# 
# 8, Ticket：船票号码。
# 
# 9, Fare：船票价钱。
# 
# 10, Cabin：船舱。
# 
# 11, Embarked：从哪个地方登上泰坦尼克号。 C = Cherbourg, Q = Queenstown, S = Southampton

# #### Data analysis

# In[ ]:


dataset.head()


# In[ ]:


dataset.info()


# In[ ]:


print(dataset.dtypes)


# In[ ]:


print(dataset.describe())


# Two interesting founds in the dataset:
# 
# 1.  it has NAN data
# 2. corelation between data

# ### Raw data analysis

# In[ ]:


dataset['Survived'].value_counts(normalize=True)


# In[ ]:


sns.countplot(dataset['Survived'])


# Sex - Survival

# In[ ]:


print (dataset['Sex'][0:10])
Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

print(Survived_m, Survived_f)
df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df
df.plot(kind='bar', stacked=True)
plt.title("Survived by sex")
plt.xlabel("Live") 
plt.ylabel("Count")
plt.show()


# Age - Survival

# In[ ]:


print (dataset['Age'][0:10])   # 等同于 dataset.Age[0:10]
dataset['Age'].hist()  ## == dataset.Age.hist()
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.show()  

print(dataset['Age'].isnull().values.any())  #.values是把pandas对象剥开，返回其numpy对象。这里values其实不加也可以，可以直接.any

plt.scatter(dataset['Survived'], dataset['Age'])
plt.ylabel("Age") 
plt.xlabel("Survived") 
plt.title("Age Distribution")
plt.show()


# Fare - Survival

# In[ ]:


print (dataset['Fare'][0:10])
dataset['Fare'].hist()  
dataset.Fare.max()
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.show()  

survived_0 =dataset.Fare[dataset.Survived == 0]
survived_0.hist()
plt.xlabel("Survived 1")
plt.show()
survived_1 =dataset.Fare[dataset.Survived == 1]
survived_1.hist()
plt.xlabel("Survived 2")
plt.show()

print(dataset['Fare'].isnull().values.any())
plt.scatter(dataset['Survived'], dataset['Fare'])
plt.ylabel("Fare") 
plt.xlabel("Survived") 
plt.title("Fare Distribution")
plt.show()


# Class - Survival

# In[ ]:


print (dataset['Pclass'][0:10])
dataset['Pclass'].hist()  
plt.show()
print(dataset['Pclass'].isnull().values.any())

sns.countplot(dataset['Pclass'], hue=dataset['Survived'])
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

Embark location - Survival
# In[ ]:


print (dataset['Embarked'][0:10])
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


# ## Keep useful data features
# pclass, sex, age, fare, embarked

# ### 分离label 和 训练数据

# In[ ]:


label = dataset.loc[:,'Survived']
data = dataset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdat = testset.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(data.shape)
print(data)


# Fill in the NAN data

# In[ ]:


def fill_NAN(data):  
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy


data_no_nan = fill_NAN(data)
testdat_no_nan = fill_NAN(testdat)

print(testdat.isnull().values.any())    
print(testdat_no_nan.isnull().values.any())
print(data.isnull().values.any())    
print(data_no_nan)

# print(data)


# 处理Sex 

# In[ ]:


print(data_no_nan['Sex'].isnull().values.any())

# one hot encoding
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdat_after_sex = transfer_sex(testdat_no_nan)
print(testdat_after_sex)


# 处理Embarked

# In[ ]:


def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy

data_after_embarked = transfer_embark(data_after_sex)
testdat_after_embarked = transfer_embark(testdat_after_sex)
print(testdat_after_embarked)


# In[ ]:


print(data_after_embarked)


# In[ ]:


# Training dataset

data_now = data_after_embarked
testdat_now = testdat_after_embarked

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(data_now, label, random_state=0, train_size=0.8)


# In[ ]:


print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
k = 10
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(data_now, label)


# In[ ]:


# Test dataset

predictions = classifier.predict(test_data)
print(predictions)


# In[ ]:


# Check accuracy

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(accuracy_score(test_labels, predictions))
print(classification_report(test_labels, predictions))  
print(confusion_matrix(test_labels, predictions))


# In[ ]:


# cross validation 找到最好的k值
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

k_range = range(1,50)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, data_now, label, cv = 5, scoring='accuracy')
    print("k = " + str(k) + ", score = " + str(scores) + ", mean = " +  str(scores.mean()))
    k_scores.append(scores.mean())


# In[ ]:


plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Cross Validation Accuracy')
plt.show()


# In[ ]:


# Predict test dataset

clf = KNeighborsClassifier(n_neighbors=20)
clf.fit(data_now, label)
knn_result = clf.predict(testdat_now)


# In[ ]:


print(knn_result)


# Output csv file

# In[ ]:


df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": knn_result})
print(df.shape)

df.to_csv('titanic_submission.csv',header=True,index=False)
df.head()


# ### Use Decision Tree to Predict Testset

# In[ ]:


from sklearn import tree

# 深度为3，叶子节点数不超过5的决策树
dtree_clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
dtree_clf.fit(train_data, train_labels)


# In[ ]:


import graphviz

dot_data = tree.export_graphviz(dtree_clf, out_file=None, feature_names=train_data.columns) 

graph = graphviz.Source(dot_data) 
graph.render("titanic") 
#graph.view()
graph


# In[ ]:


from sklearn import metrics
def measure_performance(X, y, clf, show_accuracy=True, show_classification_report=True, show_confussion_matrix=True):
    y_pred = clf.predict(X)
    if show_accuracy:
        print("Accuracy:{0:.3f}".format(metrics.accuracy_score(y, y_pred)),"\n")
    
    if show_classification_report:
        print("Classification report")
        print(metrics.classification_report(y, y_pred), "\n")
    
    if show_confussion_matrix:
        print("Confusion matrix")
        print(metrics.confusion_matrix(y, y_pred), "\n")


# In[ ]:


measure_performance(test_data, test_labels, dtree_clf)


# In[ ]:


dtree_result = dtree_clf.predict(testdat_now)
print(dtree_result.shape)


# In[ ]:


df = pd.DataFrame({"PassengerId": testset['PassengerId'],"Survived": dtree_result})
print(df.shape)
df.to_csv('titanic_submission_dt.csv', index=False, header=True)
df.head()


# In[ ]:




