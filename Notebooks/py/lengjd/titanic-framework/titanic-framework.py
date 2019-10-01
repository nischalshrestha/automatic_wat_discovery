#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#read in data
dataset = pd.read_csv('../input/train.csv')
testset = pd.read_csv('../input/test.csv')
#dataset = pd.read_csv('./all/train.csv')
#testset = pd.read_csv('./all/test.csv')


# In[ ]:


dataset.columns #观看所有的列


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

# In[ ]:


dataset.head()


# In[ ]:


print(dataset.dtypes)


# In[ ]:


print(dataset.describe()) #快速得到每一位feature的分布


# # 观察数据

# 观察性别是否影响生存率

# In[ ]:


Survived_m = dataset.Survived[dataset.Sex == 'male'].value_counts()
Survived_f = dataset.Survived[dataset.Sex == 'female'].value_counts()

df = pd.DataFrame({'male': Survived_m, 'female': Survived_f})
df.plot(kind = 'bar', stacked=True) #画柱状图
plt.title('Survival by Sex')
plt.xlabel('Survival')
plt.ylabel('count')
plt.show()


# 观察年龄是否影响生存率

# In[ ]:


dataset['Age'].hist() # histogram 分布图
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution')
plt.show()

dataset[dataset.Survived==0]['Age'].hist() 
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Age'].hist() 
plt.ylabel('Number')
plt.xlabel('Age')
plt.title('Age distribution of people who survived')
plt.show()


# 观察票价是否影响生存率

# In[ ]:


dataset['Fare'].hist() # histogram 分布图
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare distribution')
plt.show()

dataset[dataset.Survived==0]['Fare'].hist() 
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Fare'].hist() 
plt.ylabel('Number')
plt.xlabel('Fare')
plt.title('Fare distribution of people who survived')
plt.show()


# In[ ]:


dataset['Fare'].hist(density=True) # histogram 分布图用比率
plt.ylabel('Ratio')
plt.xlabel('Fare')
plt.title('Fare distribution')
plt.show()

dataset[dataset.Survived==0]['Fare'].hist(density=True) 
plt.ylabel('Ratio')
plt.xlabel('Fare')
plt.title('Fare distribution of people who did not survive')
plt.show()

dataset[dataset.Survived==1]['Fare'].hist(density=True) 
plt.ylabel('Ratio')
plt.xlabel('Fare')
plt.title('Fare distribution of people who survived')
plt.show()


# 观察乘客舱层是否影响生存率

# In[ ]:


dataset['Pclass'].hist()
plt.show()
Survived_p1 = dataset.Survived[dataset['Pclass']==1].value_counts()
Survived_p2 = dataset.Survived[dataset['Pclass']==2].value_counts()
Survived_p3 = dataset.Survived[dataset['Pclass']==3].value_counts()

df = pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("Survival by Pclass")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()


# 观察上船地点是否影响生存率

# In[ ]:


Survived_S = dataset.Survived[dataset['Embarked'] == 'S'].value_counts()
Survived_C = dataset.Survived[dataset['Embarked'] == 'C'].value_counts()
Survived_Q = dataset.Survived[dataset['Embarked'] == 'Q'].value_counts()
df = pd.DataFrame({'S':Survived_S, 'C': Survived_C, 'Q': Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("Suvived by Embarked")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# In[ ]:


label = dataset['Survived']
data = dataset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
testdata = testset[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print(data.shape)
print(data)


# In[ ]:


data['Sex'].value_counts()


# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


def fill_NAN(data):
    data_copy = data.copy(deep=True) # deep copy
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median()) #fillna--填充空值 
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Sex'] = data_copy['Sex'].fillna('male') # 男性多
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')# 从S上船的多
    return data_copy


# In[ ]:


data_no_nan = fill_NAN(data)
testdata_no_nan = fill_NAN(testdata)


# In[ ]:


print(data_no_nan)


# In[ ]:


# Change String to Values
# Sex
def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0 # loc is to access a group of rows
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1 
    return data_copy

data_after_sex = transfer_sex(data_no_nan)
testdata_after_sex = transfer_sex(testdata_no_nan)
print(testdata_after_sex)


# In[ ]:


# Embarked
def transfer_embarked(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0 
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1 
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2 
    return data_copy

data_after_embarked = transfer_embarked(data_after_sex)
testdata_after_embarked = transfer_embarked(testdata_after_sex)
print(testdata_after_embarked)


# In[ ]:


#training data
data_now = data_after_embarked
testdata_now = testdata_after_embarked
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data_now, label, random_state=0, test_size=0.2)


# In[ ]:


print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_scores = []
k_range = range(1, 51)
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_data, train_labels) # 保存数据
    print('K=', K)
    predictions = clf.predict(test_data)
    score = accuracy_score(test_labels, predictions)
    print(score)
    k_scores.append(score)


# In[ ]:


plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores).argsort())
print(np.array(k_range)[32]) # k_range是从1开始的


# In[ ]:


# predict on test data
clf = KNeighborsClassifier(n_neighbors = 33)
clf.fit(data_now, label)
result = clf.predict(testdata_now) # 用现有的数据预测测试数据


# In[ ]:


result


# In[ ]:


df = pd.DataFrame({'PassengerId': testset['PassengerId'], 'Survived': result})
df.to_csv('submission.csv', header=True)


# In[ ]:





# In[ ]:




