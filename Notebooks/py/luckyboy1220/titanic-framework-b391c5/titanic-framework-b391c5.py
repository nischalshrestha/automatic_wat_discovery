#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 读入数据

# In[ ]:


data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


data.columns


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

# # 查看读入数据

# In[ ]:


data.info()


# In[ ]:





# In[ ]:





# In[ ]:





# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


data['Age'].describe()


# 看看年龄

# In[ ]:


data['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution')
plt.show() 

data[data.Survived==0]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who did not survive')
plt.show()

data[data.Survived==1]['Age'].hist()  
plt.ylabel("Number") 
plt.xlabel("Age") 
plt.title('Age distribution of people who survived')
plt.show()

#Found out that age and survive are dependent


# 看看船票价钱

# In[ ]:


data['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution')
plt.show() 

data[data.Survived==0]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution of people who did not survive')
plt.show()

data[data.Survived==1]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare distribution of people who survived')
plt.show()


# 观察乘客舱层

# In[ ]:


data['Pclass'].hist()  
plt.show()  
print(data['Pclass'].isnull().values.any())

Survived_p1 = data.Survived[data['Pclass'] == 1].value_counts()
Survived_p2 = data.Survived[data['Pclass'] == 2].value_counts()
Survived_p3 = data.Survived[data['Pclass'] == 3].value_counts()

df=pd.DataFrame({'p1':Survived_p1, 'p2':Survived_p2, 'p3':Survived_p3})
print(df)
df.plot(kind='bar', stacked=True)
plt.title("survived by pclass")
plt.xlabel("pclass") 
plt.ylabel("count")
plt.show()


# 观察登船地点

# In[ ]:


Survived_S = data.Survived[data['Embarked'] == 'S'].value_counts()
Survived_C = data.Survived[data['Embarked'] == 'C'].value_counts()
Survived_Q = data.Survived[data['Embarked'] == 'Q'].value_counts()

print(Survived_S)
df = pd.DataFrame({'S':Survived_S, 'C':Survived_C, 'Q':Survived_Q})
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("Embarked") 
plt.ylabel("count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


train_label=data.loc[:,'Survived']
train_feature=data.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
test_feature=test.loc[:,['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]

print(train_feature.shape)
print(train_feature)


# 处理空数据

# In[ ]:


def fill_NAN(data):  
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy


train_feature_no_nan = fill_NAN(train_feature)
test_feature_no_nan = fill_NAN(test_feature)

print(test_feature.isnull().values.any())    
print(test_feature_no_nan.isnull().values.any())
print(train


# In[ ]:


def fill_NAN(data):  
    data_copy = data.copy(deep=True)
    data_copy.loc[:,'Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy.loc[:,'Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy.loc[:,'Pclass'] = data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy.loc[:,'Sex'] = data_copy['Sex'].fillna('female')
    data_copy.loc[:,'Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy


train_feature_no_nan = fill_NAN(train_feature)
test_feature_no_nan = fill_NAN(test_feature)

print(test_feature.isnull().values.any())    
print(test_feature_no_nan.isnull().values.any())
print(train_feature.isnull().values.any())   
print(train_feature_no_nan.isnull().values.any())    

print(train_feature_no_nan)


# 处理Sex 

# In[ ]:



def transfer_sex(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male', 'Sex'] = 1
    return data_copy

train_feature_no_nan_no_string = transfer_sex(train_feature_no_nan)
test_feature_no_nan_no_string = transfer_sex(test_feature_no_nan)
print(test_feature_no_nan_no_string)
    


# 处理Embarked

# In[ ]:


def transfer_embark(data):
    data_copy = data.copy(deep=True)
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 2
    return data_copy


train_feature_no_nan_no_string = transfer_embark(train_feature_no_nan_no_string)
test_feature_no_nan_no_string = transfer_embark(test_feature_no_nan_no_string)
print(test_feature_no_nan_no_string)


# 利用KNN训练数据

# In[ ]:


train_feature_knn = train_feature_no_nan_no_string
test_feature_knn = test_feature_no_nan_no_string
from sklearn.model_selection import train_test_split


train_feature,vali_feature, train_labels, vali_labels=train_test_split(train_feature_knn,train_label,random_state=0,test_size=0.2)


# In[ ]:


print(train_feature.shape, vali_feature.shape, train_labels.shape, vali_labels.shape)


# In[ ]:


# 预测
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
k_range = range(1, 51)
k_scores = []
k_precision = []
k_recall = []
for K in k_range:
    clf = KNeighborsClassifier(n_neighbors = K)
    clf.fit(train_feature,train_labels)
    print('K=', K)
    predictions=clf.predict(vali_feature)
    score = accuracy_score(vali_labels,predictions)
    precision = precision_score(vali_labels, predictions)
    print("accuracy is " + str(score))
    print("precision is " + str(precision))
    k_scores.append(score)


# In[ ]:


# 检测模型precision， recall 等各项指标


# In[ ]:





# In[ ]:


# 预测
plt.plot(k_range, k_scores)
plt.xlabel('K for KNN')
plt.ylabel('Accuracy on validation set')
plt.show()
print(np.array(k_scores))
print(np.array(k_scores).argsort())


# In[ ]:


# predict test data on optimum k = 33
clf=KNeighborsClassifier(n_neighbors=33)
clf.fit(train_feature,train_labels)
result=clf.predict(test_feature_no_nan_no_string)
print(result)


# 打印输出

# In[ ]:


df = pd.DataFrame({"PassengerId": test['PassengerId'],"Survived": result})
df.to_csv('submission.csv',header=True, index=False)


# In[ ]:




