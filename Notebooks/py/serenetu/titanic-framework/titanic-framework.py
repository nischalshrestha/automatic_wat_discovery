#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# 读入数据

# In[ ]:


data_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')


# In[ ]:


print(data_set.columns)
print(test_set.columns)


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


print(data_set.head())


# In[ ]:


print(type(data_set))
print(data_set.dtypes)
print(data_set.info())


# In[ ]:


print(data_set.describe())


# In[ ]:


m_survived = data_set[data_set.Sex == 'male'].Survived.value_counts()
print(type(m_survived))
print('man survived')
print(m_survived)
f_survived = data_set[data_set.Sex == 'female'].Survived.value_counts()
print('woman survived')
print(f_survived)

survived_df = pd.DataFrame({'male': m_survived, 'female': f_survived})
print(survived_df)
survived_df.plot(kind = 'bar', stacked=True)
plt.title('survived by sex')
plt.xlabel('sex')
plt.ylabel('count')
plt.show()


# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄

# In[ ]:


print(data_set['Age'].head())
print(data_set['Age'].describe())


# 看看年龄

# In[ ]:


data_set['Age'].hist()
plt.title('Age Dist: All')
plt.show()

survived_age = data_set[data_set.Survived == 1].Age
survived_age.hist()
plt.title('Age Dist: Survived')
plt.show()

nosurvived_age = data_set[data_set.Survived == 0].Age
nosurvived_age.hist()
plt.title('Age Dist: Not Survived')
plt.show()

# need to look
age_disc = pd.cut(data_set.Age, np.linspace(0, 80, 9), duplicates='drop').value_counts()
print (age_disc)
age_sur_disc = pd.cut(data_set[data_set['Survived'] == 1].Age, np.linspace(0, 80, 9), duplicates='drop').value_counts()
print(age_sur_disc)
age_sur_persent = age_sur_disc / age_disc
age_sur_persent.plot(kind = 'bar')
plt.title('Survived Persentage in Age Range')
plt.show()


# 看看船票价钱

# In[ ]:


data_set['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare Dist: All')
plt.show() 

data_set[data_set.Survived==0]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare Dist: Not Survive')
plt.show()

data_set[data_set.Survived==1]['Fare'].hist()  
plt.ylabel("Number") 
plt.xlabel("Fare") 
plt.title('Fare Dist: Survived')
plt.show()


# 观察乘客舱层

# In[ ]:


print(data_set['Pclass'].head())
data_set['Pclass'].hist(bins = 3)
plt.title('Pclass All: Dist')
plt.show()

data_set['Pclass'].isnull().values.any()

p1_survived = data_set[data_set.Pclass == 1].Survived.value_counts()
p2_survived = data_set[data_set.Pclass == 2].Survived.value_counts()
p3_survived = data_set[data_set.Pclass == 3].Survived.value_counts()

p_survived = pd.DataFrame({'p1': p1_survived, 'p2': p2_survived, 'p3': p3_survived})
print(p_survived)
p_survived.plot(kind = 'bar', stacked = True)


# 观察登船地点

# In[ ]:


embarked_c = data_set[data_set.Embarked == 'C'].Survived.value_counts()
embarked_q = data_set[data_set.Embarked == 'Q'].Survived.value_counts()
embarked_s = data_set[data_set.Embarked == 'S'].Survived.value_counts()

embarked = pd.DataFrame({'C': embarked_c, 'Q': embarked_q, 'S': embarked_s})
print(embarked)
embarked.plot(kind = 'bar', stacked = True)


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


label = data_set['Survived']
feature = data_set[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print(label.head())
print(feature.head())

test = test_set[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print(test.head())


# 处理空数据

# In[ ]:


def fill_nan(data_set):
    data_no_nan = data_set.copy(deep = True)
    data_no_nan['Pclass'] = data_set['Pclass'].fillna(data_set['Pclass'].median())
    data_no_nan['Age'] = data_set['Age'].fillna(data_set['Age'].median())
    data_no_nan['Fare'] = data_set['Fare'].fillna(data_set['Fare'].median())
    data_no_nan['Sex'] = data_set['Sex'].fillna('female')
    data_no_nan['Embarked'] = data_set['Embarked'].fillna('S')
    return data_no_nan

def fill_series(column):
    column_copy = column.copy(deep = True)
    all_values = column_copy[column_copy.isnull() == False].values
    len_nan = len(column_copy[column_copy.isnull()])
    rand_values = np.random.choice(all_values, len_nan)
    column_copy.loc[column_copy.isnull()] = rand_values
    return column_copy

def fill_nan_rand(data_set):
    data_no_nan = data_set.copy(deep = True)
    data_no_nan.loc[:, 'Pclass'] = fill_series(data_no_nan['Pclass'])
    data_no_nan.loc[:, 'Age'] = fill_series(data_no_nan['Age'])
    data_no_nan.loc[:, 'Fare'] = fill_series(data_no_nan['Fare'])
    data_no_nan.loc[:, 'Sex'] = fill_series(data_no_nan['Sex'])
    data_no_nan.loc[:, 'Embarked'] = fill_series(data_no_nan['Embarked'])
    return data_no_nan

feature_no_nan = fill_nan_rand(feature)
print(feature.head())
print(feature.isnull().any())
test_no_nan = fill_nan_rand(test)


# 处理Sex 

# In[ ]:


def transfer_sex(data_set):
    data_copy = data_set.copy(deep = True)
    data_copy.loc[data_copy['Sex'] == 'female', 'Sex'] = 0
    data_copy.loc[data_copy['Sex'] == 'male' ,'Sex'] = 1
    return data_copy

feature = transfer_sex(feature_no_nan)
test = transfer_sex(test_no_nan)


# 处理Embarked

# In[ ]:


def transfer_embarked(data_set):
    data_copy = data_set.copy(deep = True)
    data_copy.loc[data_copy['Embarked'] == 'C', 'Embarked'] = 0
    data_copy.loc[data_copy['Embarked'] == 'Q', 'Embarked'] = 1
    data_copy.loc[data_copy['Embarked'] == 'S', 'Embarked'] = 2
    return data_copy

feature = transfer_embarked(feature)
test= transfer_embarked(test)
    


# 利用KNN训练数据

# In[ ]:


from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, random_state = 0, test_size = 0.2)
print(feature_train.shape, feature_test.shape, label_train.shape, label_test.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

k_range = range(1, 51)
score = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(feature_train, label_train)
    label_predict = knn.predict(feature_test)
    score.append(accuracy_score(label_test, label_predict))


# In[ ]:


plt.plot(k_range, score)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
k = np.array(score).argsort()[-1] + 1
print('Max Score:', score[k - 1])
print('Better K:', k)


# In[ ]:


# 预测
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(feature, label)
test_predict = knn.predict(test)


# In[ ]:


# 检测模型precision， recall 等各项指标
from sklearn.metrics import classification_report, confusion_matrix
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(feature_train, label_train)
label_predict = knn.predict(feature_test)
print(classification_report(label_test, label_predict))
print(confusion_matrix(label_test, label_predict))


# In[ ]:


# cross validation 找到最好的k值
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

scores = []
k_range = range(1, 51)
for k in k_range:
    print('K =', k)
    knn = KNeighborsClassifier(n_neighbors = k)
    onescore = cross_val_score(knn, feature, label, cv = 10, scoring='accuracy')
    scores.append(np.array(onescore).mean())
    print('Scores:', scores[-1])
plt.plot(k_range, scores)
k = np.array(scores).argsort()[-1] + 1
print('Best K from Cross Validation:', k, 'Accuracy:', scores[k-1])
    


# In[ ]:





# In[ ]:


# 预测
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(feature, label)
test_predict = knn.predict(test)


# In[ ]:





# 打印输出

# In[ ]:


res = pd.DataFrame({'PassengerId': test_set['PassengerId'], 'Survived': test_predict})
res.to_csv('submission.csv', header = True)

