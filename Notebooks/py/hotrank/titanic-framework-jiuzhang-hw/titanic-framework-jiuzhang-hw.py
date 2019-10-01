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


data_dir = '../input/'

train_raw = pd.read_csv(data_dir + 'train.csv')
test_raw = pd.read_csv(data_dir + 'test.csv')


# In[ ]:


train_raw.columns


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


train_raw.dtypes


# In[ ]:


print(train_raw.shape, test_raw.shape)
train_raw.head(20)


# In[ ]:


train_raw.describe()


# In[ ]:





# In[ ]:





# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察年龄
# 
# - hist() is a method of pandas framework/series

# In[ ]:


train_raw.Age.hist()
plt.title('Age histogram - All passengers')
plt.xlabel('Age')
plt.ylabel('number of passengers')
plt.show()

train_raw[train_raw.Survived == 1].Age.hist()
plt.title('Age histogram - survived passengers')
plt.xlabel('Age')
plt.ylabel('number of passengers')
plt.show()

train_raw[train_raw.Survived == 0].Age.hist()
plt.title('Age histogram - died passengers')
plt.xlabel('Age')
plt.ylabel('number of passengers')
plt.show()


# 观察性别

# In[ ]:


survived_m = train_raw[train_raw.Sex == 'male'].Survived.value_counts()
survived_f = train_raw[train_raw.Sex == 'female'].Survived.value_counts()

df = pd.DataFrame({'male': survived_m, 'female': survived_f})
df.head()
df.plot(kind = 'bar', stacked = True)


# 看看船票价钱

# In[ ]:


train_raw.Fare.hist(density = True)
plt.title('fare histogram - all passengers')
plt.show()

train_raw[train_raw.Survived == 1].Fare.hist(density = True)
plt.title('fare histogram - survived passengers')
plt.show()

train_raw[train_raw.Survived == 0].Fare.hist(density = True)
plt.title('fare histogram - died passengers')
plt.show()


# 观察乘客舱层

# In[ ]:


train_raw.Pclass.hist(density = True)
plt.title('Pclass histogram - all passengers')
plt.show()

train_raw[train_raw.Survived == 1].Pclass.hist(density = True)
plt.title('Pclass histogram - survived passengers')
plt.show()

train_raw[train_raw.Survived == 0].Pclass.hist(density = True)
plt.title('Pclass histogram - died passengers')
plt.show()


# 观察登船地点

# In[ ]:


set(train_raw.Embarked.values)  # check possible values

survived_S = train_raw[train_raw.Embarked == 'S'].Survived.value_counts()
survived_C = train_raw[train_raw.Embarked == 'C'].Survived.value_counts()
survived_Q = train_raw[train_raw.Embarked == 'Q'].Survived.value_counts()

df = pd.DataFrame({'S': survived_S, 'C': survived_C, 'Q': survived_Q })
df.plot(kind = 'bar', stacked = True)
plt.title('survival by embarked location - bar chart')
plt.show()


# # 保留下有效数据并分离label和features
# pclass, sex, age, fare, embarked

# In[ ]:


y_train_full = train_raw.Survived
X_raw = train_raw[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
X_test_raw = test_raw[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
print(y_train_full.shape, X_raw.shape, X_test_raw.shape)


# ## 处理空数据
# - sex, embarked need to be convereted to numbers
# - all features should be checked for none value and replace by median (continious variable) or most probable value (discrete variable)

# ### check Pclass - no missing values

# In[ ]:


print(X_raw.Pclass.isnull().values.any())
print(X_test_raw.Pclass.isnull().values.any())


# ### check Fare -  test case has missing values

# In[ ]:


print(X_raw.Fare.isnull().values.any())
print(X_test_raw.Fare.isnull().values.any())



# ### check Sex - no missing values

# In[ ]:


print(X_raw.Sex.isnull().values.any())
print(X_test_raw.Sex.isnull().values.any())

# X_raw.Sex[X_raw.Sex == 'male'] = 1     
# X_raw.Sex[X_raw.Sex == 'female'] = 0
# pandas gives SettingWithCopy warning for the above two lines due to editing with chained indexing. (X_raw is indexed from
# train_raw) This means it does not know whether this will edit the original DataFrame train_row
# see http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy


# ### check Embarked - training data has missing values

# In[ ]:


print(X_raw.Embarked.isnull().values.any())
print(X_test_raw.Embarked.isnull().values.any())


# ### check Age - both training and testing data has missing values

# In[ ]:


print(X_raw.Age.isnull().values.any())
print(X_test_raw.Age.isnull().values.any())


# ### fill missing data： Fare, Embarked, Age 

# In[ ]:


def fill_NAN(data):
    data_copy = data.copy(deep = True)
    data_copy['Age'] = data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Fare'] = data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Embarked'] = data_copy['Embarked'].fillna('S')
    return data_copy

X_train_noNAN = fill_NAN(X_raw)
X_test_noNAN = fill_NAN(X_test_raw)

print(X_train_noNAN.isnull().values.any())
print(X_test_noNAN.isnull().values.any())


# ### convert text features to number features: Sex, Embarked

# In[ ]:


# when editing data, make deep copy

def text_feature_2number(dataframe, column_name, lut = None):
    '''Given a dataframe and the name (string) of a column, return a copy of the dataframe with the text values in the specifie
    column converted to numbers, also return the look up table as a dictionary'''
    data_copy = dataframe.copy(deep = True)
    if lut is None:
        values = set(data_copy[column_name])
        numbers = range(len(values))
        lut = dict(zip(numbers, values))
    
    for key, value in lut.items():
        data_copy.loc[data_copy[column_name] == value, column_name] = key
    
    return data_copy, lut

X_train_full, sex_dict = text_feature_2number(X_train_noNAN, 'Sex')  
X_train_full, embarked_dict = text_feature_2number(X_train_full, 'Embarked') 
X_test, _ = text_feature_2number(X_test_noNAN, 'Sex', sex_dict)
X_test, _ = text_feature_2number(X_test, 'Embarked', embarked_dict)

print(X_train_full.head())
print(X_test_noNAN.head())
print(X_test.head())
print(sex_dict, embarked_dict)


# ## Split the training data for training and validation

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_vali, y_train, y_vali = train_test_split(X_train_full, y_train_full, test_size = 0.2, random_state = 0)

print(X_train.shape, X_vali.shape, y_train.shape, y_vali.shape)


# 利用KNN训练数据

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score

k_range = range(1, 51)
k_accuracy, k_precision, k_f1 = [], [], []

for k in k_range:
    knn = KNeighborsClassifier(k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_vali)
    accuracy = accuracy_score(y_vali, predictions)
    precision = precision_score(y_vali, predictions)
    f1 = f1_score(y_vali, predictions)
    # print('k = ', k, '\t', 'score = ', score)
    k_accuracy.append(accuracy)
    k_precision.append(precision)
    k_f1.append(f1)
    
plt.plot(k_range, k_accuracy)
plt.title('Accuracy Score')
plt.xlabel('k')
plt.show()

plt.plot(k_range, k_precision)
plt.title('Precisoin')
plt.xlabel('k')
plt.show()

plt.plot(k_range, k_f1)
plt.title('f1 Score')
plt.xlabel('k')
plt.show()
    


# In[ ]:


# cross validation 找到最好的k值
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

k_cv_score = []

for k in k_range:
    knn = KNeighborsClassifier(k)
    score = cross_val_score(knn, X_train_full, y_train_full, cv = 5, scoring = 'accuracy' ).mean()
    k_cv_score.append(score)
    
plt.plot(k_range, k_cv_score)
plt.title('f1_score after cross validation')
plt.xlabel('k')
plt.show()

k_ans = k_range[np.array(k_cv_score).argsort()[-1]+1]
print('the best "k" is', k_ans)
    


# In[ ]:


# 预测
knn = KNeighborsClassifier(k_ans)
knn.fit(X_train_full, y_train_full)
predictions = knn.predict(X_test)
predictions.shape


# 打印输出

# In[ ]:


df = pd.DataFrame({'PassengerId': test_raw['PassengerId'], 'Survived': predictions})
df.to_csv('submission.csv', header = True)

