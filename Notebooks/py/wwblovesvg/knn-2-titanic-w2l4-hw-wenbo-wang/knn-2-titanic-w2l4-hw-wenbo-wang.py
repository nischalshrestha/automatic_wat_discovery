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


# function to read data
def load_data(data_dir):
    train = pd.read_csv(data_dir + "train.csv", header = 0, sep = ',')
    test = pd.read_csv(data_dir + "test.csv", header = 0, sep = ',')
    print(train.head())
    print(test.head())
    print(train.shape, test.shape)
    return train, test

data_dir = "../input/"
train, test = load_data(data_dir)


# In[ ]:


print(train.columns)
print(train.dtypes)
print(train.info())
print(train.describe())


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


print(train.head(20))


# In[ ]:


for col in train.columns:
    print("{} has any NaN? {}".format(col, train[col].isnull().values.any()))


# In[ ]:





# In[ ]:





# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 观察性别

# In[ ]:


print(train.Sex.value_counts())
survived_male = train.Survived[train.Sex == "male"].value_counts()
survived_female = train.Survived[train.Sex == 'female'].value_counts()
df = pd.DataFrame({"male": survived_male, "female": survived_female})
df = df.rename(index = {1:"Life", 0:"Death"}, inplace = False)
df.plot(kind = "bar", stacked = True)
plt.title("Survival by Sex")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()


# 看看年龄

# In[ ]:


train.Age.hist()
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

train.Age[train.Survived == 0].hist()
plt.title("Age distribution of people who did not survive")
plt.xlabel("Age distribution")
plt.ylabel("Count")
plt.show()

train.Age[train.Survived == 1].hist()
plt.title("Age distribution of people who survived")
plt.xlabel("Age distribution")
plt.ylabel("Count")
plt.show()


# 
# 看看船票价钱

# In[ ]:


train.Fare.hist()
plt.title("Fare distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

train.Fare[train.Survived == 0].hist()
plt.title("Fare distribution of people who did not survive")
plt.xlabel("Fare distribution")
plt.ylabel("Count")
plt.show()

train.Fare[train.Survived == 1].hist()
plt.title("Fare distribution of people who survived")
plt.xlabel("Fare distribution")
plt.ylabel("Count")
plt.show()


# 观察乘客舱层

# In[ ]:


train.Pclass.hist()
plt.title("Pclass distribution")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.show()

survived_p1 = train.Survived[train.Pclass == 1].value_counts()
survived_p2 = train.Survived[train.Pclass == 2].value_counts()
survived_p3 = train.Survived[train.Pclass == 3].value_counts()

df = pd.DataFrame({"P1": survived_p1, "P2": survived_p2, "P3": survived_p3})
df = df.rename(index = {1: "Life", 0: "Death"}, inplace = False)

df.plot(kind = "bar", stacked = True)
plt.title("Survival by Pclass")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()


# 观察登船地点

# In[ ]:


survived_q = train.Survived[train.Embarked == "Q"].value_counts()
survived_s = train.Survived[train.Embarked == "S"].value_counts()
survived_c = train.Survived[train.Embarked == "C"].value_counts()

df = pd.DataFrame({"Q": survived_q, "S": survived_s, "C": survived_c})
df = df.rename(index = {1: "Life", 0: "Death"}, inplace = False)

df.plot(kind = "bar", stacked = True)
plt.title("Survival by Embarked place")
plt.xlabel("Survival")
plt.ylabel("Count")
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# # 分离label 和 训练数据

# In[ ]:


labels = ['Pclass', 'Sex', "Age", "Fare", "Embarked"]
temp_x_train = train[labels]
temp_y_train = train.Survived
temp_x_test = test[labels]

print(temp_x_train.shape, temp_x_test.shape)


# 处理空数据

# In[ ]:


# first see the sex and embarked counts
print(train.Sex.value_counts())
print(train.Embarked.value_counts())


# In[ ]:


# define a function to deal with NaN
def fill_nan(data):
    copy = data.copy(deep = True)
    
    # get medians
    age_median = copy["Age"].median()
    pclass_median = copy["Pclass"].median()
    fare_median = copy["Fare"].median()
    sex_median = "male"
    embarked_median = "S"
    
    # fill with median
    copy["Age"] = copy["Age"].fillna(age_median)
    copy["Pclass"] = copy["Pclass"].fillna(pclass_median)
    copy["Fare"] = copy["Fare"].fillna(fare_median)
    copy["Sex"] = copy["Sex"].fillna(sex_median)
    copy["Embarked"] = copy["Embarked"].fillna(embarked_median)
    
    return copy

fillna_x_train, fillna_x_test = fill_nan(temp_x_train), fill_nan(temp_x_test)
print(fillna_x_train.head(20))
print(fillna_x_train.isnull().values.any(), fillna_x_test.isnull().values.any())


# 处理Sex 

# In[ ]:


# transfer sex to numerical values
def transfer_sex(data):
    copy = data.copy(deep = True)
    
    # turn male to 1, female to 0
    copy.loc[data.Sex == "male", "Sex"] = 1
    copy.loc[data.Sex == 'female', "Sex"] = 0
    
    return copy

sex_x_train, sex_x_test = transfer_sex(fillna_x_train), transfer_sex(fillna_x_test)
print(sex_x_train.head(10))
print(sex_x_test.head(10))


# 处理Embarked

# In[ ]:


# make a function to turn embarked alphabetical values into numerical values
def transfer_embarked(data):
    copy = data.copy(deep = True)
    
    copy.loc[copy.Embarked == "Q", "Embarked"] = 1
    copy.loc[copy.Embarked == "S", "Embarked"] = 2
    copy.loc[copy.Embarked == 'C', "Embarked"] = 3
    
    return copy

embarked_x_train, embarked_x_test = transfer_embarked(sex_x_train), transfer_embarked(sex_x_test)
print(embarked_x_train.shape, embarked_x_test.shape)
print(embarked_x_train.head(10))
print(embarked_x_test.head(10))


# 利用KNN训练数据

# In[ ]:


# generate original train and test data set
origin_x_train, origin_y_train, origin_x_test = embarked_x_train, temp_y_train, embarked_x_test
print(origin_x_train.shape, origin_y_train.shape, origin_x_test.shape)
print(origin_y_train[:10])


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_vali, y_train, y_vali = train_test_split(origin_x_train, origin_y_train, test_size = 0.2, random_state = 0)
print(x_train.shape, x_vali.shape, y_train.shape, y_vali.shape)


# In[ ]:


import time
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# 预测
k_range = range(1, 51)
scores = list()
for k in k_range:
    start = time.time()
    print("k = {} now starts...".format(k))
    
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_vali)
    
    accuracy = accuracy_score(y_vali, y_pred)
    scores.append(accuracy)
    print(confusion_matrix(y_vali, y_pred))
    print(classification_report(y_vali, y_pred))
    
    end = time.time()
    print("k = {} now ends, time spent = {}".format(k, end - start))


# In[ ]:


# 检测模型precision， recall 等各项指标
plt.title("Accuracy by k values")
plt.plot(k_range, scores)
plt.xlabel("K value")
plt.ylabel("Accuracy score")
plt.show()


# In[ ]:


sorted = np.array(scores).argsort()
best_accuracy = scores[sorted[-1]]
best_k = sorted[-1] + 1
print("best accuracy = {}, best k = {}".format(best_accuracy, best_k))


# In[ ]:


# 预测
knn = KNeighborsClassifier(n_neighbors = best_k)
knn.fit(origin_x_train, origin_y_train)
final_y_pred = knn.predict(origin_x_test)
print(final_y_pred)


# In[ ]:





# 打印输出

# In[ ]:


df = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final_y_pred})
print(df.head(20))
df.to_csv("submission.csv", header = True, index = False)

