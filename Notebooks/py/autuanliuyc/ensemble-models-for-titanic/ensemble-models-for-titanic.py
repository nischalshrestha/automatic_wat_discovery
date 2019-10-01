#!/usr/bin/env python
# coding: utf-8

# [Titanic: Machine Learning from Disaster | Kaggle](https://www.kaggle.com/c/titanic/data)

# This script is using for data pre-processing, please to [Github](https://github.com/AutuanLiu/Kaggle-Compettions/tree/master/Titanic) see the code.

# In[ ]:


get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'reload_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[ ]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import re
from collections import Counter
sns.set_style('whitegrid')
sns.set_palette('Set1')


# ## 1 导入数据

# In[ ]:


datadir = '../input/'


# In[ ]:


# read data from file
train_data = pd.read_csv(f'{datadir}train.csv')
test_data = pd.read_csv(f'{datadir}test.csv')


# ## 2 数据基本信息

# In[ ]:


train_data.head()
test_data.head()


# In[ ]:


train_data.describe()
test_data.describe()


# In[ ]:


train_data.info()
test_data.info()


# ### 2.1 异常值检测

# In[ ]:


def detect_outliers(Q1, Q3, df, col):
    outlier_indices = []
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
    outlier_indices.extend(outlier_list_col)     
    return outlier_indices


# In[ ]:


train_data.Age.describe()
train_data.Fare.describe()
train_data.SibSp.describe()
train_data.Parch.describe()


# In[ ]:


train_data.Fare.plot(kind='kde')


# In[ ]:


train_data.Age.plot(kind='kde')


# In[ ]:


out1 = detect_outliers(20.125, 38, train_data, 'Age')
out2 = detect_outliers(7.9, 31, train_data, 'Fare')


# In[ ]:


np.array(out1)
np.array(out2)


# Fare分布不均衡，去掉一部分过大的Fare，怀疑其可能是离群值

# In[ ]:


train_data.loc[out2]


# 有多个 512值，暂时不认为其实离群值

# In[ ]:


# 备份数据
# 因为要做一些统一处理，所以将训练集测试集拼接起来
train = train_data
test = test_data
all_data = pd.concat([train, test], axis=0, sort=False)


# In[ ]:


train.shape, test.shape, all_data.shape


# In[ ]:


# one-hot编码
def dummies(col,data):
    data_dum = pd.get_dummies(data[col])
    data = pd.concat([data, data_dum], axis=1)
    data.drop(col, axis=1, inplace=True)
    return data


# ## 3 分类变量分析

# In[ ]:


# 获取所有的列名
all_data.columns.tolist()


# 数据总共有12个变量，其中PassengerId、Survived两个变量分别是样本编号和预测目标，这里暂时不考虑。由于test数据集在应用最终模型的过程之前都是未知的，所以，这里主要分析train，之后将相同的预处理操作应用到test，以保证模型的泛化能力。

# ### 3.1 Pclass

# In[ ]:


train.Pclass.value_counts()


# Pclass一共有三种数值，其中Pclass3的人数最多

# In[ ]:


sns.countplot(y='Pclass', hue='Survived', data=all_data)


# 上图可以看出：Pclass=1的乘客中，存活的数量明显大于死亡的人数，而Pclass=2的乘客生存和死亡的数量大致是相当的，Pclass=3的乘客死亡的人数占多数；这充分说明Pclass对最终的预测有很大的影响，是一个重要的特征。

# In[ ]:


any(all_data.Pclass.isna())


# In[ ]:


all_data.Pclass.head()


# In[ ]:


all_data = dummies('Pclass', all_data)


# In[ ]:


all_data.head()


# In[ ]:


all_data.rename(columns={1: 'Pclass1', 2: 'Pclass2', 3: 'Pclass3'}, inplace=True)


# In[ ]:


all_data.head()


# In[ ]:


backup1 = all_data
train = all_data[:891]
test = all_data[-418:]


# ### 3.2 Name

# In[ ]:


# 恢复数据
all_data = backup1


# In[ ]:


any(all_data.Name.isna())


# In[ ]:


all_data.Name.head(10)


# 名字中最有用的信息就是称呼，例如Mrs. 鸟事已经结婚的太太, Miss 表示未结婚的年轻女子，Master表示研究生等，社会地位的高低可能会影响最终的生存情况。首先，将称呼提取出来，然后观察是否与生存有关。

# In[ ]:


title = all_data.Name.map(lambda x: re.compile(",(.*?)\.").findall(x)[0].strip())


# In[ ]:


title.unique()


# In[ ]:


# 将 title 添加到最后
all_data['Title'] = title
all_data.drop('Name', axis=1, inplace=True)


# In[ ]:


all_data.head()


# In[ ]:


backup2 = all_data
train = all_data[:891]
test = all_data[-418:]


# In[ ]:


sns.countplot(y='Title', hue='Survived', data=all_data)


# In[ ]:


all_data.Title.value_counts()


# 这里有法语有英语，我们统一为英语并且合并数量较少的类别:
# * Mlle, Lady, Jonkheer, Dona --> Other
# * Mme --> Other
# * Don, Sir, Rev, Col --> Other
# * Ms --> Other
# * Major, Master, Dr, the Countess, Capt --> High

# In[ ]:


all_data.Title.replace({'Mlle': 'Other', 'Lady': 'Other', 'Dona': 'Other', 'Jonkheer': 'Other', 'Mme': 'Other', 
                        'Don': 'Other', 'Sir': 'Other', 'Rev': 'Other', 'Col': 'Other', 'Major': 'High', 'Ms': 'Other',
                        'Master': 'High', 'Dr': 'High', 'the Countess': 'High', 'Capt': 'High'}, inplace=True)


# In[ ]:


all_data.Title.unique()


# In[ ]:


all_data.Title.value_counts()


# In[ ]:


sns.countplot(y='Title', hue='Survived', data=all_data)


# 最终只留下四个类别，这些特征可能与性别存在冲突

# In[ ]:


all_data = dummies('Title', all_data)


# In[ ]:


backup2 = all_data


# ### 3.3 Sex

# In[ ]:


all_data = backup2


# In[ ]:


any(all_data.Sex.isna())


# In[ ]:


all_data.Sex.value_counts()


# In[ ]:


sns.countplot(y='Sex', hue='Survived', data=all_data)


# In[ ]:


all_data = dummies('Sex', all_data)


# In[ ]:


all_data.head()


# In[ ]:


backup3 = all_data


# ### 3.4 Age

# 年龄因素必然是会对生存情况存在重要影响的，老人小孩一定会有更大的生存机会，而青年人相对较小的生存几率。

# In[ ]:


all_data = backup3


# In[ ]:


any(all_data.Age.isna()) # 存在空值


# In[ ]:


num_nona = all_data.Age.count() # 非空数值个数
num_nona


# In[ ]:


num_na = all_data.Age.shape[0] - num_nona  # 空值个数
num_na


# In[ ]:


# 去掉空值行
nona_data = all_data.Age.dropna(axis=0, how='all', inplace=False)


# In[ ]:


# 未缺失的年龄的大致分布
nona_data.plot(kind='kde')


# In[ ]:


nona_data.describe()


# 插值方式：
# * Mr, Miss --> 21
# * Mrs --> 28
# * High --> 39

# In[ ]:


data = all_data.values
data.shape
len = data.shape[0]


# In[ ]:


for idx in range(len):
    tmp = data[idx, 2]
    if np.isnan(tmp):
        if data[idx, 12] == 1:
            data[idx, 2] = 39.
#         elif data[idx, 13] == 1 or data[idx, 14] == 1:
#             data[idx, 2] = 28.
        else:
            data[idx, 2] = 28.
    else:
        continue


# In[ ]:


all_data['Age'] = data[:, 2]


# In[ ]:


all_data.head()


# In[ ]:


all_data.shape


# In[ ]:


any(all_data.Age.isna())


# In[ ]:


# 整体年龄的大致分布
all_data.Age.plot(kind='kde')
# 未缺失的年龄的大致分布
nona_data.plot(kind='kde')


# ### 3.5 Ticket

# * Ticket 没有什么用处所以删除

# In[ ]:


all_data.drop(columns=['Ticket'], axis=1, inplace=True)


# In[ ]:


all_data.head()


# In[ ]:


all_data.shape


# ### 3.6 Cabin

# In[ ]:


nona = all_data.Cabin.count()
nona


# In[ ]:


num_na =all_data.shape[0] - nona
num_na


# 缺失值太多，所以选择删除该列

# In[ ]:


all_data.drop(columns=['Cabin'], axis=1, inplace=True)


# In[ ]:


all_data.head()


# In[ ]:


all_data.shape


# ### 3.7 Embarked

# In[ ]:


any(all_data.Embarked.isna())


# In[ ]:


nona = all_data.Embarked.count()
nona


# In[ ]:


num_na =all_data.shape[0] - nona
num_na


# In[ ]:


all_data.Embarked.describe()


# * 使用众数进行插值
# * one-hot编码

# In[ ]:


all_data.Embarked.fillna('S', inplace=True)
all_data.head()


# In[ ]:


any(all_data.Embarked.isna())


# In[ ]:


all_data = dummies('Embarked', all_data)


# In[ ]:


all_data.head()


# ### 3.8 SibSp and Parch

# * SibSp: 有误兄弟姐妹和配偶在船上
# * Parch： 有无父母在船上

# In[ ]:


any(all_data.SibSp.isna())
any(all_data.Parch.isna())


# * 不存在空值
# * 兄弟姐妹、配偶、父母都是家庭成员的一部分
# * 如果船上有任意的家庭成员，那么其本人一定会优先考虑妻子老人或者小孩
# * 使用二者的和作为新的特征

# In[ ]:


all_data['Family'] = all_data['SibSp'] + all_data['Parch']
all_data.head()


# In[ ]:


all_data.drop(columns=['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


all_data.head(10)


# ### 3.9 Fare

# In[ ]:


any(all_data.Fare.isna())


# In[ ]:


nona = all_data.Fare.count()
nona
num_na =all_data.shape[0] - nona
num_na


# In[ ]:


all_data.Fare.describe()


# * 中值填充缺失值

# In[ ]:


all_data.Fare.fillna(15, inplace=True)


# In[ ]:


any(all_data.Fare.isna())


# ### 3.10 数据类型转换

# In[ ]:


all_data.head()


# In[ ]:


all_data.PassengerId.dtype
all_data.Survived.dtype
all_data.Age.dtype
all_data.Fare.dtype
all_data.Pclass1.dtype
all_data.Pclass2.dtype
all_data.Pclass3.dtype
all_data.High.dtype
all_data.Miss.dtype
all_data.Mr.dtype
all_data.Mrs.dtype
all_data.female.dtype
all_data.male.dtype
all_data.C.dtype
all_data.Q.dtype
all_data.S.dtype
all_data.Family.dtype


# In[ ]:


all_data.head()


# In[ ]:


all_data.Age = all_data.Age.apply(pd.to_numeric).astype('float32')


# In[ ]:


all_data.head()


# In[ ]:


all_data.PassengerId.dtype
all_data.Survived.dtype
all_data.Age.dtype
all_data.Fare.dtype
all_data.Pclass1.dtype
all_data.Pclass2.dtype
all_data.Pclass3.dtype
all_data.High.dtype
all_data.Miss.dtype
all_data.Mr.dtype
all_data.Mrs.dtype
all_data.female.dtype
all_data.male.dtype
all_data.C.dtype
all_data.Q.dtype
all_data.S.dtype
all_data.Family.dtype


# ### 3.11 数据划分

# * 重新划分训练街测试集并保存

# In[ ]:


train = all_data[:891]
test = all_data[-418:]
test = test.drop(columns=['Survived'], axis=1, inplace=False)


# In[ ]:


# train.to_csv(f'{datadir}train_process.csv', index=False)
# test.to_csv(f'{datadir}test_process.csv', index=False)


# * 使用部分特征（丢掉与名字有关的特征）

# In[ ]:


all_data.head()


# In[ ]:


part_data = all_data.drop(columns=['High', 'Miss', 'Mr', 'Mrs', 'Other'], axis=1, inplace=False)
part_data.head()


# In[ ]:


train1 = part_data[:891]
test1 = part_data[-418:]
test1 = test1.drop(columns=['Survived'], axis=1, inplace=False)


# In[ ]:


# train1.to_csv(f'{datadir}part_train_process.csv', index=False)
# test1.to_csv(f'{datadir}part_test_process.csv', index=False)


# In[ ]:




