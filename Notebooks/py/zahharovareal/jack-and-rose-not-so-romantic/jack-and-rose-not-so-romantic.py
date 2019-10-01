#!/usr/bin/env python
# coding: utf-8

# **故事开始了**
# -----
# 
#  1. 问题背景定义
#  2. 获取训练和测试数据
#  3. 数据清洗
#  4. 探索性分析
#  5. 特征工程
#  6. 建立模型进行预测
#  7. 模型评估
# 
# 
# 

# ## 问题以及背景 ##
# 
#  -  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. 
#  - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.（告诉我们获救与否不是随机的，而是有一些条件的）
# 
#  
# ## 目标/衡量指标##
# 
#  - Goal
#  It is your job to predict if a passenger survived the sinking of the Titanic or not. 
# For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
#  - Metric
# Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.

# In[ ]:


#数据分析库
import pandas as pd 
import numpy as np 

#数据可视化
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

#机器学习
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# ## 获取训练以及测试数据##

# In[ ]:


data_test = pd.read_csv("../input/test.csv")
data_train = pd.read_csv("../input/train.csv")
data_train.head()


# ## 哪些属性有缺失值？ ##
#  - 训练数据集共有891个记录。age和cabin属性有很多缺失值，embarked属性有两个缺失值。
#  - 测试数据集共有418个记录。同样的，age和cabin属性有很多缺失值，fare属性有一个缺失值。
# 
# **有哪些是类别属性？**
# -------------
# 
#  - Survived,Sex,and Embarked(non-ordinal 无序)。Pclass(ordinal 有序)
# 
# **有哪些是数值属性？**
# -------------
# 
#  - Age,Fare(连续值)。SibSp,Parch(离散值)
# 
# 

# In[ ]:


data_train.info()
data_test.info()


# ## 训练样本中数值属性的分布是怎样的？ ##
# 
#  - 训练样本数目是891，是泰坦尼克号真实人数（2224）的40%
#  
#  - Suvived是类别属性（1-存活，0-未存活），训练样本中大约有38%的人获救了。
#  - ...

# In[ ]:


data_train.describe()


# ## 数据可视化分析 ##
# 表格看晕了，直接画几张图看看吧
# 
#  - 获救的人数300多一点，三等舱人数最多，获救年龄分布很广，头等舱的年纪偏大，S登船口人数最多。
#  - 我们大概有个假设：年龄/乘客等级对获救可能有影响。

# In[ ]:


fig = plt.figure(figsize=(13,7))
fig.set(alpha=0.2)  # 设定图表颜色alpha参数
plt.rcParams['axes.unicode_minus'] = False #设置负号的正常显示


plt.subplot2grid((2,3),(0,0))  # 在一张大图里分列几个小图

data_train.Survived.value_counts().plot(kind='bar')
plt.title("Suvived ") 
plt.ylabel("count")  

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel("count")
plt.title("Pclass")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel("Age")                         
plt.grid(b=True, which='major', axis='y') 
plt.title("survived distribution by age")


plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')   
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel("age")
plt.ylabel("density") 
plt.title("age distribution by class")
plt.legend(('1', '2','3'),loc='best') # 设置图表的图例

# 上面讲到在训练集中embarked有两个缺失值，我们这里用出现最多次数的S代替
data_train["Embarked"] = data_train["Embarked"].fillna("S")

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title("count of each embarked")
plt.ylabel("count")  
plt.show()


# 数据可视化分析
# -------
# 现在我们可以画图继续验证我们的假设，上面是统计了一些基本情况。接下来我们看看具体的每个属性与获救与否的关系。
# 
#  - 一等舱的获救比例明显高于三等舱。说明乘客等级肯定对模型有影响，所以应该作为一个特征。
#  - 女性获救比例明显高于男性。说明性别肯定对模型有影响，所以应该作为一个特征。
#  - 我们创建了一个新的特征-家庭（sibsp+parch）,可以看出有家庭的获救比例高一些，所以也可以作为一个特征。
#  - ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，不纳入考虑的特征范畴
#  - 有cabin记录的乘客survival比例稍高，那先试试把这个值分为两类，有cabin值/无cabin值，加到类别属性
#  - 我们用随机森林回归预测年龄的空缺值，年幼的有更多的机会获救，所以把年纪纳入一个特征。

# In[ ]:


#看看各乘客等级的获救情况
fig = plt.figure(figsize=(13,7))
fig.set(alpha=0.2)  # 设定图表颜色alpha参数


Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'survival':Survived_1, 'unsurvival':Survived_0})
df.plot(kind='bar', stacked=True)
plt.title("survived by Pclass")
plt.xlabel("Pclass") 
plt.ylabel("count") 

plt.show()


# In[ ]:


#看看各登录港口的获救情况
fig = plt.figure(figsize=(13,7))
fig.set(alpha=0.2)  # 设定图表颜色alpha参数


Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df=pd.DataFrame({'Survived_1':Survived_1, 'Survived_0':Survived_0})
df.plot(kind='bar', stacked=True)#堆积条形图
plt.title("survived by each embarked ")
plt.xlabel("embarked") 
plt.ylabel("count") 

plt.show()


# In[ ]:


#看看各性别的获救情况
fig = plt.figure(figsize=(13,7))
fig.set(alpha=0.2)  # 设定图表颜色alpha参数


Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
df.plot(kind='bar', stacked=True)
plt.title("survived by sex")
plt.xlabel("sex") 
plt.ylabel("count")
plt.show()


# In[ ]:


# 家庭
# 我们可以用家庭替代sibsp（兄弟姐妹/夫妇）和parch（父母/孩子）两列
data_train['Family'] =  data_train["Parch"] + data_train["SibSp"]
data_train['Family'].loc[data_train['Family'] > 0] = 1
data_train['Family'].loc[data_train['Family'] == 0] = 0
data_test['Family'] =  data_test["Parch"] + data_test["SibSp"]
data_test['Family'].loc[data_test['Family'] > 0] = 1
data_test['Family'].loc[data_test['Family'] == 0] = 0

# drop Parch & SibSp
data_train = data_train.drop(['SibSp','Parch'], axis=1)
data_test= data_test.drop(['SibSp','Parch'], axis=1)

import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')
# plot
fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

sns.countplot(x='Family', data=data_train, order=[1,0], ax=axis1)

family_perc = data_train[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

axis1.set_xticklabels(["With Family","Alone"], rotation=0)


# In[ ]:


#cabin
#看看这个值的有无，对于survival的分布状况，影响如何
fig = plt.figure(figsize=(13,7))
fig.set(alpha=0.5)  # 设定图表颜色alpha参数

Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df=pd.DataFrame({'havecabin':Survived_cabin, 'nocabin':Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title("survived by cabin")
plt.xlabel("have cabin record?") 
plt.ylabel("count")
plt.show()


def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df
data_train = set_Cabin_type(data_train)



# In[ ]:


#年龄
#使用 RandomForestClassifier 填补缺失的年龄属性
from sklearn.ensemble import RandomForestRegressor
def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Family', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    return df, rfr



data_train, rfr = set_missing_ages(data_train)

data_train.head()

#绘制年龄的核密度图
facet = sns.FacetGrid(data_train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, data_train['Age'].max()))
facet.add_legend()
#按年龄看平均获救率
fig, axis1 = plt.subplots(1,1,figsize=(14,4))
average_age = data_train[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)


# ## 特征离散化 ##
# 
#  - 因为逻辑回归建模时，需要输入的特征都是数值型特征
#  - 我们先对类目型的特征离散/因子化
#  - 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
#  - 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
#  - 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
#  - 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上

# In[ ]:



dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df.head()


# 数据归一化
# -----
# 
#  接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
# 这样可以加速logistic regression的收敛
# 
# 

# In[ ]:


import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)


# In[ ]:


# 我们把需要的feature字段取出来，转成numpy格式
train_df = df.filter(regex='Survived|Age_.*|Family|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]
X.shape


# ## 测试集上的数据变换 ##

# In[ ]:


# 对于测试集，fare有一个缺失值，我们用中位数插值
data_test["Fare"].fillna(data_test["Fare"].median(), inplace=True)

# 接着我们对test_data做和train_data中一致的特征变换

# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Family', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

#对测试集的cabin,embarked,sex,pclass作因子化
data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

#对测试集中的年龄和票价做归一化
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)
df_test.head()


# 
# 模型，预测
# -----
# 这是有监督的分类问题，我们可以考虑如下模型
# 
#  - LogisticRegression
#  - Support Vector Machines
#  - Random Forests
#  - Gaussian Naive Bayes
#  - knn
#  - ...

# In[ ]:


# 我们把需要的feature字段取出来，转成numpy格式
train_df = df.filter(regex='Survived|Age_.*|Family|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

#对测试集做同样处理
test = df_test.filter(regex='Age_.*|Family|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')


# In[ ]:


# LogisticRegression建模
from sklearn import linear_model
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)
#预测
Y_pred= clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':Y_pred.astype(np.int32)})
result.to_csv("clf.csv", index=False)
#训练集上的得分
clf.score(X, y)


# 通过逻辑回归模型的系数，可以验证我们的特征选择的是否够好。

# In[ ]:


pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})


# In[ ]:



# Support Vector Machines

svc = SVC()
svc.fit(X, y)

Y_pred= svc.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':Y_pred.astype(np.int32)})
result.to_csv("svc.csv", index=False)

svc.score(X, y)


# In[ ]:


# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)

Y_pred = random_forest.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':Y_pred.astype(np.int32)})
result.to_csv("random_forest.csv", index=False)
random_forest.score(X, y)


# In[ ]:


#Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X, y)

Y_pred = gaussian.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':Y_pred.astype(np.int32)})
result.to_csv("gaussian.csv", index=False)
gaussian.score(X, y)


# In[ ]:


#knn
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y)

Y_pred = knn.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':Y_pred.astype(np.int32)})
result.to_csv("knn.csv", index=False)
knn.score(X, y)

