#!/usr/bin/env python
# coding: utf-8

# ## 1. 导入包

# In[ ]:


import pandas as pd
import numpy as np

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)


# ## 2. 数据加载

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# ## 3. 数据处理

# ### 3.1 观察特征

# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.describe()


# In[ ]:


test.describe()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


# any的意思是其中有null的，all的意思是这一列中全是null
train_null_columns = train.columns[train.isnull().any()]


# In[ ]:


test_null_columns = test.columns[test.isnull().any()]


# 总结： train中找到有缺失的特征"Age"、"Cabin"、"Embarked"；test中找到有缺失的特征"Age"、"Fare"、"Cabin"

# ### 3.2 缺失值的弥补

# 先从缺失少的开始，分别是train中的embarked和test中的fare

# **3.2.1 Embarked**

# In[ ]:


train[train['Embarked'].isnull()]


# embarked的意思是港口号。
# 
# 这两位都是女士，都在一等舱，船票费用都为80美金。
# 
# 也许可以通过其他在一等舱的，船票费用也为80美金的人的港口号来推测。

# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train);


# 用“箱形图”进行可视化。可见，pclass=1，embarked=C的乘客的fare中位数是80左右
# 
# 中位数表示整体数据的“集中趋向”，可以用来简单地弥补缺失数据。
# 
# 所以这里我们就将缺失的embarked标记为C

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('C')


# **3.2.2 Fare**  

# 还是使用中位数来补充好了

# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test);


# What a pity!!! 看不清楚.......那写个算法来找找？
# 
# 要找出pclass=3，embarked=S的乘客船票费的中位数

# In[ ]:


fare_median = test[(test['Pclass'] == 3) & (test['Embarked'] == 'S')]['Fare'].median()


# In[ ]:


test['Fare'] = test['Fare'].fillna(fare_median)


# **3.2.3 Age & Cabin** 

# 因为age与生存率息息相关，后面将使用随机森林来更加准确地预测年龄。
# 
# 而cabin中其实deck才是重要的，后面会将deck从cabin中分离出来，并补上缺失的deck
# 
# 在这之前，先引入一些新的特征。

# ### 3.3 新特征的生成

# **3.3.1 Deck** 

# 生存几率与处于船舱的哪个位置也有很大关系。
# 
# 通过cabin来获取船舱数据。

# In[ ]:


train['Deck'] = train['Cabin'].str[0]
test['Deck'] = test['Cabin'].str[0]


# In[ ]:


train['Deck'].unique() #查看船舱


# In[ ]:


train[train['Deck'].isnull()]['Pclass'].describe()


# In[ ]:


test['Deck'].unique() #查看船舱


# In[ ]:


test[test['Deck'].isnull()]['Pclass'].describe()


# 发现基本上都是三等舱的人，所以根据三等舱的取个众数吧。

# In[ ]:


train.loc[train['Pclass'] == 3]['Deck'].describe()


# In[ ]:


test.loc[test['Pclass'] == 3]['Deck'].describe()


# 看到众数是F，所以填补上F，但是经过实验发现，众数还不如随便一个没有出现过的数，不知道为什么，所以这里还是补上一个没有出现过的deck值‘Z’好了

# In[ ]:


# train['Deck'] = train['Deck'].fillna('F')
# test['Deck'] = test['Deck'].fillna('F')

train['Deck'] = train['Deck'].fillna('Z')
test['Deck'] = test['Deck'].fillna('Z')


# In[ ]:


train['Deck'].unique() #查看船舱


# In[ ]:


test['Deck'].unique() #查看船舱


# **3.3.2 Family** 

# 把SibSp和Parch合成一个family特征

# In[ ]:


train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1


# 看一下family和生存率的关系

# In[ ]:


sns.countplot(x="Family", hue="Survived", data=train);


# 看到“独身一人”的死亡率高于生存率 - singleton
# 
# 2～4人随行的生存率高于死亡率 - small
# 
# 大于5人以上随行的也是死亡率高于生存率 - large
# 
# 重新将其分类为“singleton”、“small”、“large”

# In[ ]:


#train[train['Family'] == 1]['FamilyType'] = 'singleton'
#train[(train['Family'] > 1) & (train['Family'] < 5)]['FamilyType'] = 'small'
#train[train['Family'] > 4]['FamilyType'] = 'large'

#要使用loc才行，一部分不能作为左值
train.loc[train['Family'] == 1, 'FamilyType'] = 'singleton'
train.loc[(train['Family'] > 1) & (train['Family'] < 5), 'FamilyType'] = 'small'
train.loc[train['Family'] > 4, 'FamilyType'] = 'large'


# In[ ]:


test.loc[test['Family'] == 1, 'FamilyType'] = 'singleton'
test.loc[(test['Family'] > 1) & (test['Family'] < 5), 'FamilyType'] = 'small'
test.loc[test['Family'] > 4, 'FamilyType'] = 'large'


# **3.3.3 Title** 

# 那个年代的外国人的名字中包含着称谓，通过这个称谓可以分析出“性别”、“地位”、“家族”等信息

# In[ ]:


train['Name']


# In[ ]:


import re

def get_title(name):
    title = re.compile('(.*, )|(\\..*)').sub('',name)
    return title

titles = train['Name'].apply(get_title)
print(pd.value_counts(titles))
train['Title'] = titles

titles = test['Name'].apply(get_title)
test['Title'] = titles


# title种类太多了，合并一些title。

# In[ ]:


rare_title = ['Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer']
train.loc[train["Title"] == "Mlle", "Title"] = 'Miss'
train.loc[train["Title"] == "Ms", "Title"] = 'Miss'
train.loc[train["Title"] == "Mme", "Title"] = 'Mrs'
train.loc[train["Title"] == "Dona", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Lady", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Countess", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Capt", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Col", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Don", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Major", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Rev", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Sir", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Jonkheer", "Title"] = 'Rare Title'
train.loc[train["Title"] == "Dr", "Title"] = 'Rare Title'


# In[ ]:


test.loc[test["Title"] == "Mlle", "Title"] = 'Miss'
test.loc[test["Title"] == "Ms", "Title"] = 'Miss'
test.loc[test["Title"] == "Mme", "Title"] = 'Mrs'
test.loc[test["Title"] == "Dona", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Lady", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Countess", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Capt", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Col", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Don", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Major", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Rev", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Sir", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Jonkheer", "Title"] = 'Rare Title'
test.loc[test["Title"] == "Dr", "Title"] = 'Rare Title'


# In[ ]:


test['Title'].value_counts()


# **接上一节3.2.3 Age** 

# 随机森林对Age的预测

# 首先是将“分类”转变成“one-hot”形式

# In[ ]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelEnc=LabelEncoder()

cat_vars=['Embarked','Sex',"Title","FamilyType",'Deck']
for col in cat_vars:
    train[col]=labelEnc.fit_transform(train[col])
    test[col]=labelEnc.fit_transform(test[col])

train.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

def fill_missing_age(data):
    
    #Feature set
    features = data[['Age','Embarked','Fare', 'Parch', 'SibSp',
                 'Title','Pclass','Family',
                 'FamilyType', 'Deck']]
    # Split sets into train and prediction
    train  = features.loc[ (data.Age.notnull()) ]# known Age values
    prediction = features.loc[ (data.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(prediction.values[:, 1::])
    
    # Assign those predictions to the full data set
    data.loc[ (data.Age.isnull()), 'Age' ] = predictedAges 
    
    return data

train=fill_missing_age(train)
test=fill_missing_age(test)


# ### 3.4 数据归一化

# 这里的连续变量有“Age”和“Fare”，来看看它们。

# In[ ]:


train['Age'].describe()


# In[ ]:


train['Fare'].describe()


# 这两个连续变量差的多，做一个数据归一化。

# In[ ]:


from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(train[['Age', 'Fare']])
train[['Age', 'Fare']] = std_scale.transform(train[['Age', 'Fare']])


std_scale = preprocessing.StandardScaler().fit(test[['Age', 'Fare']])
test[['Age', 'Fare']] = std_scale.transform(test[['Age', 'Fare']])


# ### 3.5 关联特征分析

# In[ ]:


correlation=train.corr()
plt.figure(figsize=(10, 10))

sns.heatmap(correlation, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='YlGnBu',linecolor="white")
plt.title('Correlation between features');


# In[ ]:


train.corr()['Survived']


# 看survived这一列，正相关比较高的是fare（交了越多的钱，就越有可能活下来）；
# 
# 负相关最高的是sex（女子比男子更容易活下来），然后是pclass（class越小，即等级越高的，活下来可能性越大），然后是deck（月靠前，越容易存活）。

# In[ ]:


print(train.isnull().sum())
print(test.isnull().sum())


# ## 4. 训练模型，预测开始

# In[ ]:


features = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
             "Embarked", "FamilyType", "Title","Deck"]
#features = ["Pclass", "Sex", "Age", "Fare", "Family",
              #"Embarked", "FamilyType", "Title","Deck"]
target="Survived"

X_train = train[features]

y_train = train[target]

X_test = test[features]


# ### 4.1 Linear Regression

# 参考了[Poonam Ligade‘s post][1]中的linear regression
# 
# 
#   [1]: https://www.kaggle.com/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

lr = LinearRegression()

lr.fit(X_train, y_train)

y_test = lr.predict(X_test)

y_test[y_test > .5] = 1
y_test[y_test <=.5] = 0

y_test = y_test.astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic_linear.csv', index=False)


# ### 4.2 Decision Tree

# In[ ]:


from sklearn import tree

dtc = tree.DecisionTreeClassifier()
dtc = dtc.fit(X_train, y_train)

y_test = dtc.predict(X_test)
y_test = y_test.astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic_tree.csv', index=False)


# ### 4.3 Random Forest

# In[ ]:


from sklearn import ensemble

rf = ensemble.RandomForestClassifier()
rf = rf.fit(X_train, y_train)

y_test = rf.predict(X_test)
y_test = y_test.astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic_forest.csv', index=False)


# ### 4.4 XGBoost

# In[ ]:


import xgboost as xgb

model = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)

y_test = model.predict(X_test)
y_test = y_test.astype(int)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": y_test
    })
submission.to_csv('titanic_xgboost.csv', index=False)


# ## 5. 总结
# 
# 最后的实验结果：
# 
# No1. Linear Regression - 0.78947
# 
# No2. Radom Forest - 0.77512
# 
# No3. XGBoost - 0.74641
# 
# No4. Decision Tree - 0.67464
