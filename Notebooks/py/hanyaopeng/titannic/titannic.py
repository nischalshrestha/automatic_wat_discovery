#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


root_path = '../input'
train = pd.read_csv('%s/%s' % (root_path, 'train.csv'))
test = pd.read_csv('%s/%s' % (root_path, 'test.csv'))


# In[ ]:


train.describe()


# In[ ]:


train.Survived.value_counts()


# In[ ]:


train_corr = train.drop('PassengerId',axis=1).corr()
train_corr
#相关性协方差表,corr()函数,返回结果接近0说明无相关性,大于0说明是正相关,小于0是负相关.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
# 画出相关性热力图
a = plt.subplots(figsize=(15,9))#调整画布大小
a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)  #画热力图


# In[ ]:


train.groupby(['Pclass'])['Pclass','Survived'].mean()   # 对 Pclass 进行分组 求平均值


# In[ ]:


train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()


# In[ ]:


train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()       # 统计 女性 存活率 和 男性存活率 


# In[ ]:


train.groupby(['Sex'])['Survived','Sex'].mean()


# In[ ]:


train[['SibSp','Survived']].groupby(['SibSp']).mean()


# In[ ]:


train[['Age','Survived']].groupby(['Age']).mean()


# In[ ]:


g = sns.FacetGrid(train, col='Survived',size=5)
g.map(plt.hist, 'Age', bins=50)  # bins 数据的宽度。


# In[ ]:


train.groupby(['Age'])['Survived'].mean().plot()


# In[ ]:


sns.countplot('Embarked',hue='Survived',data=train)


# In[ ]:


train.groupby(['Fare'])['Survived'].mean()


# In[ ]:


train.Fare.value_counts()


# In[ ]:


test['Survived'] = 0
train_test = train.append(test)


# In[ ]:


train_test.info()


# In[ ]:


train_test = pd.get_dummies(train_test,columns=["Sex"]) # 对性别进行分列
train_test.head(5)


# In[ ]:


train_test['SibSp_Parch'] = train_test['SibSp'] + train_test['Parch']


# In[ ]:


train_test.head(5)


# In[ ]:


train_test = pd.get_dummies(train_test,columns = ['SibSp','Parch','SibSp_Parch']) 


# In[ ]:


train_test.head(5)


# In[ ]:


train_test = pd.get_dummies(train_test,columns=["Embarked"])  # 对数据集按照列进行 分列。


# In[ ]:


train_test.head(5)


# In[ ]:


train_test['Lname'] = train_test.Name.apply(lambda x: x.split(' ')[0])
train_test['NamePrefix'] = train_test.Name.apply(lambda x: x.split(' ')[1])
train_test.head(5)


# In[ ]:


train_test = train_test.drop(['Name','Lname'],axis = 1)
train_test.head(5)


# In[ ]:


#从上面的分析,发现该特征train集无miss值,test有一个缺失值,先查看
train_test.loc[train_test["Fare"].isnull()]       # loc 行的索引号


# In[ ]:


#票价与pclass和Embarked有关,所以用train分组后的平均数填充
train.groupby(by=["Pclass","Embarked"]).Fare.mean()


# In[ ]:


#用pclass=3和Embarked=S的平均数14.644083来填充
train_test["Fare"].fillna(14.435422,inplace=True)


# In[ ]:


train_test.head()


# In[ ]:


#将Ticket提取字符列
#str.isnumeric()  如果S中只有数字字符，则返回True，否则返回False
train_test['Ticket_Letter'] = train_test['Ticket'].str.split().str[0]
train_test['Ticket_Letter'] = train_test['Ticket_Letter'].apply(lambda x:np.nan if x.isnumeric() else x)
train_test.drop('Ticket',inplace=True,axis=1)
train_test.head()


# In[ ]:


#分列,此时nan值可以不做处理
train_test = pd.get_dummies(train_test,columns=['Ticket_Letter'],drop_first=True)
train_test.head()


# In[ ]:


train_test.loc[train_test["Age"].isnull()]['Survived'].mean() # Age 不为空 存活率


# In[ ]:


# 所以用年龄是否缺失值来构造新特征
train_test.loc[train_test["Age"].isnull() ,"age_nan"] = 1
train_test.loc[train_test["Age"].notnull() ,"age_nan"] = 0
train_test = pd.get_dummies(train_test,columns=['age_nan'])
train_test.head()


# In[ ]:


train_test.info()


# In[ ]:


#创建没有['Age','Survived']的数据集
missing_age = train_test.drop(['Survived','Cabin'],axis=1)
#将Age完整的项作为训练集、将Age缺失的项作为测试集。
missing_age_train = missing_age[missing_age['Age'].notnull()]
missing_age_test = missing_age[missing_age['Age'].isnull()]
missing_age_train.head()
missing_age_test.head()


# In[ ]:


#构建训练集合预测集的X和Y值
missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
missing_age_Y_train = missing_age_train['Age']
missing_age_X_test = missing_age_test.drop(['Age'], axis=1)


# In[ ]:


missing_age_X_train.head()


# In[ ]:


missing_age_X_train=missing_age_X_train.drop(['NamePrefix'],axis = 1)
missing_age_X_test = missing_age_X_test.drop(['NamePrefix'],axis=1)


# In[ ]:


# 先将数据标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
#用测试集训练并标准化
ss.fit(missing_age_X_train)
missing_age_X_train = ss.transform(missing_age_X_train)
missing_age_X_test = ss.transform(missing_age_X_test)


# In[ ]:


missing_age_X_train


# In[ ]:


#使用贝叶斯预测年龄
from sklearn import linear_model
lin = linear_model.BayesianRidge()
lin.fit(missing_age_X_train,missing_age_Y_train)



# In[ ]:


train_test.loc[(train_test['Age'].isnull()), 'Age'] = lin.predict(missing_age_X_test)


# In[ ]:


train_test.isnull().count()


# In[ ]:


#将年龄划分是个阶段10以下,10-18,18-30,30-50,50以上
train_test['Age'] = pd.cut(train_test['Age'], bins=[0,10,18,30,50,100],labels=[1,2,3,4,5])

train_test = pd.get_dummies(train_test,columns=['Age'])


# In[ ]:


train_test.head()


# In[ ]:


#cabin项缺失太多，只能将有无Cain首字母进行分类,缺失值为一类,作为特征值进行建模
train_test['Cabin_nan'] = train_test['Cabin'].apply(lambda x:str(x)[0] if pd.notnull(x) else x)
train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
train_test.head()


# In[ ]:


#cabin项缺失太多，只能将有无Cain首字母进行分类,
train_test.loc[train_test["Cabin"].isnull() ,"Cabin_nan"] = 1
train_test.loc[train_test["Cabin"].notnull() ,"Cabin_nan"] = 0
train_test = pd.get_dummies(train_test,columns=['Cabin_nan'])
train_test.drop('Cabin',axis=1,inplace=True)
train_test.head()


# In[ ]:


train_data = train_test[:891]
test_data = train_test[891:]
train_data_X = train_data.drop(['Survived'],axis=1)
train_data_Y = train_data['Survived']
test_data_X = test_data.drop(['Survived'],axis=1)


# In[ ]:


train_data_X.head()


# In[ ]:


train_data_X=train_data_X.drop(['NamePrefix'],axis=1)
test_data_X=test_data_X.drop(['NamePrefix'],axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss2 = StandardScaler()
ss2.fit(train_data_X)
train_data_X_sd = ss2.transform(train_data_X)
test_data_X_sd = ss2.transform(test_data_X)


# In[ ]:


# 随机森林
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6,oob_score=True)
rf.fit(train_data_X,train_data_Y)
test["Survived"] = rf.predict(test_data_X)
RF = test[['PassengerId','Survived']].set_index('PassengerId')
RF.to_csv('RF.csv')


# In[ ]:


# xgboost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=150,min_samples_leaf=3,max_depth=6)
xgb_model.fit(train_data_X,train_data_Y)

test["Survived"] = xgb_model.predict(test_data_X)
XGB = test[['PassengerId','Survived']].set_index('PassengerId')
XGB.to_csv('XGB5.csv')


# In[ ]:


# 逻辑回归
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV

lr = LogisticRegression()
param = {'C':[0.001,0.01,0.1,1,10], "max_iter":[100,250]}
clf = GridSearchCV(lr, param,cv=5, n_jobs=-1, verbose=1, scoring="roc_auc")
clf.fit(train_data_X_sd, train_data_Y)

# 打印参数的得分情况
clf.grid_scores_
# 打印最佳参数
clf.best_params_

# 将最佳参数传入训练模型
lr = LogisticRegression(clf.best_params_)
lr.fit(train_data_X_sd, train_data_Y)

# 输出结果
test["Survived"] = lr.predict(test_data_X_sd)
test[['PassengerId', 'Survived']].set_index('PassengerId').to_csv('LS5.csv')


# In[ ]:


# SVM
from sklearn import svm
svc = svm.SVC()

clf = GridSearchCV(svc,param,cv=5,n_jobs=-1,verbose=1,scoring="roc_auc")
clf.fit(train_data_X_sd,train_data_Y)

clf.best_params_

svc = svm.SVC(C=1,max_iter=250)

# 训练模型并预测结果
svc.fit(train_data_X_sd,train_data_Y)
svc.predict(test_data_X_sd)

# 打印结果
test["Survived"] = svc.predict(test_data_X_sd)
SVM = test[['PassengerId','Survived']].set_index('PassengerId')
SVM.to_csv('svm1.csv')


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
# gbdt
gbdt = GradientBoostingClassifier(learning_rate=0.7,max_depth=6,n_estimators=100,min_samples_leaf=2)

gbdt.fit(train_data_X,train_data_Y)

test["Survived"] = gbdt.predict(test_data_X)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('gbdt3.csv')


# In[ ]:


# 模型融合 voting
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
xgb_model = xgb.XGBClassifier(max_depth=6,min_samples_leaf=2,n_estimators=100,num_round = 5)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200,min_samples_leaf=2,max_depth=6,oob_score=True)

from sklearn.ensemble import GradientBoostingClassifier
gbdt = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=2,max_depth=6,n_estimators=100)

vot = VotingClassifier(estimators=[( ('rf', rf),('gbdt',gbdt),('xgb',xgb_model))], voting='hard')
vot.fit(train_data_X_sd,train_data_Y)

test["Survived"] = vot.predict(test_data_X_sd)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('vot5.csv')


# In[ ]:


# 划分train数据集,调用代码,把数据集名字转成和代码一样
X = train_data_X_sd
X_predict = test_data_X_sd
y = train_data_Y

'''模型融合中使用到的各个单模型'''
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

clfs = [LogisticRegression(C=0.1,max_iter=100),
        xgb.XGBClassifier(max_depth=6,n_estimators=100,num_round = 5),
        RandomForestClassifier(n_estimators=100,max_depth=6,oob_score=True),
        GradientBoostingClassifier(learning_rate=0.3,max_depth=6,n_estimators=100)]

# 创建n_folds
from sklearn.cross_validation import StratifiedKFold
n_folds = 5
skf = list(StratifiedKFold(y, n_folds))

# 创建零矩阵
dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_predict.shape[0], len(clfs)))

# 建立模型
for j, clf in enumerate(clfs):
    '''依次训练各个单模型'''
    # print(j, clf)
    dataset_blend_test_j = np.zeros((X_predict.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        '''使用第i个部分作为预测，剩余的部分来训练模型，获得其预测的输出作为第i部分的新特征。'''
        # print("Fold", i)
        X_train, y_train, X_test, y_test = X[train], y[train], X[test], y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:, 1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_predict)[:, 1]
    '''对于测试集，直接用这k个模型的预测值均值作为新的特征。'''
    dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

# 用建立第二层模型
clf2 = LogisticRegression(C=0.1,max_iter=100)
clf2.fit(dataset_blend_train, y)
y_submission = clf2.predict_proba(dataset_blend_test)[:, 1]

test = pd.read_csv("../input/test.csv")
test["Survived"] = clf2.predict(dataset_blend_test)
test[['PassengerId','Survived']].set_index('PassengerId').to_csv('stack3.csv')


# In[ ]:




