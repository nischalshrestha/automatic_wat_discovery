#!/usr/bin/env python
# coding: utf-8

# # 一、读取数据

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))


# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_train


# In[ ]:


data_train.info()


# In[ ]:


data_train.describe()


# # 二、特征分析

# ## 与数值特征的关系

# In[ ]:


sns.boxplot(x='Survived',y='Age',data=data_train)


# In[ ]:


sns.boxplot(x='Survived',y='Fare',data=data_train)


# 可以看到，年龄和票价对生存率有些关系。

# ## 与类别特征的关系

# In[ ]:


sns.barplot(x='Sex',y='Survived',data=data_train)


# In[ ]:


sns.barplot(x='Embarked',y='Survived',hue='Sex',data=data_train)


# In[ ]:


sns.barplot(x='Pclass',y='Survived',data=data_train)


# In[ ]:


sns.barplot(x='SibSp',y='Survived',data=data_train)


# In[ ]:


sns.barplot(x='Parch',y='Survived',data=data_train)


# In[ ]:


fig, axis1 = plt.subplots(1,1,figsize=(18,4))
data_train['Name_length'] = data_train['Name'].apply(len)
name_length = data_train[['Name_length','Survived']].groupby(['Name_length'],as_index=False).mean()
print(name_length)
sns.barplot(x='Name_length', y='Survived', data=name_length)


# 可以看到，各个类别特征，除了Ticket复杂难以提取和Cabin缺失过多，其他类别特征都与生存率有一定关系，其中SibSp和Parch可以相加成FamilySize。

# ## 相关性矩阵（热力图）

# In[ ]:


corrmat = data_train.corr() #correlation matrix
sns.heatmap(corrmat, vmax=.8, annot=True, square=True);


# 可以看到，Survived与我们上面所观察到的相对一致，Age对Survived几乎是最小，但我们主观上还是要把Age加入特征。

# # 三、缺失数据

# In[ ]:


total = data_train.isnull().sum().sort_values(ascending=False)
percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# 删除Cabin，尝试用机器学习的方法填充Age，用众数填充Embarked。

# ## 删除Cabin

# In[ ]:


data_train = data_train.drop((missing_data[missing_data['Total'] > 177]).index,1)
data_train


# ## 填充Embarked

# In[ ]:


data_train.loc[data_train['Embarked'].isnull()].index


# In[ ]:


data_train.loc[data_train.Embarked.isnull(), 'Embarked'] = 'S'


# ## 生成FamilySize

# In[ ]:


data_train['FamilySize'] = data_train.Parch + data_train.SibSp
data_train


# ## 删除旧的无用特征

# In[ ]:


data_train.drop('Name', axis=1, inplace=True)
data_train


# In[ ]:


data_train.drop(['Parch','SibSp'], axis=1, inplace=True)
data_train


# In[ ]:


data_train.drop(['Ticket'], axis=1, inplace=True)
data_train


# ## 填充Age

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


# 把已有的数值型特征取出来丢进Random Forest Regressor中
age_df = data_train[['Age','Fare', 'FamilySize', 'Pclass', 'Name_length']]

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
predictedAges = rfr.predict(unknown_age[:, 1:])

data_train.loc[(data_train.Age.isnull()),'Age'] = predictedAges


# In[ ]:


data_train


# # 四、异常值检测

# 因为这里的任务是分类，所以不用进行异常值检测，如果任务是回归的话，则需要。

# # 五、特征统一化

# ## one-hot生成

# In[ ]:


data_train


# In[ ]:


dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

data_train = pd.concat([data_train, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data_train.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
data_train


# ## 预处理：正则化

# In[ ]:


import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(data_train['Age'].values.reshape(-1, 1))
data_train['Age_scaled'] = scaler.fit_transform(data_train['Age'].values.reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(data_train['Fare'].values.reshape(-1, 1))
data_train['Fare_scaled'] = scaler.fit_transform(data_train['Fare'].values.reshape(-1, 1), fare_scale_param)
namelength_scale_param = scaler.fit(data_train['Name_length'].values.reshape(-1, 1))
data_train['Name_length_scaled'] = scaler.fit_transform(data_train['Name_length'].values.reshape(-1, 1), namelength_scale_param)
familysize_scale_param = scaler.fit(data_train['FamilySize'].values.reshape(-1, 1))
data_train['FamilySize_scaled'] = scaler.fit_transform(data_train['FamilySize'].values.reshape(-1, 1), familysize_scale_param)
data_train.drop(['Age', 'Fare', 'Name_length', 'FamilySize'], axis=1, inplace=True)
data_train


# # 六、训练模型

# In[ ]:


def classifier_cv(model,X,y):
    classifier_loss = cross_validation.cross_val_score(model, X, y, cv=10)
    return classifier_loss


# In[ ]:


from sklearn import linear_model, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age_.*|Fare_.*|Name_.*|FamilySize_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

models = [linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-1, max_iter=100), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=25), GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=200), SVC(C=1, degree=2, kernel='rbf', probability=True)]
names = ["LR", "KNN", "RF", "GBC", "SVC"]
for name, model in zip(names, models):
    clf = model.fit(X, y)
    score = classifier_cv(clf, X, y)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))


# ## 网格化调参

# In[ ]:


class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X,y)
        print(grid_search.best_params_, grid_search.best_score_)
        print(pd.DataFrame(grid_search.cv_results_)[['params','std_test_score','mean_test_score']])


# ### LR

# In[ ]:


grid(linear_model.LogisticRegression()).grid_get(X, y, {'penalty':['l1', 'l2'], 'tol':[1e-1, 1e-2, 1e-3], 'max_iter':[100, 1000, 10000]})


# ### KNN

# In[ ]:


grid(KNeighborsClassifier()).grid_get(X, y, {'n_neighbors':[3, 4, 5, 6, 7]})


# ### RF

# In[ ]:


grid(RandomForestClassifier()).grid_get(X, y, {'n_estimators':[5, 10, 15, 20, 25, 30]})


# ### GBC

# In[ ]:


grid(GradientBoostingClassifier()).grid_get(X, y, {'n_estimators':[150, 200, 250], 'learning_rate':[1, 1e-1, 1e-2, 1e-3], 'max_depth':[2, 3, 4, 5]})


# ### SVC

# In[ ]:


grid(SVC(probability=True)).grid_get(X, y, {'C':[0.5, 0.75, 1, 1.25], 'kernel':['linear', 'poly', 'rbf', 'sigmoid'],'degree':[2, 3, 4]})


# # 七、测试集统一化

# In[ ]:


data_test = pd.read_csv('../input/test.csv')
data_test.loc[data_test.Embarked.isnull(), 'Embarked'] = 'S'
data_test['FamilySize'] = data_test.Parch + data_test.SibSp
data_test['Name_length'] = data_test['Name'].apply(len)
data_test.drop(['Cabin', 'Name', 'Parch', 'SibSp', 'Ticket'], axis=1, inplace=True)
tmp_df = data_test[['Age','Fare', 'FamilySize', 'Pclass', 'Name_length']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
Z = null_age[:, 1:]
predictedAges = rfr.predict(Z)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
data_test = pd.concat([data_test, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
data_test.drop(['Pclass', 'Sex', 'Embarked'], axis=1, inplace=True)
data_test['Age_scaled'] = scaler.transform(data_test['Age'].values.reshape(-1, 1), age_scale_param)
data_test['Fare_scaled'] = scaler.transform(data_test['Fare'].values.reshape(-1, 1), fare_scale_param)
data_test['Name_length_scaled'] = scaler.transform(data_test['Name_length'].values.reshape(-1, 1), namelength_scale_param)
data_test['FamilySize_scaled'] = scaler.transform(data_test['FamilySize'].values.reshape(-1, 1), familysize_scale_param)
data_test.loc[data_test.Fare_scaled.isnull(), 'Fare_scaled'] = data_train.Fare_scaled.mean()
data_test.drop(['Age', 'Fare', 'Name_length', 'FamilySize'], axis=1, inplace=True)
data_test


# # 八、预测

# In[ ]:


test_df = data_test.filter(regex='Age_.*|Fare_.*|Name_.*|FamilySize_.*|Embarked_.*|Sex_.*|Pclass_.*')
test_np = test_df.as_matrix()

models = [linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-1, max_iter=100), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=25), GradientBoostingClassifier(learning_rate=0.1, max_depth=3, n_estimators=200), SVC(C=1, degree=2, kernel='rbf', probability=True)]
names = ["LR", "KNN", "RF", "GBC", "SVC"]
for name, model in zip(names, models):
    clf = model.fit(X, y)
    score = classifier_cv(clf, X, y)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))
    predictions = clf.predict(test_np)
    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
    result.to_csv("./"+name+"_predictions.csv", index=False)

'''
predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("./predictions.csv", index=False)
'''


# # 九、检查模型系数

# In[ ]:


# pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})


# # 十、融合模型

# ## Bagging（相同权重或者不同权重）

# In[ ]:


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone, BaseEstimator
from sklearn.metrics import classification_report


# In[ ]:


class AverageWeight(BaseEstimator, RegressorMixin):
    def __init__(self, mod, weight):
        self.mod = mod
        self.weight = weight
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.mod]
        for model in self.models_:
            model.fit(X,y)
        return self
    
    def predict(self, X):
        w = list()
        pred = np.array([model.predict(X) for model in self.models_])
        # for every data point, single model prediction times weight, then add them together
        for data in range(pred.shape[1]):
            single = [pred[model,data]*weight for model,weight in zip(range(pred.shape[0]),self.weight)]
            w.append(np.sum(single))
        return w


# In[ ]:


len(models)


# In[ ]:


w=[0.1, 0.2, 0.1, 0.2, 0.4]


# In[ ]:


weight_avg = AverageWeight(mod = models, weight = w)


# In[ ]:


cross_validation.cross_val_score(weight_avg, X, y, cv=2)


# 这里有个很奇怪的问题，原作者在`weight_avg`初始化之后就计算损失，貌似少了一步。我自己用了交叉验证中的`cross_val_score`进行多折验证，但是这样出来的正确率很低，前前后后琢磨调试了两天，还是找不出原因，自己猜测很可能是`cross_val_score()`中某些设置的原因，因为自己也亲自拟合和预测，然后比对结果，这样出来的结果没有问题，还是等到了房价预测再试试会不会有相同情况。

# ## Stacking

# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,mod,meta_model):
        self.mod = mod
        self.meta_model = meta_model
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)
        
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]
        oof_train = np.zeros((X.shape[0], len(self.mod)))
        
        for i,model in enumerate(self.mod):
            for train_index, val_index in self.kf.split(X,y):
                renew_model = clone(model)
                renew_model.fit(X[train_index], y[train_index])
                self.saved_model[i].append(renew_model)
                oof_train[val_index,i] = renew_model.predict(X[val_index])
        
        self.meta_model.fit(oof_train,y)
        return self
    
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model])
        return self.meta_model.predict(whole_test)
    
    def get_oof(self,X,y,test_X):
        oof = np.zeros((X.shape[0],len(self.mod)))
        test_single = np.zeros((test_X.shape[0],5))
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):
                clone_model = clone(model)
                clone_model.fit(X[train_index],y[train_index])
                oof[val_index,i] = clone_model.predict(X[val_index])
                test_single[:,j] = clone_model.predict(test_X)
            test_mean[:,i] = test_single.mean(axis=1)
        return oof, test_mean


# ## 普通的Stacking

# In[ ]:


stack_model = stacking(mod=models,meta_model=SVC(C=1, degree=2, kernel='rbf', probability=True))


# In[ ]:


stack_model


# In[ ]:


stack_score = cross_validation.cross_val_score(stack_model, X, y, cv=5)


# In[ ]:


np.mean(stack_score)


# 跟bagging类似的正确率……无法理解，找不出是哪里写错了，还是下一题在房价预测那里试试好了。

# ## 改进的Stacking：与原有特征进行拼接

# In[ ]:


X_train_stack, X_test_stack = stack_model.get_oof(X, y, test_np)


# In[ ]:


X_train_stack.shape


# In[ ]:


X_test_stack.shape


# In[ ]:


X_train_add = np.hstack((X, X_train_stack))


# In[ ]:


X_train_add.shape


# In[ ]:


X_test_add = np.hstack((test_np,X_test_stack))


# In[ ]:


X_test_add.shape


# In[ ]:


m_stack_score = cross_validation.cross_val_score(stack_model, X_train_add, y, cv=5)


# In[ ]:


np.mean(m_stack_score)


# # 最后
# 最后的模型融合不知道什么原因不起作用，只能前面尝试的单模型尝试提交，最终SVC拿到了最高分，0.78947，top30%左右，不过这个名次不能说明什么，每两个星期没提交就会被清除leaderboard。
