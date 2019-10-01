#!/usr/bin/env python
# coding: utf-8

# It's my first DataMining

# In[ ]:


#导入模块和读取数据
import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
full_df = [train_df, test_df]


# In[ ]:


#检查数据集完整性
#发现Age,Fare,Cabin,Embarked几个特征有缺失值
print(train_df.shape)
print(test_df.shape)
print(train_df.columns)
print(test_df.columns)

print(' _'*40)
train_df.info()
print(' _'*30)
test_df.info()


# In[ ]:


#查看数据分布统计数据,
#观察到'PassengerId','Name','Ticket', 'Cabin'4个特征的值相当分散
#观察到'Fare'的方差、均值都较大，数据可能呈现偏态分布
pd.options.display.float_format = '{:,.2f}'.format 
print(train_df.describe())
print(train_df.describe(include=['O']))


# In[ ]:


#去除不必要的特征,减少计算量
train_df = train_df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Name','Ticket', 'Cabin'], axis=1)
full_df = [train_df, test_df]
train_df.head()


# In[ ]:


#求生存率比例
surv = train_df[train_df['Survived']==1]
nosurv = train_df[train_df['Survived']==0]

print("Is Survived : Not Survived = %.1f : %.1f, Total: %i"      %(len(surv)/len(train_df)*10,len(nosurv)/len(train_df)*10, len(train_df)))


# In[ ]:


#用直方图显示各特征值密度分布
#train_df = train_df.set_index('Survived')
#train_df['Age'].plot(kind='bar')
#train_df.plot(x='Survived', y='Age', kind='scatter')
#train_df.plot(kind='kde')
train_df.ix[:,:-1].hist(figsize=(6,8))


# In[ ]:


##因为Parch和SibSp的数据分布较接近，因此考虑合并
for dataset in full_df:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
full_df = [train_df, test_df]

print(train_df.head())


# In[ ]:


#用频繁值填充登船地点，用平均数填充费用值
freq_port = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'] = train_df['Embarked'].fillna(freq_port)
test_df['Fare'].fillna(test_df['Fare'].dropna().mean(), inplace=True)


# In[ ]:


#将性别特征、登船地点进行数值化
for dataset in full_df:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_df.head()


# In[ ]:


#打印关系矩阵图，找到和Age相关性最大的特征
#上一版本Parch和SibSp2个特征对Age的相关性还有些差异，下次改进
f,ax = plt.subplots(figsize=(12, 9))
sns.heatmap(train_df.corr(), vmax=1, square=True, annot = True);


# In[ ]:


#用图形显示各特征值与生存率的关系
#FamilySize、Fare两个图示还不完善
plt.figure(figsize=[16,8])
plt.subplots_adjust( hspace=0.2,wspace=0.2)

plt.subplot(231)
sns.barplot('Pclass', 'Survived', data=train_df)
plt.subplot(232)
sns.barplot('Sex', 'Survived', data=train_df)
plt.subplot(234)
sns.barplot('FamilySize', 'Survived', data=train_df)
plt.subplot(235)
sns.barplot('Embarked', 'Survived', data=train_df)
#plt.subplot(334)
#sns.barplot('SibSp', 'Survived', data=train_df)
#plt.subplot(335)
#sns.barplot('Parch', 'Survived', data=train_df)
plt.subplot(233)
sns.distplot(surv['Age'].dropna().values, bins=20, kde=False,color="blue")
sns.distplot(nosurv['Age'].dropna().values, bins=20, kde=False,color="red",axlabel='Age')
plt.subplot(236)
sns.distplot(np.log10(surv['Fare'].dropna().values+1), bins=10, kde=False )
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1),bins=10,  kde=False, color="red",axlabel='Fare')


# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 20)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# In[ ]:


for dataset in full_df:
    dataset.loc[ dataset['Fare'] <= 10.5, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 13), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 13) & (dataset['Fare'] <= 16.1), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 16.1) & (dataset['Fare'] <= 26), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 26) & (dataset['Fare'] <= 27), 'Fare'] = 4
    dataset.loc[(dataset['Fare'] > 27) & (dataset['Fare'] <= 39.688), 'Fare'] = 5
    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 77.958), 'Fare']   = 6
    dataset.loc[ dataset['Fare'] > 77.958, 'Fare'] = 7
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
full_df = [train_df, test_df]
    
train_df.head()


# In[ ]:


#图形显示Age和Pclass的关联，
grid = sns.FacetGrid(train_df, col='Pclass', size=2, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)


# In[ ]:


#图形显示Age和FamilySize的关联
grid = sns.FacetGrid(train_df, col='FamilySize', size=11, aspect=1.5)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)


# In[ ]:


#根据FamilySize特征分别和Age、Survived的相关分布，对原本的数值做出精简
for dataset in full_df:
    dataset['FamilySize'] = dataset['FamilySize'].map( {1:0,2:1,2:1,3:1,4:1,5:2,6:2,7:2,8:3,9:3,10:3,11:3} ).astype(int)
train_df.head()


# In[ ]:


#按Pclass和FamilySize关系补充缺失的年龄值,这里出错
#ValueError: cannot convert float NaN to integer
#FamilySize值暂时先用IsAlone代替完成,下次改进
#*******************************************
for dataset in full_df:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 0, 'IsAlone'] = 1
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

full_df = [train_df, test_df]

guess_ages = np.zeros((2,3))

for dataset in full_df:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['IsAlone'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.IsAlone == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df = train_df.drop(['IsAlone'], axis=1)
test_df = test_df.drop(['IsAlone'], axis=1)
full_df = [train_df, test_df]    

train_df.info(),train_df.head()


# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 20)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# In[ ]:


for dataset in full_df:    
    dataset.loc[ dataset['Age'] <= 8, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 8) & (dataset['Age'] <= 12), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 12) & (dataset['Age'] <= 16), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 36), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 48), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 52), 'Age'] = 6
    dataset.loc[(dataset['Age'] > 52) & (dataset['Age'] <= 64), 'Age'] = 7
    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 76), 'Age'] = 8
    dataset.loc[(dataset['Age'] > 76) , 'Age'] = 9
train_df = train_df.drop(['AgeBand'], axis=1)
full_df = [train_df, test_df]
train_df.head()


# In[ ]:


#计算各特征值的支持度
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Pclass', ascending=True))
print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Sex', ascending=True))
print(train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True))
print(train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=True))
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))


# In[ ]:





# In[ ]:


test_df.head(10)


# In[ ]:





# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[ ]:


# 导入机器学习模块
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# 训练集对测试集的逻辑回归的准确率得分，比原题提高0.22百分点
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[ ]:


#得到各个特征的相关系数
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[ ]:


#得到的预测准确率平均比原题提高3.675%
#提高最大的是，梯度下降，提升15.15%
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)


# In[ ]:




