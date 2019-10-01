#!/usr/bin/env python
# coding: utf-8

# ## Introduction ##
# 
# This is my first work of machine learning. the notebook is written in python and has inspired from ["Exploring Survival on Titanic" by Megan Risdal, a Kernel in R on Kaggle][1].
# 
# #### Forked from Sina:https://www.kaggle.com/sinakhorami/titanic-best-working-classifier?scriptVersionId=566580
# #### 修改人：白文杰
# 
# 原notebook存在的问题：
# - age,fare没有对测试集做变换。
# - 应该对类别数据采用one-hot encoding。
# - 应考虑进行异常值去除，数据纠偏，数据标准化等。(使用分位数离散化数值型数据已起到相似效果)
# - 最终考虑使用F1分数评估，accuracy容易过拟合。
# - 可以考虑使用XGBoost。
# - 可以提供更多的参数进行网格搜索。
# - 最终使用Voting把所有分类器组合起来。
# 
# 
#   [1]: https://www.kaggle.com/mrisdal/titanic/exploring-survival-on-the-titanic

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import re as re

# header : int or list of ints, default 'infer'
#     Row number(s) to use as the column names, and the start of the
#     data.  Default behavior is to infer the column names: if no names
#     are passed the behavior is identical完全相同的 to ``header=0`` and column
#     names are inferred from the first line of the file, if column
#     names are passed explicitly显式的 then the behavior is identical to
#     ``header=None``. Explicitly pass ``header=0`` to be able to
#     replace existing names. The header can be a list of integers that
#     specify row locations for a multi-index on the columns
train = pd.read_csv('../input/train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('../input/test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test] # 没有拷贝原对象，修改此处的对象即修改原对象

print (train.info())
train.head()


# # Feature Engineering #

# ## 1. Pclass ##
# there is no missing value on this feature and already a numerical value. so let's check it's impact on our train set.
# 
# 比较实验中不同水平的均值是否有显著差别来判断因素的显著性，思想类似单因素方差分析。（假设各水平的总体相互独立且服从同方差的正态分布，误差满足iid $N(0,\sigma^2)$）
# 
# 【此处对于不同因素对于结果的显著性讨论更为准确，理论性的讨论考虑应该使用数据集不平衡时的方差分析方法（数据个数不同置信度应该不同）。】
# 
# 不显著的因子考虑引入domain knowledge来进行特征组合转换提高显著性。
# 
# scikit-learn中没有方差分析。R语言中的多元方差分析：https://blog.csdn.net/u011955252/article/details/50729815

# In[ ]:


print (train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean())
print (train[['Pclass', 'Survived']].groupby(['Pclass']).mean())


# ## 2. Sex ##

# In[ ]:


print (train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean())


# ## 3. SibSp and Parch ##
# With the number of siblings/spouse and the number of children/parents we can create new feature called Family Size.

# In[ ]:


# 均值差异不大
print (train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean())
print (train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean())


# In[ ]:


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
print (train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean())


# it seems has a good effect on our prediction but let's go further and categorize people to check whether they are alone in this ship or not.

# In[ ]:


for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())


# good! the impact is considerable.

# ## 4. Embarked ##
# the embarked feature has some missing value. and we try to fill those with the most occurred value ( 'S' ).

# In[ ]:


for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
print (train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())


# ## 5. Fare ##
# Fare also has some missing value and we will replace it with the median. then we categorize it into 4 ranges.

# In[ ]:


for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 4) # Quantile-based discretization离散化 function.考虑异常值的切

print(type(train['CategoricalFare'][0]))
print (train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())


# ## 6. Age ##
# we have plenty of missing values in this feature. # generate random numbers between (mean - std) and (mean + std).
# then we categorize age into 5 range.

# In[ ]:


for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

    # 【cut 和 qcut有区别】
    # 并非使用分位数而是使用数值，符合年龄的特性
    dataset['CategoricalAge'] = pd.cut(dataset['Age'], 5)

print (train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())


# ## 7. Name ##
# inside this feature we can find the title of people.

# In[ ]:


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)

print(pd.crosstab(train['Title'], train['Sex']))


#  so we have titles. let's categorize it and check the title impact on survival rate.

# In[ ]:


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

print (train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())


# In[ ]:


print(train.columns)
print(test.columns)


# # Data Cleaning #
# great! now let's clean our data and map our features into numerical values.

# In[ ]:


for dataset in full_data:
    print(dataset.info())


# In[ ]:


# from sklearn.preprocessing import OneHotEncoder


# In[ ]:


# 【使用one-hot】
for dataset in full_data:
    # Mapping Sex
#     dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
#     title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
#     dataset['Title'] = dataset['Title'].map(title_mapping)
#     dataset['Title'] = dataset['Title'].fillna(0)
    
    # Mapping Embarked
#     dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
#     Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4


# In[ ]:


# 直接对列表内的对象修改可以，
# Feature Selection
# for dataset in full_data:
#     # 无法正确处理inteval型数据，会按每个整数处理
#     dataset = pd.get_dummies(dataset,columns=['Sex','Title','Embarked','Fare','Age']) # 直接赋值不行
# #     OneHotEncoder()
#     print (dataset.columns)

train = pd.get_dummies(train,columns=['Sex','Title','Embarked','Fare','Age'])
test = pd.get_dummies(test,columns=['Sex','Title','Embarked','Fare','Age'])


# In[ ]:


print(full_data[0].columns) # 【暂不清楚列表内部机制，故以后不推荐使用列表组合train，test】
train.columns


# In[ ]:


# cabin 直接丢弃
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',                 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)
test = test.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

print (train.columns)
print (full_data[0].columns)


# In[ ]:


train.info()


# In[ ]:


# pd.plotting.scatter_matrix(train[['']],figsize=(14,14)) # 无数值型数据


# In[ ]:


# 【数据标准化】


# In[ ]:


train = train.values # 变成ndarray
test  = test.values


# good! now we have a clean dataset and ready to predict. let's find which classifier works better on this dataset. 

# # Classifier Comparison #

# In[ ]:


import matplotlib.pyplot as plt
# Python数据可视化-seabornhttps://www.cnblogs.com/gczr/p/6767175.html
import seaborn as sns # Seaborn其实是在matplotlib的基础上进行了更高级的API封装

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss,f1_score

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


# In[ ]:


classifiers = [
    KNeighborsClassifier(n_neighbors=5,weights='distance',p=2,n_jobs=-1),
    SVC(probability=True,random_state=666,tol=1e-6),
    DecisionTreeClassifier(min_samples_split=10,random_state=666),
    RandomForestClassifier(n_estimators=500,random_state=666,n_jobs=-1),
	AdaBoostClassifier(n_estimators=500,random_state=666),
    GradientBoostingClassifier(n_estimators=500,random_state=666,min_samples_split=10),
    GaussianNB(),
    BernoulliNB(),
    LinearDiscriminantAnalysis(tol=1e-6),
    QuadraticDiscriminantAnalysis(tol=1e-6),
    LogisticRegression(tol=1e-6,random_state=666,class_weight='balanced',n_jobs=-1),
    XGBClassifier(n_estimators=500,n_jobs=-1,random_state=666)]

log_cols = ["Classifier", "Accuracy","F1"]
log 	 = pd.DataFrame(columns=log_cols)

# 等于是cross validation，分层抽样做十次crossvalidation
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

X = train[0::, 1::]
y = train[0::, 0]

acc_dict = {}
f1_dict = {}

for train_index, test_index in sss.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        f1 = f1_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc # 直接相加，没有采用平方和
        else:
            acc_dict[name] = acc
        if name in f1_dict:
            f1_dict[name] += f1
        else:
            f1_dict[name] = f1

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0 # 取所有交叉验证结果的平均值
    f1_dict[clf] = f1_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf],f1_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)


# In[ ]:


plt.figure(figsize=(12,6))
plt.subplot(121)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log[['Classifier','Accuracy']], color="b") # 使用figsize要使用plt.subplot包含

plt.subplot(122)
plt.xlabel('F1')
plt.title('Classifier F1')
sns.set_color_codes("muted")
sns.barplot(x='F1', y='Classifier', data=log[['Classifier','F1']], color="b") # 使用figsize要使用plt.subplot包含
plt.show()


# # Prediction #
# now we can use SVC classifier to predict our data.

# In[ ]:


# candidate_classifier = SVC()
# candidate_classifier.fit(train[0::, 1::], train[0::, 0])
# result = candidate_classifier.predict(test)

# 【使用Voting】
clf1 = KNeighborsClassifier(n_neighbors=5,weights='distance',p=2,n_jobs=-1)
clf2 = SVC(probability=True,random_state=666,tol=1e-6)
clf3 = DecisionTreeClassifier(min_samples_split=10,random_state=666)
clf4 = RandomForestClassifier(n_estimators=500,random_state=666,n_jobs=-1)
clf5 = AdaBoostClassifier(n_estimators=500,random_state=666)
clf6 = GradientBoostingClassifier(n_estimators=500,random_state=666,min_samples_split=10)
clf7 = BernoulliNB()
clf8 = LinearDiscriminantAnalysis(tol=1e-6)
clf9 = LogisticRegression(tol=1e-6,random_state=666,class_weight='balanced',n_jobs=-1)
clf10 = XGBClassifier(n_estimators=500,n_jobs=-1,random_state=666)

# 过拟合，考虑减少分类器数量
optimal = VotingClassifier(estimators=[('1',clf1),('4',clf4),('5',clf5),('7',clf7),                                       ('9',clf9),('10',clf10)],n_jobs=-1,voting='hard')
# optimal = optimal.fit(train.drop('Survived',axis=1),train['Survived'])
optimal = optimal.fit(train[:,1:],train[:,0])
result = optimal.predict(test)

# 【应该计算本地的CV得分】


# In[ ]:


final_file = pd.DataFrame({'PassengerId':full_data[1].PassengerId,'Survived':result.astype(int)})
final_file.to_csv('Titanic best working Classifier.csv',index=False)


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

import warnings
warnings.filterwarnings("ignore")

# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=-1, 
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"score on training set")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"score on validation set")
    
        plt.legend(loc="best")
        
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

plot_learning_curve(optimal, u"学习曲线", X, y)
# (0.8222621625725879, 0.09783507114575962)
# (0.8144932466820674, 0.05223306841711761) # 去掉三个
# (0.8252823963459736, 0.09848227190853831) # 去掉LR
# (0.8266282889044352, 0.09578503616909961) # 去掉LD

