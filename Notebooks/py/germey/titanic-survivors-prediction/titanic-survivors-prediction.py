#!/usr/bin/env python
# coding: utf-8

# # 问题定义
# ## 背景
# 
# * 泰坦尼克号：英国白星航运公司下辖的一艘奥林匹克级邮轮，于1909年3月31日在爱尔兰贝尔法斯特港的哈兰德与沃尔夫造船厂动工建造，1911年5月31日下水，1912年4月2日完工试航。
# * 首航时间：1912年4月10日
# * 航线：从英国南安普敦出发，途经法国瑟堡-奥克特维尔以及爱尔兰昆士敦，驶向美国纽约。
# * 沉船：1912年4月15日（1912年4月14日23时40分左右撞击冰山）
# * 船员+乘客人数：2224
# * 遇难人数：1502（67.5%）
# 
# ## 目标
# 根据训练集中各位乘客的特征及是否获救标志的对应关系训练模型，预测测试集中的乘客是否获救。（二元分类问题）
# 
# ## 数据字典
# ### 基础字段
# * PassengerId 乘客id
#     * 训练集891（1- 891），测试集418（892 - 1309）
# * Survived 是否获救
#     * 1=是，2=不是
#     * 获救：38%
#     * 遇难：62%（实际遇难比例：67.5%）
# * Pclass 船票级别
#     * 代表社会经济地位。 1=高级，2=中级，3=低级
#     * 1 : 2 : 3 = 0.24 : 0.21 : 0.55
# * Name 姓名
#     * 示例：Futrelle, Mrs. Jacques Heath (Lily May Peel)
#     * 示例：Heikkinen, Miss. Laina
# * Sex 性别
#     * male 男 577，female 女 314
#     * 男 : 女 = 0.65 : 0.35
# * Age 年龄（缺少20%数据）
#     * 训练集：714/891 = 80%
#     * 测试集：332/418 = 79%
# * SibSp 同行的兄弟姐妹或配偶总数
#     * 68%无，23%有1个 … 最多8个
# * Parch 同行的父母或孩子总数
#     * 76%无，13%有1个，9%有2个 … 最多6个
#     * Some children travelled only with a nanny, therefore parch=0 for them.
# * Ticket 票号（格式不统一）
#     * 示例：A/5 21171
#     * 示例：STON/O2. 3101282
# * Fare 票价
#     * 测试集缺一个数据
# * Cabin 船舱号
#     * 训练集只有204条数据，测试集有91条数据
#     * 示例：C85
# * Embarked 登船港口
#     * C = Cherbourg（瑟堡）19%, Q = Queenstown（皇后镇）9%, S = Southampton（南安普敦）72%
#     * 训练集少两个数据
# 
# ### 衍生字段（部分，在后续代码中补充）
# * Title 称谓
#     * dataset.Name.str.extract( “ ([A-Za-z]+)\.”, expand = False)
#     * 从姓名中提取，与姓名和社会地位相关
# * FamilySize 家庭规模
#     * Parch + SibSp + 1
#     * 用于计算是否独自出行IsAlone特征的中间特征，暂且保留
# * IsAlone 独自一人
#     * FamilySize == 1
#     * 是否独自出行
# * HasCabin 有独立舱室
#     * 不确定没CabinId的样本是没有舱室还是数据确实

# # 备注
# 
# > [Kernel参考：Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 
# ## 工作流7个步骤
# 
# 1. 问题定义Question or problem definition.
# 1.  获取训练和测试数据Acquire training and testing data.
# 1. 数据准备和清洗Wrangle, prepare, cleanse the data.
# 1. 分析，识别数据模型，探索数据Analyze, identify patterns, and explore the data.
# 1. 建模，预测，解决问题Model, predict and solve the problem.
# 1. 可视化，报表，展示解决步骤和最终解决方案Visualize, report, and present the problem solving steps and final solution.
# 1. 提交结果Supply or submit the results.
# 
# ## 特征工程
# 
# * Classifying：样本分类或分级
# * Correlating：样本预测结果和特征的关联程度，特征之间的关联程度
# * Converting：特征转换（向量化）
# * Completing：特征缺失值预估完善
# * Correcting：对于明显离群或会造成预测结果明显倾斜的异常数据，进行修正或排除
# * Creating：根据现有特征衍生新的特征，以满足关联性、向量化以及完整度等目标上的要求
# * Charting：根据数据性质和问题目标选择正确的可视化图表
# 

# # 1.获取数据
# * 导入机器学习模型的过程也可以放到模型拟合前

# In[ ]:


# 导入库
# 数据分析和探索
import pandas as pd
import numpy as np
import random as rnd

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# 获取数据，训练集train_df，测试集test_df，合并集合combine（便于对特征进行处理时统一处理：for df in combine:）
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df, test_df]


# In[ ]:


train_df


# # 2.探索数据

# ## 2.1 特征基本信息（head，info，describe）

# In[ ]:


# 探索数据
# 查看字段结构、类型及head示例
train_df.head(10)


# In[ ]:


# 查看各特征非空样本量及字段类型
train_df.info()
print("_"*40)
test_df.info()


# In[ ]:


# 查看数值类（int，float）特征的数据分布情况
train_df.describe()


# In[ ]:


# 查看非数值类（object类型）特征的数据分布情况
train_df.describe(include=["O"])


# ## 2.2 几个枚举型特征与Survived的关联性（直接group汇总求均值）

# In[ ]:


train_df[["Pclass","Survived"]].groupby(["Pclass"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 富人和中等阶层有更高的生还率，底层生还率低


# In[ ]:


train_df[["Sex","Survived"]].groupby(["Sex"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 性别和是否生还强相关，女性用户的生还率明显高于男性


# In[ ]:


train_df[["SibSp","Survived"]].groupby(["SibSp"],as_index=False).mean().sort_values(by="Survived",ascending=False)
# 有0到2个兄弟姐妹或配偶的生还几率会高于有更多的


# In[ ]:


train_df[["Parch","Survived"]].groupby(["Parch"],as_index = False).mean().sort_values(by="Survived",ascending=False)
# 同行的父母或孩子总数相关


# ## 2.3 对年龄这类跨度较长的特征使用直方图分别查看生还与否的分布

# In[ ]:


g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Age",bins=20)
# 婴幼儿的生存几率更大


# In[ ]:


# Fare
g = sns.FacetGrid(train_df,col="Survived")
g.map(plt.hist,"Fare",bins=10)
# 票价最便宜的幸存几率低


# ## 2.4 其他特征可视化探索
# * pointplot不太理解，待细看

# In[ ]:


grid = sns.FacetGrid(train_df,row="Survived",col="Sex",aspect=1.6)
grid.map(plt.hist,"Age",alpha=.5,bins=20)
grid.add_legend()
# 女性的幸存率更高，各年龄段均高于50%
# 男性中只有婴幼儿幸存率高于50%，年龄最大的男性（近80岁）幸存


# In[ ]:


grid1 = sns.FacetGrid(train_df,col="Embarked")
grid1.map(sns.pointplot,"Pclass","Survived","Sex",palette = "deep")
#


# In[ ]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


# # 3.特征清洗

# ## 3.1 NameLength
# * 从别的kernel看到的，不确定效果

# In[ ]:


# Some features of my own that I have added in
# Gives the length of the name
train_df['NameLength'] = train_df['Name'].apply(len)
test_df['NameLength'] = test_df['Name'].apply(len)


# In[ ]:


train_df


# ## 3.2 HasCabin

# In[ ]:


# Feature that tells whether a passenger had a cabin on the Titanic
train_df['HasCabin'] = train_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test_df['HasCabin'] = test_df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


# In[ ]:


train_df


# * 剔除Ticket（人为判断无关联）和Cabin（有效数据太少）两个特征

# In[ ]:


# 剔除Ticket（人为判断无关联）和Cabin（有效数据太少）两个特征
train_df = train_df.drop(["Ticket","Cabin"],axis=1)
test_df = test_df.drop(["Ticket","Cabin"],axis=1)
combine = [train_df,test_df]
print(train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)


# ## 3.3 Title

# In[ ]:


# 根据姓名创建称号特征，会包含性别和阶层信息
# dataset.Name.str.extract(' ([A-Za-z]+)\.' -> 把空格开头.结尾的字符串抽取出来
# 和性别匹配，看各类称号分别属于男or女，方便后续归类

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex']).sort_values(by=["male","female"],ascending=False)


# In[ ]:


# 把称号归类为Mr,Miss,Mrs,Master,Rare_Male,Rare_Female(按男性和女性区分了Rare)
for dataset in combine:
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess', 'Dona'],"Rare_Female")
    dataset["Title"] = dataset["Title"].replace(['Capt', 'Col','Don','Dr','Major',
                                                 'Rev','Sir','Jonkheer',],"Rare_Male")
    dataset["Title"] = dataset["Title"].replace('Mlle', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Miss')


# In[ ]:


dataset


# In[ ]:


# 按Title汇总计算Survived均值，查看相关性
train_df[["Title","Survived"]].groupby(["Title"],as_index=False).mean()


# In[ ]:


# title特征映射为数值
title_mapping = {"Mr":1,"Miss":2,"Mrs":3,"Master":4,"Rare_Female":5,"Rare_Male":6}
for dataset in combine:
    dataset["Title"] = dataset["Title"].map(title_mapping)
    dataset["Title"] = dataset["Title"].fillna(0)
    # 为了避免有空数据的常规操作
train_df.head()


# In[ ]:


# Name字段可以剔除了
# 训练集的PassengerId字段仅为自增字段，与预测无关，可剔除
train_df = train_df.drop(["Name","PassengerId"],axis=1)
test_df = test_df.drop(["Name"],axis=1)


# In[ ]:


# 每次删除特征时都要重新combine
combine = [train_df,test_df]
combine[0].shape,combine[1].shape


# ## 3.4 Sex

# In[ ]:


# sex特征映射为数值
for dataset in combine:
    dataset["Sex"] = dataset["Sex"].map({"female":1,"male":0}).astype(int)
    # 后面加astype(int)是为了避免处理为布尔型？
train_df.head()


# ## 3.5 Age

# In[ ]:


# 对Age字段的空值进行预测补充
# 取相同Pclass和Title的年龄中位数进行补充（Demo为Pclass和Sex）

grid = sns.FacetGrid(train_df,col="Pclass",row="Title")
grid.map(plt.hist,"Age",bins=20)


# In[ ]:


guess_ages = np.zeros((6,3))
guess_ages


# In[ ]:


# 给age年龄字段的空值填充估值
# 使用相同Pclass和Title的Age中位数来替代（对于中位数为空的组合，使用Title整体的中位数来替代）


for dataset in combine:
    # 取6种组合的中位数
    for i in range(0, 6):
        
        for j in range(0, 3):
            guess_title_df = dataset[dataset["Title"]==i+1]["Age"].dropna()
            
            guess_df = dataset[(dataset['Title'] == i+1) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median() if ~np.isnan(guess_df.median()) else guess_title_df.median()
            #print(i,j,guess_df.median(),guess_title_df.median(),age_guess)
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
    # 给满足6中情况的Age字段赋值
    for i in range(0, 6):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Title == i+1) & (dataset.Pclass == j+1),
                        'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# ## 3.5 IsChildren

# In[ ]:


#创建是否儿童特征
for dataset in combine:
    dataset.loc[dataset["Age"] > 12,"IsChildren"] = 0
    dataset.loc[dataset["Age"] <= 12,"IsChildren"] = 1
train_df.head()


# In[ ]:


# 创建年龄区间特征
# pd.cut是按值的大小均匀切分，每组值区间大小相同，但样本数可能不一致
# pd.qcut是按照样本在值上的分布频率切分，每组样本数相同
train_df["AgeBand"] = pd.qcut(train_df["Age"],8)
train_df[["AgeBand","Survived"]].groupby(["AgeBand"],as_index = False).mean().sort_values(by="AgeBand",ascending=True)


# In[ ]:


# 把年龄按区间标准化为0到4
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 17, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 21), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 21) & (dataset['Age'] <= 25), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 26), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 31), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 36.5), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 36.5) & (dataset['Age'] <= 45), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 7
train_df.head()


# In[ ]:


# 移除AgeBand特征
train_df = train_df.drop(["AgeBand"],axis=1)
combine = [train_df,test_df]
train_df.head()


# ## 3.6 FamilySize

# In[ ]:


# 创建家庭规模FamilySize组合特征
for dataset in combine:
    dataset["FamilySize"] = dataset["Parch"] + dataset["SibSp"] + 1
train_df[["FamilySize","Survived"]].groupby(["FamilySize"],as_index = False).mean().sort_values(by="FamilySize",ascending=True)


# ## 3.7 IsAlone

# In[ ]:


# 创建是否独自一人IsAlone特征
for dataset in combine:
    dataset["IsAlone"] = 0
    dataset.loc[dataset["FamilySize"] == 1,"IsAlone"] = 1
train_df[["IsAlone","Survived"]].groupby(["IsAlone"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# 移除Parch,Sibsp,FamilySize（暂且保留试试）
# 给字段赋值可以在combine中循环操作，删除字段不可以，需要对指定的df进行操作
train_df = train_df.drop(["Parch","SibSp"],axis=1)
test_df = test_df.drop(["Parch","SibSp"],axis=1)
combine = [train_df,test_df]
train_df.head()


# In[ ]:


# 创建年龄*级别Age*Pclass特征
# 这个有啥意义？
#for dataset in combine:
#    dataset["Age*Pclass"] = dataset["Age"] * dataset["Pclass"]
#train_df.loc[:,["Age*Pclass","Age","Pclass"]].head()


# ## 3.8 Embarked

# In[ ]:


# 给Embarked补充空值
# 获取上船最多的港口
freq_port = train_df["Embarked"].dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port)
train_df[["Embarked","Survived"]].groupby(["Embarked"],as_index=False).mean().sort_values(by="Survived",ascending=False)


# In[ ]:


# 把Embarked数字化
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].map({"S":0,"C":1,"Q":2}).astype(int)
train_df.head()


# In[ ]:


# 去掉Embarked试试。。
#train_df = train_df.drop(["Embarked"],axis=1)
#test_df = test_df.drop(["Embarked"],axis=1)
#combine=[train_df,test_df]
#train_df.head()


# ## 3.9 Fare

# In[ ]:


# 给测试集中的Fare填充空值，使用中位数
test_df["Fare"].fillna(test_df["Fare"].dropna().median(),inplace=True)
test_df.info()


# In[ ]:


# 创建FareBand区间特征
train_df["FareBand"] = pd.qcut(train_df["Fare"],4)
train_df[["FareBand","Survived"]].groupby(["FareBand"],as_index=False).mean().sort_values(by="FareBand",ascending=True)


# In[ ]:


# 根据FareBand将Fare特征转换为序数值
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# ## 3.10 特征工程完成
# * Pclass特征不需要特殊处理

# In[ ]:


test_df.head(10)


# ## 3.11 特征相关性可视化

# In[ ]:


# 用seaborn的heatmap对特征之间的相关性进行可视化
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# In[ ]:


# 用seaborn的pairplot看各特征组合的样本分布
g = sns.pairplot(train_df[[u'Survived', u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked',
       u'FamilySize', u'Title', u'IsChildren', u'IsAlone', u'HasCabin',u'NameLength']], 
                 hue='Survived', palette = 'seismic',size=1.2,diag_kind = 'kde',
                 diag_kws=dict(shade=True),plot_kws=dict(s=10) )
g.set(xticklabels=[])
# 有点浮夸，需要指点


# ## 3.12 训练集和测试集准备

# In[ ]:


X_train = train_df.drop("Survived",axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId",axis=1).copy()
X_train.shape,Y_train.shape,X_test.shape


# # 4.建模和优化

# ## 4.1 模型比较
# ### 4.1.1 逻辑回归

# In[ ]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred_logreg = logreg.predict(X_test)
acc_log = round(logreg.score(X_train,Y_train)*100,2)
acc_log


# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# ### 4.1.2 SVC

# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ### 4.1.3 KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ### 4.1.4 Naive Bayes

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ### 4.1.5 Perceptron 感知器

# In[ ]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_perceptron = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# ### 4.1.6 Linear SVC

# In[ ]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# ### 4.1.7 SGD

# In[ ]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# ### 4.1.8 决策树

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ### 4.1.9 随机森林
# * 包含拆分验证集及Kfold方法

# In[ ]:


from sklearn.model_selection import train_test_split

X_all = train_df.drop(['Survived'], axis=1)
y_all = train_df['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)


# In[ ]:


# Random Forest
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
random_forest = RandomForestClassifier()

parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }
acc_scorer = make_scorer(accuracy_score)
grid_obj = GridSearchCV(random_forest, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)
clf = grid_obj.best_estimator_
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
acc_random_forest_split=accuracy_score(y_test, pred)
acc_random_forest_split


#random_forest.fit(X_train, Y_train)
#Y_pred_random_forest = random_forest.predict(X_test)
#random_forest.score(X_train, Y_train)
#acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
#acc_random_forest


# In[ ]:


from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)


# In[ ]:


Y_pred_random_forest_split = clf.predict(test_df.drop("PassengerId",axis=1))


# In[ ]:


#from sklearn.cross_validation import KFold

#def run_kfold(clf):
#    kf = KFold(891, n_folds=10)
#    outcomes = []
#    fold = 0
#    for train_index, test_index in kf:
#        fold += 1
#        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
#        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
#        clf.fit(X_train, y_train)
#        predictions = clf.predict(X_test)
#        accuracy = accuracy_score(y_test, predictions)
#        outcomes.append(accuracy)
#        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
#    mean_outcome = np.mean(outcomes)
#    print("Mean Accuracy: {0}".format(mean_outcome)) 

#run_kfold(clf)


# ## 4.2 模型效果比较

# In[ ]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, 
              acc_log, 
              acc_random_forest_split,
              #acc_random_forest,
              acc_gaussian, 
              acc_perceptron, 
              acc_sgd, 
              acc_linear_svc, 
              acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# ## 提交结果

# In[ ]:


import time
print(time.strftime('%Y%m%d%H%M',time.localtime(time.time())))


# In[ ]:


# 取最后更新的随机森林模型的预测数据进行提交

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_random_forest_split
        #"Survived": Y_pred_random_forest
    })
submission.to_csv('submission_random_forest_'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_decision_tree
    })
submission.to_csv('submission_decision_tree'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_knn
    })
submission.to_csv('submission_knn_'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)


# In[ ]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred_svc
    })
submission.to_csv('submission_svc_'
                  +time.strftime('%Y%m%d%H%M',time.localtime(time.time()))
                  +".csv", 
                  index=False)

