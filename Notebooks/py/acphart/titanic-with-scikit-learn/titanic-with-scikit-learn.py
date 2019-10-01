#!/usr/bin/env python
# coding: utf-8

# # 1. 概述

# **注： 此项目为原创，但作者水平有限，如有不正确或者需要改进的地方，请不吝赐教 :)**

# ### 1.1 项目介绍
# - 我们需要先在Titanic的Data页面看一下数据集的介绍，以及对结果的要求，如果没看先回去看一下。
# - Data中有三个文件，train.csv、test.csv和gender_submission.csv，分别是训练集、测试集和结果提交示例
# - train.csv比test.csv多了一列Survived标签，我们的工作就是用训练集训练算法之后预测出测试集的Survived标签，之后再提交

# ### 1.2 工作思路
# - 对于工作流程，其实个人有个人的喜好，我还是比较喜欢Ng在机器学习课程中描述的思路：
# > ### 如果你准备研究机器学习的东西，或者构造机器学习应用程序，最好的实践方法不是建立一个非常复杂的系统，拥有多么复杂的变量；而是构建一个简单的算法，这样你可以很快地实现它。
# > - 每当我研究机器学习的问题时，我最多只会花一天的时间，就是字面意义上的 24 小时，来试图很快的把结果搞出来，即便效果不好。
# 坦白的说，就是根本没有用复杂的系统，但是只是很快的得到结果。即便运行得不完美，但是也把它运行一遍，最后通过交叉验证来检
# 验数据。
# > - 一旦做完，你可以画出学习曲线，通过画出学习曲线，以及检验误差，来找出你的算法是否有高偏差和高方差的问题，或者别的问题。
# 在这样分析之后，再来决定用更多的数据训练，或者加入更多的特征变量是否有用。
# > - 这么做的原因是：这在你刚接触机器学习问题时是一个很好的方法，你并不能提前知道你是否需要复杂的特征变量，或者你是否需要更
# 多的数据，还是别的什么。提前知道你应该做什么，是非常难的，因为你缺少证据，缺少学习曲线。因此，你很难知道你应该把时间花在什
# 么地方来提高算法的表现。但是当你实践一个非常简单即便不完美的方法时，你可以通过画出学习曲线来做出进一步的选择。你可以用这种
# 方式来避免一种电脑编程里的过早优化问题，这种理念是：
# > - **我们必须用证据来领导我们的决策，怎样分配自己的时间来优化算法，而不是仅仅凭直觉，凭直觉得出的东西一般总是错误的。**
# > - 除了画出学习曲线之外，一件非常有用的事是误差分析，通过检查交叉验证集中被错误预测的数据，你可以发现某些系统性的规律： 
# 什么类型的特征总是容易出错。 经常地这样做之后， 这个过程能启发你构造新的特征变量， 或者告诉你： 现在这个系统的短处，然后启
# 发你如何去提高它。

# - 当然，这个项目只是一个入门级的小项目，并不需要过多的优化，因为：Titanic号幸存的人虽然有一些系统性的特征，
# 但是否存活有很大的偶然性，优化过多其实意义不大，能够把预测正确率达到80%以上就可以了，进一步的学习应该转战其他有大量数据集的项目。
# - 至于排行榜中一些达到90%以上的，其特征变量的确够复杂的，有兴趣可以观摩一下。

# ### 1.3 我的流程
# - 依照Ng推荐的流程，先快速实现一个简单的预测系统，再进行分析和优化

# ### 1.4 导入数据分析相关库
# **注： 这里我设置了`IPython.core`中cell交互的方式，所以下面cell中所有单行存在的变量都可以自动输出**
# 
#    **没有设置的话，只有最后一个单行存在的变量可以自动输出**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from IPython.core.interactiveshell import InteractiveShell

# 为了输出美观一点，忽略警告，因为后面有数组除数组会出现除0的情况，但并不影响操作，可以暂时把下面这句注释掉
warnings.filterwarnings('ignore')
# 下面这句是为了让cell中所有单行存在的变量都能输出，默认情况下是只输出最后一个单行存在的变量
InteractiveShell.ast_node_interactivity = 'all'     


# ### 1.5 读入训练数据和测试数据
# - 这是第一步，不管后面要干什么，最好先看一下自己的数据是什么样子的。
# - 输出信息还是比较多的，能看出train和test数据形式是一样的，除了train的第二列是Survived标签
# - 其中还有一些数据有缺失，如Age、Cabin、Embarked，还有test里Fare列缺一个数据，这些之后需要相应的处理

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
if True:
    for data in train, test:   
        data.head()                                      # 输出前5行数据
        data.describe()                                  # 输出数值型数据的基本统计描述信息
        data.describe(include=[np.object, 'category'])   # 输出类别型数据的基本统计描述信息


# # 2. 初步数据特征探索

# - **这里我们观察每一个特征，以及这个特征与存活率的相关性**
# - **这里需要注意一点，当我们将单个特征分组之后计算每组的存活率时，要看一下各组总的个体数量，从而评估得出来的相关性会有多大程度是偶然性导致的，从而指导我们更改分组的方式**

# ### 2.1  Pclass 船票的等级
# - 从结果可以看出，船票等级是一个相当有分量的特征，等级的高低和存活率有明显的相关性

# In[ ]:


train['Pclass'].value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()


# ### 2.2 Name 姓名
# - 名字891个没有重复，输出几个名字来看一下
# - 其实一般来讲名字不能算特征，但这里面的名字还真有一些特征：
#  1. 名字中有诸如Mr、Miss、Mrs、Master等等标识乘客身份和性别的特征
#  2. 名字中的姓氏也可能隐含着乘客之间的亲属关系
#  3. 名字中其他部分含有的信息我也不了解，对外国人的名字就了解这么多了 :(
# - 不过基于我们的思路，这个特征可取也可不取，反正暂时不管它。

# In[ ]:


train['Name'].unique().size
train['Name'].head(10)


# ### 2.3  Sex
# - 输出很完美，性别果然是第一大特征

# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()


# ### 2.4 Age
# - 这里先看一下年龄的整体分布情况，以及存活者的年龄分布情况

# In[ ]:


n_bins = 40
fig, ax = plt.subplots(1, 1)
all_n, all_bins, all_patches = ax.hist(train['Age'].dropna(), bins=n_bins, color='r')
svd_n, svd_bins, svd_patches = ax.hist(train.loc[train['Survived']==1, 'Age'].dropna(), bins=n_bins, color='g')


# - 然后再看一下年龄和存活率的关系
# - 结果看起来还是有相关性的，但表现得不够完美，后面需要处理

# In[ ]:


svd_rate = svd_n/all_n
fig, ax = plt.subplots(1, 1)
_ = ax.plot(all_bins[:-1], svd_rate)


# ### 2.5 SibSp 兄弟姐妹和配偶（在船上的同辈亲属总数）
# - 分组结果显示出来这个特征和存活率还是有相关性的，但相关性的表现还不够完美，后面需要处理

# In[ ]:


train['SibSp'].value_counts()   # 各组的总人数
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()    #各组的存活率


# ### 2.6 Parch 在船上的父母和子女总数
# - 和上面一样，分组结果显示出来这个特征和存活率还是有相关性的，但相关性的表现还不够完美，后面需要处理

# In[ ]:


train['Parch'].value_counts()   # 各组的总人数
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()   # 各组的存活率


# ### 2.7 Ticket  船票编号
# - 看不懂船票编号有什么特征， 跳过  ~~~

# In[ ]:


train['Ticket'].unique().size
train['Ticket'].head(10)


# ### 2.8 Fare 票价
# - 查看各个票价区间的人数及相应存活的人数
# - 结果很明显，都不用算比率了，能看出票价越高存活率越高

# In[ ]:


n_bins = 30
fig, ax = plt.subplots(1, 1)
_ = ax.hist(train['Fare'], bins=n_bins, color='r')
_ = ax.hist(train.loc[train['Survived']==1, 'Fare'], bins=n_bins, color='g')


# ### 2.9 Cabin  船舱编号
# - 这个特征数据缺失的有点严重啊，缺了将近80%的数据
# - 测试集那边也是缺了将近80%，所以这个特征暂时不用

# In[ ]:


train['Cabin'].count()/891
test['Cabin'].count()/418


# ### 2.10 Embarked  登船港口

# - 这上船的港口和存活率也有较强的相关性，应该是这三个地方的经济情况不一样，我们可以验证一下
# - 结果如下，和猜想差不多，C港口的人平均票价和存活率明显高于另外两个地方，至于Q、S港口没有符合猜想，可能是因为Q港口的人比较少导致偶然性误差比较大

# In[ ]:


# 各个港口登船的人数及相应的存活率
train['Embarked'].value_counts()
train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()

# 看一下登船港口的平均票价
train[['Embarked', 'Fare']].groupby(['Embarked'], as_index=False).mean()


# # 3. 快速实现一个简单的预测系统

# ### 3.1 选取特征和处理数据
# - 为了快速得到结果，先选直接能用的：Pclass、Sex
# - 其中Sex是字符型数据，需要将其处理成数值型数据

# In[ ]:


# 将字符型数据转换成数值型数据，这里顺便也把测试集也转换
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1}).astype(int)
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1}).astype(int)

# 选取特征
quick_feature = ['Pclass', 'Sex']


# ### 3.2 切分训练集和测试集
# - 切分比例是train：qtest = 7:3
# - 切分前最好先打乱数据，这样可以最大程度上避免切分之后的数据有系统性的特征偏差

# In[ ]:


np.random.seed(20180818)

Qtest_n = 630
# 打乱数据
train = train.sample(frac = 1.0)
train = train.reset_index()

# 切分数据
X_train = train.loc[0:Qtest_n, quick_feature].values
y_train = train.loc[0:Qtest_n, 'Survived'].astype(int).values
X_Qtest = train.loc[Qtest_n+1:891, quick_feature].values
y_Qtest = train.loc[Qtest_n+1:891, 'Survived'].astype(int).values


# ### 3.3 训练算法并查看性能
# - 这里选两个算法：逻辑回归和支持向量机

# In[ ]:


from sklearn import linear_model, svm

lgr = linear_model.LogisticRegression()
svc = svm.SVC()

for clf in lgr, svc:
    _ = clf.fit(X_train, y_train)
    y_ = clf.predict(X_Qtest)
    # 计算准确率，其实对于二分类也可以直接用clf.score(X_Qtest, y_Qtest)得出准确率
    (y_ == y_Qtest).sum() / y_Qtest.size      


# - $\Large\color{Red}{\mbox{Amazing !}}$

# ### 感觉见证了奇迹一般，其实数据根本没有改动一下，而且只用了两个特征，正确率就达到了0.7885

# # 4. 分析结果

# ### 4.1 正确率究竟是多少
# - 虽然一下子就得到了结果，但一定要冷静，需要分析这个正确率究竟是怎样的
# - 为什么说这个呢，举个很简单的例子：
# > 假设一个识别中国人民族的算法，它不用训练，拿到数据直接输出汉族，准确率也能到92%，但这样的算法能信吗？
# - 我们先看一下如果全部预测死亡，正确率有多少

# In[ ]:


1 - y_Qtest.sum()/y_Qtest.size


# - 全蒙死亡， 预测正确率是0.6577，说明我们的学习算法还是有点料子的，接下来我们要精确评价我们的预测性能
# - 通过计算真阳性率（灵敏度）和真阴性率（特异度），看看咱们的正确率究竟几分是真的
# > - 真阳性率（True Positive Rate）：真正活着的人中预测存活的比例
# > - 真阴性率（True Negative Rate）：真正去世的人中预测死亡的比例
# - 约登指数可以评价整体的预测性能
# > - 约登（Youden）指数：真阳性率 + 真阴性率 - 1  

# In[ ]:


# 1 & 1 =1    1 | 1 = 1
# 1 & 0 =0    1 | 0 = 1
# 0 & 1 =0    0 | 1 = 1
# 0 & 0 =0    0 | 0 = 0
Qtest_svd = y_Qtest.sum()
Qtest_die = y_Qtest.size - y_Qtest.sum()
TPR = (y_ & y_Qtest).sum() / Qtest_svd
TNR = (y_Qtest.size - (y_ | y_Qtest).sum()) / Qtest_die
print("survived : {0} \t TPR : {1:.4f}".format(Qtest_svd, TPR))
print("    dead : {0} \t TNR : {1:.4f}".format(Qtest_die, TNR))
print("Yd_index : {0:.4f}".format(TPR+TNR-1))


# - 结果表明正确预测死亡能达到0.8187，正确预测生存能达到0.7303
# - 说明咱们的算法的正确率基本是真的（别被这句话绕进去了 ~~~），但是预测性能不够好
# - 比作考试的话，就是说我们的算法的分是靠自己做出来的，就是分不够高，还需努力
# - 这里还有一点就是，可是多改几次前面打乱数据时设置的随机种子，看看不同的随机切分对结果的影响，由于目前的算法还很简单，这些就到后面再弄吧。

# # 5. 下一步要做的事（步入正题）

# ### 5.1 我们快速实现的预测系统存在的问题及相应的改进方向
# - 当然，第一个就是预测正确率不高，而且灵敏度与特异度差异有点大
# - 特征变量过少，因为前面有一些明显相关的特征变量并没有选入，所以需要增加特征变量
# - 目前我们还没有关注算法训练过程，所以后面还需要检测算法训练过程及进行相应的优化
# - 不过目前的结果也提示，抓住主要特征能够快速得到一个不错的结果

# # 6. 进一步探索数据特征

# #### 在初步探索特征中，我们已经能看到Pclass、Sex、Age、SibSp、Parch、Fare和Embarked和存活率有相关，其中Pclass和Sex已经可以直接使用了
# #### 接下来就是要对剩余的几个进行处理，这里要注意的是，train数据集和test数据集要做同样的处理

# ### 6.1 补全数据
# - 在初步探索中我们已经看到了各个特征的整体情况，有几个特征变量数据有缺失，所以先要补全数据
# - 其中年龄缺失的比较多，就按已有年龄的分布进行填充，Fare和Embarked缺失数据都在个位数，就直接以均值或众数填充，Cabin暂时放弃

# In[ ]:


np.random.seed(20180818)

for data in train, test:
    # 这里去除掉NaN，并将年龄转换成整数，得到一列包含已有年龄的数组
    age_list = (data['Age'].dropna() + 0.5).astype(int)    
    
    # 从已有年龄中随机抽取相应个数的年龄
    fillage = []                                         
    for i in range(len(data['Age']) - data['Age'].count()):
        fillage.append(int(np.random.choice(age_list.values, 1)))

    # 补全数据，因为Fare和Embarked都是极个别数据缺失，就直接以均值和众数进行补全
    data['Age'][pd.isna(data['Age'])] = fillage
    data['Fare'][pd.isna(data['Fare'])] = data['Fare'].mean()
    data['Embarked'][pd.isna(data['Embarked'])] = 'S'


# ### 6.2 Age
# - 年龄数据已经补全，年龄与存活率的相关性不够显著，可以选择重新分组
# - 但是怎么分呢，分多少组呢？ 这里尝试两种分组的方式，按年龄或人数分组，分多少组就得多试几个了

# In[ ]:


# 按年龄分组
for n in range(2, 10):
    train['SortAge'] = pd.cut(train['Age'], n)
    train[['SortAge', 'Survived']].groupby(['SortAge'], as_index=False).mean()


# In[ ]:


# 按人数分组
for n in range(2, 10):
    train['SortAge'] = pd.qcut(train['Age'], n)
    train[['SortAge', 'Survived']].groupby(['SortAge'], as_index=False).mean()


# - 两种分组的结果中能看出，单纯地按人数或者年龄来分组表现得都不让人满意，所以也可以尝试一下手动分组
# - 这些分组方式都试一下，后面看看那个效果最好
# - 第一种：按年龄分，由于将近90%的人都在50岁以前，所以组数不能太多，而且需要重点关注50岁以前的组别之间的差异度，那么上面分6组表现不错
# - 第二种：按人数分，从上面结果看可以分5组
# - 第三种：手动分，从上面结果可以按年龄0-17、17-21、21-30、30-36、36-80分5组

# In[ ]:


for data in train, test:
    data['SortAge_1'] = 0
    data['SortAge_2'] = 0
    data['SortAge_3'] = 0
    # SortAge_1
    data.loc[data['Age'] <= 14, 'SortAge_1'] = 0
    data.loc[(data['Age'] > 14) & (data['Age'] <= 27), 'SortAge_1'] = 1
    data.loc[(data['Age'] > 27) & (data['Age'] <= 40), 'SortAge_1'] = 2
    data.loc[(data['Age'] > 40) & (data['Age'] <= 53), 'SortAge_1'] = 3
    data.loc[(data['Age'] > 53) & (data['Age'] <= 66), 'SortAge_1'] = 4
    data.loc[ data['Age'] > 66, 'SortAge_1'] = 5
    # SortAge_2
    data.loc[data['Age'] <= 19, 'SortAge_2'] = 0
    data.loc[(data['Age'] > 19) & (data['Age'] <= 25), 'SortAge_2'] = 1
    data.loc[(data['Age'] > 25) & (data['Age'] <= 31), 'SortAge_2'] = 2
    data.loc[(data['Age'] > 31) & (data['Age'] <= 41), 'SortAge_2'] = 3
    data.loc[ data['Age'] > 41, 'SortAge_2'] = 4
    # SortAge_3
    data.loc[data['Age'] <= 17, 'SortAge_3'] = 0
    data.loc[(data['Age'] > 17) & (data['Age'] <= 21), 'SortAge_3'] = 1
    data.loc[(data['Age'] > 21) & (data['Age'] <= 30), 'SortAge_3'] = 2
    data.loc[(data['Age'] > 30) & (data['Age'] <= 36), 'SortAge_3'] = 3
    data.loc[ data['Age'] > 36, 'SortAge_3'] = 4


# In[ ]:


train['SortAge_3'].value_counts()
train[['SortAge_3', 'Survived']].groupby(['SortAge_3'], as_index=False).mean()


# - 手动分组效果不错，各组间有明显的差异，而且各组人数也够多

# ### 6.2 SibSp、Parch
# - 这两个亲属人数特征前面已经简单看了一下和存活率的关系，我们这里再看一下

# In[ ]:


train['SibSp'].value_counts()   # 各组的总人数
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()    #各组的存活率


# In[ ]:


train['Parch'].value_counts()   # 各组的总人数
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()    #各组的存活率


# - SibSp中从各组人数及相应的存活率来看，基本可以按有、无SibSp分两组进行处理
# - Parch中从各组人数及相应的存活率来看，也可以按有、无Parch分两组进行处理
# - 这样的话其实我们也可以把这两个加起来，按有没有亲属分成两组
# - 下面都试一下，看看哪一种法子效果更好一点

# In[ ]:


train['had_SibSp'] = 0
train['had_Parch'] = 0
train.loc[train['SibSp'] >0, 'had_SibSp'] = 1
train.loc[train['Parch'] >0, 'had_Parch'] = 1

train[['had_SibSp', 'Survived']].groupby(['had_SibSp'], as_index=False).mean()
train[['had_Parch', 'Survived']].groupby(['had_Parch'], as_index=False).mean()


# In[ ]:


train['FamilySize'] = train['SibSp'] + train['Parch']
train['IsAlone'] = 0
train.loc[train['FamilySize'] >0, 'IsAlone'] = 1
train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# - 结果证明按是否有家属来分组处理效果更好一些，那么我们的测试数据也要进行相应的处理

# In[ ]:


test['FamilySize'] = test['SibSp'] + test['Parch']
test['IsAlone'] = 0
test.loc[test['FamilySize'] >0, 'IsAlone'] = 1


# ### 6.3 Fare

# - 和年龄一样，我们也是金额和人数两种分组都试一下

# In[ ]:


for n in range(2, 11):
    train['SortFare'] = pd.cut(train['Fare'], n)
    train[['SortFare', 'Survived']].groupby(['SortFare'], as_index=False).mean()


# In[ ]:


for n in range(2, 11):
    train['SortFare'] = pd.qcut(train['Fare'], n)
    train[['SortFare', 'Survived']].groupby(['SortFare'], as_index=False).mean()


# - 这里可以看出按金额分组的效果很差，按人数来分组效果还不错，其中分成2、3、4组都有不错的表现
# - 可以都试一下，然后看看那个效果好
# - test集Fare缺了一个数据，所以先补齐

# In[ ]:


test['Fare'] = test['Fare'].fillna(test['Fare'].mean())
for data in train, test:
    data['SortFare_2'] = 0
    data['SortFare_3'] = 0
    data['SortFare_4'] = 0
    # SortFare_2
    data.loc[data['Fare'] <= 14.45, 'SortFare_2'] = 0
    data.loc[ data['Fare'] > 14.45, 'SortFare_2'] = 1
    # SortFare_3
    data.loc[data['Fare'] <= 8.66, 'SortFare_3'] = 0
    data.loc[(data['Fare'] > 8.66) & (data['Fare'] <= 26), 'SortFare_3'] = 1
    data.loc[ data['Fare'] > 26, 'SortFare_3'] = 2
    # SortFare_4
    data.loc[data['Fare'] <= 7.91, 'SortFare_4'] = 0
    data.loc[(data['Fare'] > 7.91)  & (data['Fare'] <= 14.45), 'SortFare_4'] = 1
    data.loc[(data['Fare'] > 14.45) & (data['Fare'] <= 31), 'SortFare_4'] = 2
    data.loc[ data['Fare'] > 31, 'SortFare_4'] = 3


# ### 6.4 Embarked

# - Embarked数据有若干个缺失，直接用众数'S'补全即可
# - 而且在初步探索中已经看到了三组的存活率的差异，这里直接将字符转换成数值数据就可以了

# In[ ]:


train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')

train['Embarked'] = train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test['Embarked'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# ### 再次检查数据，除Cabin外数据已经齐全

# In[ ]:


for data in train, test:
    data.describe()
    data.describe(include=[np.object, 'category'])


# # 7. 进一步训练算法

# ### 7.1 使用已确定的特征
# - 上面的特征处理中，有一些已经处理完全了：Pclass、Sex、IsAlone和Embarked，先用这四个，看看和之前的性能对比
# - 打乱数据并切分就和前面的一样了，为避免随机误差，这里用和前面一样的随机种子

# In[ ]:


np.random.seed(2018082401)

# 打乱数据
train = train.sample(frac = 1.0)
train = train.reset_index()
train = train.drop(columns=['index'])

# 切分数据
train_n = 630
X_train = train.loc[0:train_n, :]
y_train = train.loc[0:train_n, 'Survived'].astype(int)
X_cv = train.loc[train_n+1:891, :]
y_cv = train.loc[train_n+1:891, 'Survived'].astype(int)

base_feature = ['Pclass', 'Sex', 'IsAlone', 'Embarked']


# ### 7.2 选择算法，并比较各个算法的正确率

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

clf_set = [ KNeighborsClassifier(),
            SVC(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis(),
            LogisticRegression()]

bf = base_feature
for clf in clf_set:
    _ = clf.fit(X_train[bf].values, y_train.values)
    print("{0:>30s} : {1:<.4f}".format(clf.__class__.__name__, clf.score(X_cv[bf].values, y_cv.values)))


# - 结果表明我们进一步的特征处理还是有一点效果的，准确率比之前有了提升

# ### 7.3 确定Age和Fare哪种分组比较适合

# In[ ]:


sortage_set = ['SortAge_1', 'SortAge_2', 'SortAge_3']
for sortage in sortage_set:
    af = ['Pclass', 'Sex', 'IsAlone', 'Embarked']
    af.append(sortage)
    print("{:>41s}".format(sortage))
    for clf in clf_set:
        _ = clf.fit(X_train[af].values, y_train.values)
        print("{0:>30s} : {1:7.4f}".format(clf.__class__.__name__, clf.score(X_cv[af].values, y_cv.values)))


# In[ ]:


sortfare_set = ['SortFare_2', 'SortFare_3', 'SortFare_4']
for sortfare in sortfare_set:
    ff = ['Pclass', 'Sex', 'IsAlone', 'Embarked']
    ff.append(sortfare)
    print("{:>41s}".format(sortfare))
    for clf in clf_set:
        _ = clf.fit(X_train[ff].values, y_train.values)
        print("{0:>30s} : {1:7.4f}".format(clf.__class__.__name__, clf.score(X_cv[ff].values, y_cv.values)))


# - 走到这其实发现数据的偶然性偏差已经对算法有影响了，虽然在这上面看到效果还不错，但是尝试着多改变几次前面切分数据的随机种子，就会发现后面的训练结果都会波动，各种分组方式并不能准确地判断优劣
# - 这也就是为什么在开始的时候说不需要在这个项目上进行过多的优化
# - 然后试了多次随机种子，在结果中还是能发现一些东西的，那就是SVC（支持向量机）的score基本都比较高

# - 然后再试一下把SortAge_1和SortFare_2同时加进去

# In[ ]:


feature = ['Pclass', 'Sex', 'IsAlone', 'Embarked', 'SortAge_1', 'SortFare_2']

for clf in clf_set:
        _ = clf.fit(X_train[ff].values, y_train.values)
        print("{0:>30s} : {1:7.4f}".format(clf.__class__.__name__, clf.score(X_cv[ff].values, y_cv.values)))


# - 最后还是发现，年龄和费用加进去并没有有效改善结果，悲伤 **:(**

# ### 最后可以选SVC算法，然后特征可以把上面的base_feature和feature都试一下
# ### 另外关于Name的特征处理可以参考一下其他的Notebook，我这里就跳过了
# ### 我这里基本没有对算法的优化，这个项目主要是入门的，算法的优化在这里不是重点，效果也不大
# ### 我的就到这里了  **：）**

# In[ ]:


clf = SVC()
_ = clf.fit(X_train[feature].values, X_train['Survived'])
y_ = clf.predict(test[feature])

