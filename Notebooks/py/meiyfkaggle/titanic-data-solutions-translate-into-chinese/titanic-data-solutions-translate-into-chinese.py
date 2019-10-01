#!/usr/bin/env python
# coding: utf-8

# 引言:与君共勉，不当之处还望指出。附原文地址，[传送门](https://www.kaggle.com/startupsci/titanic/titanic-data-science-solutions)。

# ##泰坦尼克号数据科学解决方案
# 这个notebook是[数据科学解决方案](https://startupsci.com/)的指南。将带领我们通过传统的工作流，在像kaggle这样的网站上解决数据科学竞赛。
# 有很多优秀的notebooks来研究数据科学竞赛项目。但是因为这些notebooks都是专家为了专业人士所编写的，所以他们中的很多人都跳过了对有关如何形成解决方案的解释。本文的目的就是通过一个循序渐进的工作流，来解释我们在解决方案的开发中所执行的每一步和采用每一个决定的基本原理。
# 
# ##工作流阶段
# 在数据科学解决方案这本书中，竞赛解决方案工作流被划分成7个阶段。
# 
# - 1.问题或者难题的定义
# - 2.采集训练集和测试集
# - 3.数据规整，预处理，清洗数据
# - 4.分析，识别模式和数据挖掘
# - 5.模型，预测和解决问题
# - 6.可视化，报告，介绍当前解决问题的步骤以及最终的解决方案
# - 7.供应或者提交结果
# 
# 工作流显示了每个阶段如何跟着另一个阶段的一般顺序。但是有例外的情况。
# 
# - 我们可以整合多个工作流阶段。如我们可以通过可视化数据进行分析。
# - 比显示的更早的执行某个阶段。如我们可以在数据规整前后进行数据分析。
# - 在我们的工作流中多次执行某个阶段。如可视化阶段可能会被多次使用。
# - 完全放弃某个阶段。如作为竞赛，我们可能不需要供应阶段使我们的数据集来生产或服务。
# 
# ##问题和难题定义
# 像kaggle这样的竞赛网站定义了要解决的问题的，同时提供数据集来训练你的数据模型，并根据测试集测试模型结果。泰坦尼克号救援竞赛的问题就[被描述在Kaggle上](https://www.kaggle.com/c/titanic)。
# 
# 从列出了在泰坦尼克号灾难中幸存下来或没有生存的乘客的一系列训练样本中学习后，我们的模型能不能通过一个给定的不包含获救信息的测试集，预测测试集中的乘客是否获救。
# 
# 我们也想要早日理解我们问题的领域。[Kaggle 竞赛描述页](https://www.kaggle.com/c/titanic)。以下是要强调注意的。
# 
# - 1912年4月15日，泰坦尼克号在首次航行中，与冰山相撞后沉没，在2224名乘客和船员中造成1502人死亡。也就是只有32%的存活率。
# - 其中一个导致了在这场海难中损失如此多生命的原因是乘客和船员没有足够多的救生船。
# - 尽管在这次沉船事件中有一些运气的原因，但是一些像女人，孩子，上层人士这类人比其他人幸存的概率更大。
# 
# ##工作流目标
# 数据科学的解决方案工作流程解决了7个目标。
# 
# **分类.** 我们可能想要对我们的样本进行分类。我们也可能想要了解不同类别对我们的解决方案的影响(implication)和相关性。
# 
# **关联.**可以根据训练数据集中的可用特征来解决问题。数据集中的哪些特征大大有助于我们的解决方案目标？从统计学(statistically)上来说，特征和解决方案目标之间有没有相关性？当特征值改变的时候，解决方案的状态也改变了，反过来呢？这可以通过在给定的数据集中对数值型和分类特征进行测试。我们也想要确定后续目标和工作流程阶段的生存之外的特征之间的相关性。关联一定的特征可能有助于创建，完成或修正特征 。
# 
# **转化.** 建模阶段，需要对数据进行预处理。依赖于选择的模型算法，还需要把所有的特征转化成数值等效值。比如文本类型的值转化成数值类型的值。
# 
# **完成.** 数据预处理可能需要我们评估某个特征所有的缺失值。模型算法在没有缺失值的时候可能工作的更好。
# 
# **修正.** 我们还可以分析给定的训练数据集中特征的错误或可能不准确的值，并尝试更正这些值或排除包含错误的样本。一个方式是检测所有我们样本或特征中的异常值。如果它对数据分析没一点用，或者可能明显导致结果偏差很大的话，我们也可以放弃某个特征。
# 
# **创建.** 我们可以基于某个已经存在当特征或者一系列特征来创造出一个新的特征，这样新的特征就继承了其相关性，转换和完整性目标。
# 
# **制图.** 怎么选择正确的可视化图和制图依赖数据的性质和解决方案的目的。一个好的开始就是去阅读论文[Which chart or graph is right for you ?](http://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you#ERAcoH5sEG5CFlek.99)
# 
# ##2017年1月29日重构版本
# 我们基于(a)收到的读者的评论，要重构notebooo,(b)将notebook从Jupyter(2.7)内核移植Kaggle内核(3.5)的问题,并且(c)审查一些最佳实践的核心函数。
# 
# ##用户评论
# - 整合训练和测试集的一些操作，比如通过数据集把标题转换成数值类型的值。
# - 修正观点 - 船上几乎 30%的乘客是有兄弟姐妹和/或配偶的。
# - 修正理解逻辑回归系数
# 
# ##移植的问题
# - 指定图的维度，把图例加到图中
# 
# ##最佳实践
# - 在项目中更早的执行特征相关性分析
# - 用多种多样的图图例而不是覆盖可读性

# In[ ]:


# 数据分析和数据规整
import pandas as pd
import numpy as np
import random as rd

# 可视化
import seaborn as sns
import matplotlib.pyplot as plt
# 不跳出页面,在notebook中展示图表
get_ipython().magic(u'matplotlib inline')

# 机器学习算法
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# ##数据采集
# python的pandas包帮助我们处理我们的数据。我们从把训练集和测试集采集成Pandas的DataFrame开始。我们也把这些数据集整合一起执行某些操作。

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine = [train_df,test_df]


# ##通过数据描述分析
# Pandas也有助于描述数据集回答以下早在我们项目中的问题。
# **数据集中的哪些特征是有用的？**
# 注意直接操作或分析特征的名字来。这些特征名字的描述在[Kaggle的data页面](https://www.kaggle.com/c/titanic/data)

# In[ ]:


# 展示数据集中所有的列
print(train_df.columns.values)


# ##哪些特征是分类的？
# 这些值把样本分成相似样本的集合。分类特征是以名义值(nominal)，序数(ordinal)，比例(ratio)或间隔值(interval)为基础的吗？ 除此之外，这有助于我们选择适当的可视化绘图。
# 
# - 分类(categorical):Servived,Sex,Embarked。序数(ordinal):Pclass。
# 
# ##哪些特征是数值型的？
# 这些值是从不同样本变化而来的。数值型特征是以离散，连续或时间序列为基础的值吗？除此之外，这有助于我们选择适当的可视化绘图。
# 
# - 连续(continous):Age,Fare。离散的(discrete):SibSp,Parch。

# In[ ]:


# 预览前3行数据,没有参数默认前五行
train_df.head(3)


# **哪个特征是混合数据类型的？**
# 数值型的(numerical)和字母数字混合编制的(alphanumeric)数据在相同的特征中。这些就是需要修正的目标。
# 
# - Ticket 是数字和字母数字混合编制的数据类型。Cabin是字母数字混合编制的。
# 
# **哪一个特征是含有错误和打字错误的？**
# 这对大数据集来说是很难审查的，然而从一个小数据集中审查出一小部分需要修正的样本是很简单的事情。
# 
# - 特征Name可能是包含错误和打字错误的，因为有很多方式被用来描述一个名字，包括用于替代名称或缩写的称号，圆括号(round brackets)和引号(quotes)。

# In[ ]:


# 预览最后5行数据
train_df.tail()


# ##哪一个特征含有空白，null或空值?
# 这些也是需要修正的。
# 
# - Cabin>Age>Embarked 含有一些null值的特征在训练集中的顺序。
# - Cabin>Age 在测试集中是不完整的。
# 
# ##各种特征的数据类型是什么?
# 这在转化目标中会帮助我们。
# 
# - 7个特征是整数或者浮点数的，测试集中则是6个
# - 5个特征是字符串类型的

# In[ ]:


train_df.info()
print('_'*40)
test_df.info()


# ##在样本中，数值型的特征扮演了什么角色?
# 
# 除了早期的见解之外，这有助于我们确定，实际问题领域的训练集的代表性如何。
# 
# 总样本数是891或者说在泰坦尼克号上的实际的乘客数的40%。
# 
#  - Survived是一个只有0和1值的分类特征。
#  - 大约38%的样本存活代表了32%的实际存活率。
#  - 绝对多数(75%)的乘客是没有携带子女或者父母的。
#  - 船上30%的乘客是有兄弟姐妹(sibling)和/或配偶(spouse)的。
#  - 票价几乎没有乘客之间存在着显著的差异,因为只有1%的人支付了高达512美元。
#  - 上了年纪的乘客也很少(65-80之间的只占了1%)

# In[ ]:


train_df.describe()
# 使用百分数来评估存活率,正如我们的问题描述中提到的38％的存活率。
# 使用百分数来查看Parch的分布情况
# SibSp分布情况
# 年龄和票价[.1，.2，.3，.4，.5，.6，.7，.8，.9，.99]


# ##分类特征扮演了什么角色?
# 
# - Names 在数据集中是唯一的(891个没有重名的)
# - Sex 变量只有两个可能的值，其中65%的是男性(最多的是男性，频率：577/891)
# - Cabin值有些是重复(duplicates)的在样本中。也可以(alternatively)说几个乘客共享一个船舱(cabin)。
# - Ticket特征有很多重复的值，比率达到了22%(不同的只有681个)。

# In[ ]:


train_df.describe(include=['O'])


# ##基于数据分析的猜想
# 到目前为止， 完成基于数据分析的工作，我们会得到如下猜想。在采取适当的行动之前，我们需要进一步的验证这猜想。
# 
# ##关联
# 我们想要知道每一个特征是如何与存活率关联的。我们希望在项目初期就这样做，并快速将这些相关性与项目后期的建模相关性进行匹配。
# 
# ##完成
# - 因为特征Age肯定和幸存者相关，所以我们想要完成Age特征。
# - 因为特征Embarked和幸存者或者其他重要的特征相关，所以我们想要完成Embarked特征。
# 
# ##修正
# - 因为有很高的重复比例，而且Ticket和幸存者之间也没什么相关性，所以特征Ticket可以从我们的数据分析中被丢弃。
# - 因为在训练集和测试集中，特征Cabin都有大量不正确或者空值，所以我们也可以丢弃这个特征。
# - PassengerId也可以从训练集中丢弃，因为它对幸存者没有任何影响。
# - 特征Name相对是不标准(non-standard)的，可能对结果也没什么直接(directly)影响，所以也丢弃它。
# 
# ##创造
# - 我们想要创造一个基于Parch和SibSp的Family特征来统计船上家庭成员的总数。
# - 我们想要设计一个Name特征来提取Title然后形成一个新的特征。
# - 我们想要为了年龄段创造一个新的特征。这样就把一个连续(continous)的数值型特征转变成了顺序(ordinal)的分类特征。
# 
# ##分类
# 基于这个问题之前的描述，我们也可以增加进我们的猜想。
# - 女人(Sex=female)更可能获救。
# - 儿童(Age<?)更可能获救。
# - 上层人士(Pclass=1)更可能获救。
# 
# ##通过旋转特征分析
# 为了验证我们的观察和擦想，我们可以通过旋转我们的特征来快速分析特征之间的相关性以互相验证。我们只能在这些特征并没有任何空值的情况下这么做。对于分类（性别），序数（Pclass）或离散（SibSp，Parch）类型的特征，这样做也是有道理的
# 
# - Pclas 我们观察到Pclass=1和Servived（classifying #3）是正相关(>0.5)的。我们决定把这个特征加入到我们的模型中。
# - Sex 我们在问题定义中证实了我们的观察，Sex=femall有高达74%的存活率（classifying #1）。
# - SibSp and Parch 因为某些价值，这些特征是零相关的。最好从这些独特的特征（creating #1）导出一个特征或一组特征。

# In[ ]:


train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


train_df[['Sex','Survived']].groupby(['Sex'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[ ]:


train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# ##通过可视化数据进行分析
# 现在我们可以通过使用可视化工具分析数据来继续证实我们的一些猜想了。
# 
# ##数值特征的相关性
# 让我们开始理清数值型特征和我们的解决方案目标(Survived)之间的相关性吧。
# 
# 直方图对于Age这样的连续数值型变量的波段(band)或范围，用图识别(identify)是很有用的。直方图可以使用自动定义的仓或同等范围的波段来表明样本的分布。 这有助于我们回答与特定波段相关的问题（婴儿(infants)有更好的生存率吗？）
# 
# 注意直方图可视化中的y轴代表的是样本或者乘客的数量。
# 
# ##观察结果
# - 婴儿(Age<4)有较高的存活率。
# - 老年人(Age=80)有较高的存活率。
# - 大量15-25岁的人没有存活。
# - 绝大多数的人在15-35岁之间。
# 
# ## 决策
# 因为后续工作流阶段的决策，这个简单的分析就能证实我们的猜想。
# 
# - 我们应该在我们模型的训练中考虑到特征Age(我们的猜想 classifying#2)。
# - 完成特征Age，使其没有空值(completing #1)。
# - 我们应该设定年龄波段(creating #3)。

# In[ ]:


g = sns.FacetGrid(train_df,col='Survived')
g.map(plt.hist,'Age',bins=20)


# ##数值和序数特征的相关性
# 我们可以整合多个特征，使用单个图来识别相关性。这可以用具有数值的数字和分类特征来完成。
# 
# ##观察结果
# - 有大量Pclass=3的乘客，然而大部分都没有存活下来。证实了我们的classifying猜想#2。
# - Pclass=2和Pclass=3的婴儿更容易存活。进一步证实了我们的classifying猜想#2。
# - 大量Pclass=1的人存活了下来。证实了我们的classifying猜想#3。
# 
# ##决策
# - 模型训练中考虑Pclass特征。

# In[ ]:


# grid = sns.FacetGrid(train_df,col='Pclass',hue='Survived')
grid = sns.FacetGrid(train_df,col='Survived',row='Pclass',size=2.2,aspect=1.6)
grid.map(plt.hist,'Age',alpha=.5,bins=20)
grid.add_legend();


# ##分类特征的相关性
# 现在我们可以关联分类特征和我们的解决方案目标了。
# 
# ##观察结果
# - 女性乘客比男性更容易存活。证实了classifying(#1)。
# - 意外的是Embarked=C的男性有更高的存活比例。这可能说明了Pclass与Embarked之间和Pclass与Survived之间存在某种联系，而Embarked和Survived之间的联系则不是必要的。
# - C和Q港口的Pclass=2的男性Pclass=3的男性有更高的存活率。Completing(#2)。
# - 在男性乘客中，乘船的港口改变(varying)了Pclass=3的乘客的存活率(#1)。
# 
# ##决策
# - 在模型训练中考虑特征Sex。
# - 完成并增加Embarked特征来训练模型。

# In[ ]:


# grid = sns.FacetGrid(train_df,col='Embarked')
grid = sns.FacetGrid(train_df,row='Embarked',size=2.2,aspect=1.6)
grid.map(sns.pointplot,'Pclass','Survived','Sex',palette='deep')
grid.add_legend()


# ##分类和数值型特征的相关性
# 我们也想要关联分类特征(非数字的)和数值型特征。我们可以考虑Embarked(分类的非数字的)，Sex(分类的非数字的),Fare(连续数字)和Survived(分类的数字的)之间的相关性。
# 
# ##观察结果
# - 支付票价很高的乘客有更高的存活率。证实了我们creating(#4)票价范围的猜想。
# - 乘船的港口和存活率是相关的。证实了correlating(#1)和completing(#2)。
# 
# ##决策
# - 考虑特征Fare的波段。

# In[ ]:


# grid =sns.FacetGrid(train_df,col='Embarked',hue='Survived',palette={0:'k',1:'w'})
grid = sns.FacetGrid(train_df,row='Embarked',col='Survived',size=2.2,aspect=1.6)
grid.map(sns.barplot,'Sex','Fare',alpha=.5,ci=None)
grid.add_legend()


# ##数据规整
# 我们收集了数个有关我们数据集和解决方案所必要的猜想和决策。到目前为止，我们没有必要改表任何一个特征或者特征值来达成这些目的。现在为了修正，创造和完成目标,让我们执行我们的决策和猜想。
# 
# ##通过丢弃特征来修正
# 这是一个很好的起始目标。通过丢弃特征，我们只处理了很少的数据点。却让notebook加速以及简化(eases)了分析。
# 
# 根据我们的猜想和决策，我们想要丢弃特征Cabin(correcting #2)和Ticket(correcting #1)。
# 
# 注意我们要同时在合适的地方执行操作以确保训练集和测试集始终保持一致。

# In[ ]:


print('Before',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape)
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df,test_df]

'After',train_df.shape,test_df.shape,combine[0].shape,combine[1].shape


# ##从存在的特征中提出并创造一个新特征
# 我们想分析特征Name是否可以设计提取成Title特征以及测试Title特征和生存之间的关系,之前把特征Name和PassengerId丢弃了。
# 
# 在下面的代码中我们使用正则表达式提取Title特征。正则表达式(\w+\.)匹配在Name特征中以点字符结尾的第一个单词。expand=False这个标记返回一个DataFrame。
# 
# ##观察结果
# 当我们绘制Title,Age,Survive时，我们注意到了一下观察结果。
# 
# - 大多数标题绑定年龄组准确。例如：Master标题的Age是5岁。
# - 标题在年龄段之间的存活率略有不同。
# - 某些标题绝大部分都存活了下来(Mme,Lady,Sir),而有的就没有(Don,Rev,Jonkheer)。
# 
# ##决策
# - 我们决定在模型训练中保留新的特征Title

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+\.)',expand=False)
pd.crosstab(train_df['Title'],train_df['Sex'])


# 我们可以用更常用的名称替换许多titles或者将它们分类为更稀有的。

# In[ ]:


for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady','Countess','Capt','Col',
                                                'Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

train_df[['Title','Survived']].groupby(['Title'],as_index=False).mean()


# 我们可以把分类的titles转换成序数

# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


# 现在让我们安全的丢弃训练集和测试集中的特征Name吧。同样，训练集中我们也不需要特征PassengerId。

# In[ ]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


# ##转换分类特征
# 现在我们可以把包含字符串的特征转换成数值型的。这需要大量的算法。这可以帮助我们达成完成特征的目标。
# 
# 让我们通过把特征Sex转换成一个新的female=1,male=0的特征Gender着手吧。

# In[ ]:


for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


# ##完成连续的数值型特征
# 现在我们应该开始评估和完成有缺失和空值的特征了。我们就拿特征Age练手吧。
# 
# 我们认为要完成一个连续的数值型特征，有3个方法。
# 
# - 1.一个简单的方法是在平均值(mean)和标准差(standard deviation)之间生成一个随机数。
# - 2.更精确的猜测缺失值的方法是使用关联的特征。在我们的案例中观察到，Age,Gender,Pclass是有联系的。使用特征Pclaass和Gender组合的集合Age的中位数(median)来猜测Age的值。所以Pclass=1和Gender=0,Pclass=1和Gender=1等等组成的集合Age的中位数就是年龄的值。
# - 3.整合方法1和方法2。所以不是通过中位数来猜测年龄的值,而是基于Pclass和Gender组合成的集合，使用平均值和标准差之间的随机数。
# 
# 方法1和方法3将会给我们的模型带来随机噪音。执行起来结果会出现多样性。所以我们更倾向于使用方法2.

# In[ ]:


# grid = sns.FacetGrid(train_df, col='Pclass', hue='Gender')
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# 让我们通过准备一个空的数组来保存，基于Pclas x Gender集合的猜测的Age的值。

# In[ ]:


guess_ages = np.zeros((2,3))
guess_ages


# 现在我们遍历Sex(0 or 1)和Pclass(1,2,3)的6个集合来计算猜测的年龄值。

# In[ ]:


for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


# 让我们创建年龄段，并确定与Survived的相关性。

# In[ ]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# 让我们基于这些年龄段把Age用序数词替换。

# In[ ]:


for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


# 我们可以移除特征AgeBand了。

# In[ ]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


# ##组合已有的特征创建新的特征
# 我们可以创建一个整合了Parch和SibSp的新特征FamilySize。这样让我们能够从训练集中丢弃Parch和SibSp。

# In[ ]:


for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# 我们可以创建另外的特征IsAlong。

# In[ ]:


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


# 为了让FamilySize支持IsAlone,让我们丢弃特征Parch吧。

# In[ ]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


# 我么也可以整合Pclass和Age创建一个人工的特征。

# In[ ]:


for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# ##完成分类特征
# 基于乘船港口，特征Embared有S，Q，C这3个值。我们的训练集有两个缺失值。我们用最普遍发生(occurance)的情况来简单填充它。

# In[ ]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ##把分类特征转换成数值型的 
# 
# 我们可以通过创建一个数值型的Port特征来转换特征EmbarkedFill。

# In[ ]:


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


# ##快速完成和转换数值型特征
# 因为在测试集中只有单一的缺失值，我们现在可以使用众数来得到最频繁发生的值来完成特征Fare。这只需要一行代码。
# 
# 注意我们替换的仅仅只是单一值，所以我们没有无限制的创建新特征或者对猜测缺失值的相关性做更一步的分析。完成目标的实现(achieve)，必需要模型算法在非空值上进行操作。
# 
# 我们也可能想要把这个票价换成两位小数，因为代表货币(currency)。

# In[ ]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


#  我们不需要创建FareBand 。

# In[ ]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


# 基于FareBand把特征Fare转化成序数。

# In[ ]:


for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)


# 以及测试集。

# In[ ]:


test_df.head(10)


# ##模型，预测和解决

# 现在我们准备训练模型和预测必需的解决方案。有60+的建模算法(modelling algorithm)可供选择。我们必须了解问题的类型和解决方案的要求，以缩小(narrow down )到我们可以评估的几个模型。我们的问题是分类和回归问题。我们想要识别出在输出(Survived or not)和其他变量或特征(Gender,Age,Port...)之间的关系。我们也把它们划分成机器学习的分类中的监督学习(supervised learning)，因为我们在训练模型时，是用的给定的数据集。通过监督学习再加上分类和回归的标准，我们可以缩小选择的模型数量。包括：
# 
# - 逻辑回归(Logistic Regression)
# - KNN或者k近邻(k-Nearest Neighbors)
# - 支持向量机(Support Vector Machine)
# - 朴素贝叶斯分类器(Naive Bayes classifier)
# - 决策树(Decision Tree)
# - 随机森林(Random Forest)
# - 感知机(Perceptron)
# - 人工神经网络(Artificial neural network)
# - RVM或者关联向量机(Relevance Vector Machine)

# In[ ]:


X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# 逻辑回归是在工作流的早期运行的有用模型。 逻辑回归通过使用逻辑函数估计概率来衡量分类因变量（特征）与一个或多个独立变量（特征）之间的关系，其是累积逻辑分布。 参考[维基百科](https://en.wikipedia.org/wiki/Logistic_regression)
# 
# 注意基于我们的训练数据集的模型产生的信心分数。

# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# 我们可以使用逻辑斯蒂回归分类来验证我们对特征创建和完成目标的猜想和决策。这可以通过计算决策函数中特征的系数(coefficien)来完成。
# 
# 正系数(positive coefficient)提高了反应的对数几率（从而增加了概率），负系数降低了反应的对数几率（从而降低了概率）。
# 
# - Sex是最大的正系数，意味着Sex值的增加(男性:0至女性:1),Survived=1的概率(probability)最大。
# - 随着Pclass的增加，Survived=1的概率反而减少到最小。
# - Age*Class对模型来说就是很好的人工特征，因为它和Survived是第二负相关的。
# - 因此Title是第二正相关的。

# In[ ]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# 接下来，我们使用支持向量机模型，它们是具有相关学习算法的监督学习模型，用于分析分类和回归分析的数据。 给定一组训练样本，每个训练样本被标记为属于两个类别中的一个或另一个，SVM训练算法构建一个将新的测试样本分配给一个类别或另一个类别的模型，使其成为非概率二进制线性分类器。 参考[维基百科](https://en.wikipedia.org/wiki/Support_vector_machine)
# 
# 请注意，该模型产生的自信分数要高于逻辑斯蒂回归模型。

# In[ ]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# 在图案识别中，k-近邻算法（或简称k-NN）是用于分类和归一化的非参数方法。 样本被离它最近的投票最多的所分类，样本被分配给其k个最近邻居中最常见的类（k是正整数，通常较小）。 如果k = 1，则将对象简单地分配给该单个最近邻的类。 参考[维基百科](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
# 
# KNN的自信分要高于逻辑斯蒂回归，但差于SVM。

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# 在机器学习中，朴素贝叶斯分类器是一个简单的概率分类器家族，基于贝叶斯定理与特征之间的强烈（朴素）独立假设。 朴素贝叶斯分类器具有很高的可扩展性(scalable)，需要一系列参数在学习问题中的变量（特征）的数量的线性关系。 参考维基百科
# 
# 该模型产生的自信分是迄今为止评估的模型中最低的。

# In[ ]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# 感知器是用于二进制分类器的监督学习的算法（可以决定由数字向量表示的输入是否属于特定类别的函数）。 它是一类线性分类器，即基于将权重集合与特征向量组合的线性预测函数进行预测的分类算法。 该算法允许在线学习，因为它一次处理训练集中的元素。 参考[维基百科](https://en.wikipedia.org/wiki/Perceptron)

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


# 该模型使用决策树作为将特征（树枝）映射到关于目标值（树叶）的结论的预测模型。 目标变量可以采取一组有限值的树模型称为分类树; 在这些树结构中，叶表示类标签，分支表示导致这些类标签的特征的连接。 目标变量可以采用连续值（通常为实数）的决策树称为回归树。 参考[维基百科](https://en.wikipedia.org/wiki/Decision_tree_learning)
# 
# 模型信心分数在迄今评估的模型中是最高的。

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# 随机森林是最受欢迎的。 随机森林或随机决策森林是一种用于分类，回归和其他任务的综合学习方法，通过在训练时间内构建多个决策树（n_estimators = 100）并输出作为类别（分类）模式的类， 或平均预测（回归）。 参考[维基百科](https://en.wikipedia.org/wiki/Random_forest)
# 
# 模型信心分数在迄今评估的模型中是最高的。 我们决定使用此模型的输出（Y_pred）来创建我们的竞争提交的结果。

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# ##模型评估
# 
# 现在我们可以对所有模型的评估进行排序，为我们的问题选择最好的。 当决策树和随机森林得分相同时，我们选择使用随机森林，因为它们正确地决定了树木过度适应训练集的习惯。

# In[ ]:


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


# 我们提交给比赛网站Kaggle的结果是在总共6,082名中获得了3,883名。 这个结果在比赛开始时就是指导性的。 此结果仅仅是提交数据集的一部分。 对我们的第一次尝试来说还不错。 任何提高我们成绩的建议都是最受欢迎的。
