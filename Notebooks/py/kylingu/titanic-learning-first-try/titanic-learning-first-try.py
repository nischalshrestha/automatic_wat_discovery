#!/usr/bin/env python
# coding: utf-8

# I am new to Machine Learning, any comment or advice will be appreciated. As this is a record for my machine learning, so I use my native language Chinese to write kernel. But I think code and visualization is general, wish you could understand.
# 
# I have also referred some best kernel, like [End-to-End process for Titanic problem
# ](https://www.kaggle.com/massquantity/end-to-end-process-for-titanic-problem). It provide a complete best practise about machine learning, a complete way of model selection, hyperparameter tunning and ensemble methods.
# 
# I would also recommend another best article [Titanic top 4% with ensemble modeling ](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling/notebook)
# 
# # 1. Titanic Data Set Feature Description
# - survival, 存活, 0 = No, 1 = Yes
# - Name, 乘客姓名
# - pclass, 票等, 1 = 1st, 2 = 2nd, 3 = 3rd，折射阶级，1st = Upper, 2nd = Middle, 3rd = Lower
# - sex，性别	
# - Age，年龄
# - sibsp， 船上兄弟姐妹或者配偶的数量
# - parch，船上父母或儿童的数量	
# - ticket，票号
# - fare，旅客费用	
# - cabin，客舱号码，	
# - embarked，登船港口，C = Cherbourg, Q = Queenstown, S = Southampton
# 
# # 2. Question
# 问题是要预测Survived 人数。
# > On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# 
# 说明，32%应该是一个baseline了，也即我们预测的结果，应该整体合起来得在32%的存活率附近，超过或者少了，预测都没达到最好。
# 
# > One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# 
# 没有足够的救生圈给乘客和船员，那么哪些人能够容易获得救生圈呢？上层阶级？或者是那些船员或乘客乘客距离救生圈足够近的人。这些人获得救生圈应该可以大大提高存活几率。第一个是船舱比较靠近的，第二个是更有接近这些就剩设施的船员，第三个是具有更高优先级的乘客，上层或者高官。这暗示我们应该去了解船舱分布以及救生设施分布。所以去了解featureCabin,和Name里的Title。
# 
# > Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# **这句明显在给提示，women，表明性别很重要，children，表明年龄很重要，upper-class说明，舱等和Fare更高的人存活几率应该更高。**
# 
# # 3. Loading Data

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# disable copy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'
# disable sklearn deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.


# In[2]:


df_train = pd.read_csv('../input/train.csv').set_index('PassengerId')
df_test = pd.read_csv('../input/test.csv').set_index('PassengerId')
dataset = pd.concat([df_train, df_test], axis=0)
id_test = df_test.index
dataset.tail()


# In[3]:


dataset.info()


# 我们可以看到共有891条数据，其中Cabin缺失严重，只有295条，Age也缺失263条。Age或许可以根据sibsp与Parch推断。在看一下统计信息。

# In[4]:


dataset.describe(include='all').T


# 我们知道unique太多的话，不是一个很好的预测变量，因此Name显然不是一个很好的预测变量。Ticket为什么会有重复的,虽然票号不是很好的预测变量，但或许对我们推断Cabin会有用处。先看看把。

# # 4. Overview Data
# 我们先看看每个feature或者feature 组合对Survived存活的影响。按照ISL里介绍，这个应该去计算F值或者是P值，才知道有无关系，不过这是针对线性回归的，我不清楚这个分类要怎么看。现在先凭借肉眼看罢。

# In[5]:


dataset.columns


# 我们可以先看看数值类型的数据对Survived相关性，下图显示，只有Pclass和Fare有点关系，还是弱相关。

# In[6]:


fig = plt.figure(figsize=(9, 6))
sns.heatmap(df_train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), cmap='cool', annot=True, ax=fig.gca())


# ## 4.1 Pclass
# 让我们先看Pclass舱等对存活率的影响。

# In[7]:


df_train.groupby('Pclass').Survived.agg(['count', 'sum', 'mean', 'std'])


# In[8]:


# plt.bar([1, 2, 3], (df_train.groupby('Pclass').Survived.mean()), color='rgb', tick_label = ['1', '2', '3'])
# sns.despine()

sns.factorplot(x='Pclass', y='Survived', data=df_train, kind='bar', size=5, palette='cool')


# 一等舱的幸存率要比二、三等舱的都要高，并且随着舱等的下降，幸存率也在下降。
# ## 4.2 Name
# Name是一个Unique类型的数据，本来我们是不打算看这个数据的。但其实浏览一下Name也是能看出一些端倪的，比如Braund, Mr. Owen Harris，除了名字还有称呼，如Mr，Miss，Mrs以及Master这样的信息。或须也能够让我们对Survived做出一些判断。同样上文提到，我们对于其是否为上层人士非常感兴趣，所以title没准也是可以说明问题的。
# 

# In[9]:


dataset.Name.head(10)


# In[10]:


dataset['title'] = dataset.Name.apply(lambda x: x.split(', ')[1].split('. ')[0])
# Major, 少校；Lady，贵妇；Sir，子爵; Capt, 上尉；the Countess，伯爵夫人；Col，上校。Dr,医生？
dataset['title'].replace(['Mme', 'Ms', 'Mlle'], ['Mrs', 'Miss', 'Miss'], inplace = True)
dataset['title'].value_counts()


# 我们看到，贵族或者大人数，都是相对较少的，所以这里，我们只要针对统计量，就能够区分开这些人群。另外Master这个title, 对应未满18岁的小男孩，而小女孩在Miss里。其他title都是大人的尊称，这或许是个比较好的Age预测量。

# In[11]:


dataset['title'] = dataset.title.apply(lambda x: 'rare' if x not in ['Mr', 'Miss', 'Mrs', 'Master'] else x)


# 我们如果将小女孩也独立独立出来，this would be a great predictive variable to Age，但这个就相当于漏掉那些没有年龄的姑娘了，我们需要看一下，那些没有年龄的姑娘占比能有多大。

# In[12]:


age_na_miss_rate = len(dataset[(dataset.title == 'Miss') & (dataset.Age.isnull()) ]) / (dataset.title == 'Miss').sum()
age_nna_not_mister_rate = len(dataset[(dataset.title == 'Miss') & (dataset.Age.notnull()) & (dataset.Age >= 18)]) / (dataset.title == 'Miss').sum()
print(age_na_miss_rate, age_nna_not_mister_rate)

len(dataset[(dataset.title == 'Miss') & (dataset.Age.isnull())]) / len(dataset)


# 没有年龄的小姐姐占整个数据集的3.89%,因此就算因此计算错了，影响倒也不大，所以我们就这么做了。

# In[13]:


dataset.title[(dataset.title == 'Miss') & (dataset.Age < 18)] = 'Mister'


# In[14]:


dataset.groupby('title').Age.agg([('number', 'count'), 'min', 'median',  'max'])


# 我们再看看itle对survived 影响。

# In[15]:


dataset.groupby('title').Survived.agg(['count', 'sum', 'mean', 'std'])


# In[16]:


g= sns.factorplot(x='title', y = 'Survived', data=dataset, kind='bar', palette='cool', size=5)
g.ax.set_title('title for survived');


# In[17]:


dataset['title_level'] = dataset.title.map({"Miss": 3, "Mrs": 3,  "Master": 2, 'Mister': 2,  "rare":2, "Mr": 1})


# ## 4.3 Sex

# In[18]:


dataset.groupby('Sex').Survived.agg(['count', 'sum', 'mean', 'std'])


# 哇哦，女性几乎达到了74%的存活率，而男性几乎只有18%的存活几率，如果我们再排除男性青少年，估计成年男性的存活更是低了。这是一个很重要的预测变量。
# 

# In[19]:


dataset.Sex = dataset.Sex.map({'female':0, 'male':1})


# ## 4.4 SipSp与Parch
# 再来看以下，SibSp与Parch对Survived的影响。

# In[20]:


dataset.SibSp.value_counts().plot(kind='bar')


# In[21]:


dataset.Parch.value_counts().plot(kind='bar')


# In[22]:


# dataset.groupby('SibSp').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='SibSp', y='Survived', size=5, data=dataset, kind='bar', palette='cool')


# In[23]:


# df.groupby('Parch').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='Parch', y='Survived', size=5, data=dataset, kind='bar', palette='cool')


# 以上两组数据，可以认为似乎有一两个Parch或者一两个SibSp存活几率更高一点。那么我们将这个组成家庭大小，所谓人多力量大，如果有兄弟姐妹的似乎更高。很多kernel里将family size + 1，然后单独列出一个alone列。我再想familysize为0时候，不就是一个人吗？为什么要单独列出来？

# In[24]:


dataset['family_size'] = dataset.SibSp + dataset.Parch
# df.groupby('family_size').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='family_size', y='Survived', size=5, data=dataset, kind='bar', palette='cool')


# 结果显示有1-3个亲属的人，存活几率也更高。或许我们应该根据family size进行分组，如只身一人，以及有1-3人亲属的存活率也较高，而更大的家庭，虽说数量不多，但是存活几率就立刻下降。家庭成员过多，将变成负担。所以存活率因此下降。让我们试试看，结果不错哦，可以认为也是非常不错的预测变量。标签用数字，这样后面也能够计算相关性。

# In[25]:


dataset['family_size_level'] = pd.cut(dataset.family_size, bins=[-1,0, 3.5, 12], labels=['alone', 'middle', 'large'])
# df.groupby('family_size_level').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='family_size_level', y='Survived', size=5, data=dataset, kind='bar', palette='cool')
dataset['family_size_level'] = dataset['family_size_level'].map({'alone':1, 'middle':2, 'large':0})


# 或许结合一下性别看看？

# In[26]:


# dataset.groupby(['family_size_level', 'Sex']).Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot(x='family_size_level', y='Survived', hue='Sex', size=5, data=dataset, kind='bar', palette='cool')


# 女性不论是独自一人还是有亲属存在，都达到了非常高的存活几率。而独身一人的青壮中老年死亡率都很高。而对于中型家庭，除了老年存活率较低外，其他也都很高。

# ## 4.5 Fare
# 上面提到，上层的存活几率更高，所以除了Pclass能够表明身份之外，其实所花费用也是身份象征。

# In[27]:


dataset[dataset.Fare.isna()]


# In[28]:


dataset.Fare.fillna(dataset[dataset.Pclass == 3].Fare.median(), inplace=True)


# In[29]:


# dataset.Fare.hist(bins=20)
sns.distplot(dataset.Fare)


# 上图显示，绝大部分的人的费用都集中在100英镑以内（假如费用是以英镑记的话，其实这都不是问题）。我们知道财富是符合正太分布的，但乍看之下这幅图右边尾巴太长，左边太高，看起来像是对数正太分布。我们再看看Pclass和Fare的关系。

# In[30]:


dataset.Fare = dataset.Fare.apply(lambda x: np.log(x) if x > 0 else 0)
sns.distplot(dataset.Fare)


# In[31]:


dataset.groupby('Pclass').Fare.agg(['count', 'sum', 'mean'])


# 我们应该根据PClass对费用进行一个分组，不过这其实相当于另一个Pclass了，所以还是算了。感觉这个变量的作用与Pclass的重复了。
# 那么我们要怎么做呢？有两种思路，一种是将Fare按照2/8分。第二种，是构建新的Feature，即Pclass * Fare, 其中将Pclass转换成3，2，1使得舱等级越高，费用越大，让二者更加凸显身份和地位。再说这，在这两者基础上，在第二种构建完成后，再按照2/8分。似乎最后一种最靠谱，也许富人低调？花的少？首先舱等是对Fare的一种倍乘。是否需要两个feature倍乘，得看两个feature是否有相互效应，按照ISL里所说。

# In[32]:


# def doInverse(x):
#     if x == 3:
#         return 1
#     elif x == 1:
#         return 3
#     else:
#         return x

# df.Pclass.head(3), df.Pclass.apply(lambda x: doInverse(int(x))).head(3), df.Fare.head(3)

# fares = df.Fare.multiply(df.Pclass.apply(lambda x: doInverse(int(x))))
# plt.figure(figsize=(9, 6))
# fares.hist(bins=40)
# # 过滤出20%的人。
# fares.quantile(0.80)


# In[33]:


# a = list(range(0, 401, 40))
# a.append(2500)


# df_temp = df.copy()
# df_temp['fare_level']= pd.cut(fares, bins=a)
# df_temp.groupby('fare_level').Survived.agg(['count', 'sum', 'mean'])


# In[34]:


# 上限尽量设的大点，因为我们这里没把test data也一起拿出来看，所以不知道范围。
# df['upper_class'] = pd.cut(fares, bins=[0, 40, 160, 2500], labels=['low', 'middle', 'upper'])
# df.groupby('upper_class').Survived.agg(['count', 'sum', 'mean'])


# 当我把upper class，不论是加入到预测Age或者是加入到预测Survived中，最后结果都是保持不变，所以我依旧觉得这个变量与Pclass重了，冗余了。

# In[35]:


# plt.figure(figsize=(9, 6))
# sns.heatmap(df[['Pclass', '']], cmap='cool', annot=True)


# ## 4.6 Ticket

# In[36]:


dataset.Ticket.head(5)


# Ticket或许能够说明距离逃生窗口之类的远近，然后我们没有这方面的知识，暂且不考虑这个变量，否则就能够根据他们和窗口的距离划分。
# ## 4.7 Cabin
# 客舱号码类似于Ticket的一个变量，但是缺失严重，77%的数据都已经缺失。会有类似国内无座这种情况存在吗？

# In[37]:


dataset.Cabin.isna().sum() / len(dataset.Cabin)


# In[38]:


dataset.Cabin = dataset.Cabin.apply(lambda x : 0 if pd.isna(x) else 1)
sns.factorplot(x='Cabin', y='Survived', data=dataset, kind='bar')


# 有座的明显要比无座的幸存率高啊

# ## 4.8 Embarked
# 登陆港口与存活几率有关吗

# In[39]:


dataset.Embarked.isna().sum(), '--'*12, dataset.Embarked.value_counts()


# In[40]:


dataset.Embarked.fillna('S', inplace=True)


# In[41]:


dataset.groupby('Embarked').Survived.agg(['count', 'sum', 'mean', 'std'])


# 似乎各个港口登陆的存活几乎都差不多，如果我们结合PClass看看是否还是如此呢。

# In[42]:


dataset.groupby(['Pclass', 'Embarked']).Survived.agg(['count', 'sum', 'mean'])
sns.factorplot(x='Embarked', y = 'Survived', hue='Pclass', data=dataset, size=5, kind='bar', palette='cool')


# 我们注意到三等舱从S港口登船的幸存率最低，只有0.189。上面提到，女性存活率很高，那么三等舱S港口是否是男性比例较高导致的呢？那么我们重点观察以下S港口的三等舱的性别比例。

# In[43]:


# a = df.groupby(['Pclass', 'Embarked', 'Sex']).Survived.agg(['count', 'sum', 'mean', 'std'])
dataset[(dataset.Embarked == 'S') & (dataset.Pclass == 3)].groupby('Sex').Survived.agg(['count', 'sum', 'mean', 'std'])


# 三等舱的存活几率都很低，女性也超过一般死亡，男性就更低了，只有12.8%。要加入一类判断是否是三等舱，S港口的？
# 

# ## 4.9 Age
# 由于Age数据有缺失，只针对有数据的部分先做一下统计。先看看年龄分布，还是呈现正太分布的。如果我们后面Wrongle Data的话，也是有必须要看看填充后是否继续符合这个年龄分布。

# In[44]:


# df.Age.hist(bins=20)
sns.distplot(dataset.Age[dataset.Age.notna()])


# In[45]:


fig = plt.figure(figsize=(9, 7))
g = sns.kdeplot(dataset.Age[(dataset.Survived == 1) & (dataset.Age.notna())], color='r', ax=fig.gca())
g = sns.kdeplot(dataset.Age[(dataset.Survived == 0) & (dataset.Age.notna())], color='b', ax=fig.gca())
g.set_xlabel('Age')
g.set_ylabel('Survived')
g.legend(['Survived', 'Not'])


# In[46]:


dataset['age_level'] = pd.cut(dataset.Age, bins=[0, 18, 60, 100], labels=[3, 2, 1])
# dataset.groupby('age_level').Survived.agg(['count', 'sum', 'mean', 'std'])
sns.factorplot('age_level', 'Survived', hue='Sex', data=dataset, kind='bar', size=5, palette='cool')


# 说明幼儿存活率更高，老人的存活几率非常低，而青壮年男性是牺牲的主力。我们将其与性别结合看看结果。holycrap, 青壮年男性存活几率不到20%，即使是老年组里，老年男性也几乎全跪。

# # 5. Cleaning Data, Wrangle
# 我们知道缺失数据的有Cabin>Age>Embarked。Cabin我们已经将其转换为有或无了，Embarked的两个缺失也被我们填充了众数。而Age毕竟是一个非常影响预测结果的变量，我们需要好好填充一下
# ## 5.1 Fill Age
# 我们这里就不去预测离散的年龄了，而是预测年龄的分组，这样就使得回归的年龄问题转变为分组问题。

# In[47]:


dataset.head(5)


# 我们需要那些变量来预测我们的年龄呢？这里我们就可以看看相关性矩阵

# In[48]:


# dataset[dataset.age_level.notna()].age_level = dataset.age_level[dataset.age_level.notna()].astype('int')
fig = plt.figure(figsize=(9, 6))
sns.heatmap(dataset[['Age', 'Survived', 'Pclass', 'Sex', 'family_size_level', 'Parch', 'SibSp', 'title_level', 'age_level', 'Fare']].corr(), cmap='cool', annot=True, ax=fig.gca())


# 上图中，跟Age相关比较大的有Pclass, family_size, Fare，Sex和title_level无关，这可能与我们设置的值有关系。

# In[49]:


dataset.groupby('title_level').Age.agg(['mean', 'median', 'std'])


# In[50]:


dataset.groupby(['title']).Age.agg(['mean', 'median', 'std', 'max'])


# 我们可以看到，title_level设置的对Age波动非常大，而其本身title的std波动就非常小，说明基本上都在mean的附近。所以我们认为title也是与age比较相关的一个重要因素。假如我们把title按照年龄重新分组，我们就比较有可能得到相关性的证据了，我们按照Age的均值与中值对title进行重新编码值。

# In[51]:


dataset['title_age_level'] = dataset.title.map({"Master": 1, 'Mister': 1, "Miss": 2, "Mrs": 3,  "Mr": 3, "rare": 4})
# dataset['title_age_level'] 


# In[52]:


fig = plt.figure(figsize=(9, 6))
sns.heatmap(dataset[['Age', 'Survived', 'Pclass', 'Sex', 'family_size_level', 'Parch', 'SibSp', 'title_level', 'title_age_level', 'age_level', 'Fare']].corr(), cmap='cool', annot=True, ax=fig.gca())


# In[53]:


['Survived', 'Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Fare']


# 这样结果就比较明显了, 
# - Age与Pclass，family_size, title_age_level, Fare的相关性就比较明显了。
# - 同时，我们还发现Fare与Pclass也具有强相关性，
# - Sex不知道怎么与title_level达到如此强的负相关。。。。
# - 如果考虑Survived的话，也能够看出，Survived与Pclass，Sex，title_level, title_age_level, Fare的相关性都不错，这里我们的family_size居然没那么明显，看来还是值设置的有问题？

# In[54]:


df_age_train = dataset[dataset.Age.notnull()]
df_age_test = dataset[dataset.Age.isnull()]

df_age_train.shape, df_age_test.shape


# In[55]:


df_age = df_age_train[['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level', 'Fare', 'age_level']]
X_age_dummied = pd.get_dummies(df_age.drop(columns='age_level'), columns=['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level'])
X_age_dummied['SibSp_8'] = np.zeros(len(X_age_dummied))
X_age_dummied['Parch_9'] = np.zeros(len(X_age_dummied))


Y_age = df_age['age_level']


# In[56]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

params = {'n_estimators':range(10, 40, 5), 'max_depth':[3, 4, 5], 'max_features':range(3, 9)}
clf = RandomForestClassifier(random_state=0)

gscv = GridSearchCV(estimator=clf, param_grid=params, scoring='f1_micro', n_jobs=1, cv=5, verbose=1)
gscv.fit(X_age_dummied, Y_age)


# In[57]:


gscv.best_score_, gscv.best_params_, gscv.best_estimator_.feature_importances_


# > 在加入对小姐姐的titleMister后，我们的Age的预测准确率又上升了，并且经过两次调参，获得了0.94537815126050417的f1_micro.
# 
# 之前的版本达到了0.945，现在改成全部数据，反而达不到了。看来有离群点加入进来了，导致我们无法获得比较好的结果。😅

# In[58]:


pd.Series(gscv.best_estimator_.feature_importances_, index=X_age_dummied.columns).sort_values(ascending=True).plot.barh(figsize=(9, 6))
sns.despine(bottom=True)


# In[59]:


X_age_dummied = pd.get_dummies(df_age.drop(columns='age_level'), columns=['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level'])

ab = df_age_test[['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level', 'Fare']]
X_age_dummied_test = pd.get_dummies(ab, columns=['Pclass', 'Sex', 'Parch', 'SibSp', 'title_age_level'])
X_age_dummied.shape, X_age_dummied_test.shape


# In[60]:


X_age_dummied.columns, X_age_dummied_test.columns


# In[61]:


X_age_dummied_test['Parch_3'] = np.zeros(len(X_age_dummied_test))
X_age_dummied_test['Parch_5'] = np.zeros(len(X_age_dummied_test))
X_age_dummied_test['Parch_6'] = np.zeros(len(X_age_dummied_test))

X_age_dummied_test['SibSp_4'] = np.zeros(len(X_age_dummied_test))
X_age_dummied_test['SibSp_5'] = np.zeros(len(X_age_dummied_test))

df_age_test.age_level = gscv.predict(X_age_dummied_test)


# In[62]:


df_age_test.shape


# In[63]:


df_final = pd.concat([df_age_test, df_age_train]).sort_index()
df_final.info()


# ## 5.3 Clean Done
# Trainning里的所有数据都已经清理完毕，没有NA值了，接下来就进入模型评估阶段。
# # 6. Basic Model & Evalution
# ## 6.1 Model & Evaluation
# 我们选用如下的Models，都是我在课上学过的额，然后用cross_validataion_score获得一个初始评估。
# - Logistic Regression
# - KNN
# - SVM
# - Naive Bayes
# - Nerual Network, MLP
# - DecisionTree
# - RandomForest
# - Gradient Boosting Decision Tree

# In[64]:


from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[65]:


names = ['LR', 'KNN', 'SVM', 'Bayes', 'NW', 'DT', 'RF', 'GBDT']
models = [LogisticRegression(random_state=0), KNeighborsClassifier(n_neighbors=3), SVC(gamma='auto', random_state=0), GaussianNB(), MLPClassifier(solver='lbfgs', random_state=0),
         DecisionTreeClassifier(), RandomForestClassifier(random_state=0), GradientBoostingClassifier(random_state=0)]


# Feature Selection，我们选择在上面那张里和Survived具有比较大的相关性的Feature。

# In[66]:


selection = ['Survived', 'Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Fare', 'Embarked', 'Cabin']
train_set = df_final[df_final.Survived.notna()][selection]
test_set = df_final[df_final.Survived.isna()][selection]
X_train = train_set.drop(columns='Survived')
Y_train = train_set['Survived']
X_test = test_set.drop(columns='Survived')

X_train= pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])
X_test = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])
X_train.shape, X_test.shape


# In[67]:


X_train.columns, X_test.columns


# In[68]:


for name, model in zip(names, models):
    score = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc')
    print('score on {}, mean:{:.4f}, from {}'.format(name, score.mean(), score))


# 未调仓情况下，我们在LR和GBDT上获得两个最高分数0.8668，0.8842。由于某些Classifier对数据的量级敏感，因此，我们需要对数据进行一个预处理。但是，我们几乎所有的数据都被转换成了dummies的形式，Fare数据也被我们log化，转换到0-7之间了，所以我觉得并不需要`from sklearn.preprocessing import StandardScaler`它去缩放数据。
# 
# ## 6.2 Feature importance
# 我们可以直观上看看LogisticRegression认为哪些feature权重比较高。

# In[69]:


lr = LogisticRegression(random_state=0).fit(X_train, Y_train)
lr.coef_


# In[70]:


from matplotlib import cm
color = cm.inferno_r(np.linspace(.4,.8, len(X_train.columns)))

pd.DataFrame({'weights': lr.coef_[0]}, index=X_train.columns).sort_values(by='weights',ascending=True).plot.barh(figsize=(7, 7),fontsize=12, legend=False, title='Feature weights from Logistic Regression', color=color);
sns.despine(bottom=True);


# 这个Feature Weights还挺符合我们的认知的，**familysize_level_1（1人），familysize_level_2（2-4人），title_level_3(Mrs, Miss)，age_level_3(child)， Sex_0(女性)，title_level_2（Mister, Master, rare）, Fare(费用)，Cabin_1(有客舱)，Pclass_1（1等舱），Pclass_2(二等舱)，Embarked_C(C港口)**都具有提高幸存率的参数。**这里为什么familysize_level为1个人时候，为什么对判定影响这么大**？？？
# 
# 另外，**faimily_size_level_0（大型家庭），title_level_1（Mr）, Sex_1(male)，age_level_1(老年人)，Cabin_0(无客舱)，Pclass_3（三等舱），Embarked_S(S港口)、age_level_2(年轻-中年人)**都就具有较大的负权重，这些都是对Survived的debuff。与我们上面最开始feature分析的还是一致的。
# 
# 
# 我们还可以试着看看DecisionTree的feature importance

# In[71]:


def plot_decision_tree(clf, feature_names, class_names):
    from sklearn.tree import export_graphviz
    import graphviz
    export_graphviz(clf, out_file="adspy_temp.dot", feature_names=feature_names, class_names=class_names, filled = True, impurity = False)
    with open("adspy_temp.dot") as f:
        dot_graph = f.read()
    return graphviz.Source(dot_graph)

dtc = DecisionTreeClassifier().fit(X_train, Y_train)

pd.DataFrame({'importance': dtc.feature_importances_}, index=X_train.columns).sort_values(by='importance',ascending=True).plot.barh(figsize=(7, 7),fontsize=12, legend=False, title='Feature importance from Decision Tree', color=color);
sns.despine(bottom=True, left=True);

# 这个DecisionTree太大了啊，真实天性容易过拟合，这层数也太多了，所以不画了。
# plot_decision_tree(dtc, X_train.columns, 'Survived')


# DT里的feature importance，按我的理解，是按照**title_level_1（Mr）进行大范围拆分，然后是Fare、family_size_0（家庭成员过多的）、Pclass_3（三等舱）、Embarked_S（S港口），age_level_2（年轻-中年）、age_level_3(老年人)、age_level_1(儿童)、Cabin_0（无客舱）**等等等。
# 
# ## 6.3 Misclassification Analysis
# 通过观察被算法误分类的records，设计出新的feature或其他的方法，来减少误分类，最终进一步提高分类的准确度。怎么做呢？还是在K-Fold上，然后调用算法，将其分类错误的找出来。

# In[72]:


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=4, random_state=0)

# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# miss classifications
mcs = []

for train_index, test_index in skf.split(X_train, Y_train):
    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
    y_pred =lr.fit(x_train, y_train).predict(x_test)
    mcs.append(x_test[y_pred != y_test].index.tolist())


# In[73]:


mcs_index = np.concatenate((mcs))
len(mcs_index), len(Y_train), 'miss classification rate:', len(mcs_index)/len(Y_train)


# 我们看到Miss Classification Rate为18.4%，与我们之前所得也差不多。接下来我们就看看这166个，按照LR的feature weights为何会被错分类呢，有什么特点吗？

# In[74]:


# mcs_df = pd.concat([X_train.iloc[mcs_index], Y_train.iloc[mcs_index]], axis=1)
mcs_df = train_set.iloc[mcs_index]
mcs_df.head()


# In[75]:


mcs_df.describe(include='all').T


# 我们解读一下以上数据，它描述了大部分被分类错误的人的基本信息，这里的存活率为0.373，非常低，即大部分都是死亡，**也就是我们的目标是怎么让这些人被预测为死亡**。其中大部分三等舱，男性，S港口，独身，中年人被预测为存活。所以我们的目标是根据LR的feature weights，添加更容易被预测为死亡的feature融合，即Feature weights为负的那些feature。
# 
# 我们把mcs_df里那些占比例比较多的Feature，并且是LR feature weights里为正的那些主要feature提取出来，进行组合以便于降低判定为1的几率。
# > faimily_size_level_0（大型家庭），title_level_1（Mr）, Sex_1(male)，age_level_1(老年人)，Cabin_0(无客舱)，Pclass_3（三等舱），Embarked_S(S港口)、age_level_2(年轻-中年人)
# 
# 
# 正参数feature。
# 
# > familysize_level_1（1人），familysize_level_2（2-4人），title_level_3(Mrs, Miss)，age_level_3(child)， Sex_0(女性)，title_level_2（Mister, Master, rare）, Fare(费用)，Cabin_1(有客舱)，Pclass_1（1等舱），Pclass_2(二等舱)，Embarked_C(C港口)

# In[76]:


mcs_df[mcs_df.family_size_level == 1].shape, mcs_df[mcs_df.title_level == 3].shape, mcs_df[mcs_df.age_level == 3].shape


# 由于family_size_level=1（一个人）给了过大的比重，造成错误的分类，那么怎么降低这个family_size_level为一个人时候的比重呢？

# In[77]:


mcs_df.groupby('family_size_level').Survived.mean()


# In[78]:


# df_final['MPSE'] = np.ones(len(df_final))
# df_final.MPSE[(df_final.title_level == 1) & (df_final.Pclass == 3) & (df_final.Sex == 0) & (df_final.Embarked == 'S') & (df_final.family_size_level.isin([1, 2]))] = 4
# df_final.MPSE[(df_final.title_level == 1) & (df_final.Pclass == 2) & (df_final.Sex == 0) & (df_final.Embarked == 'S') & (df_final.family_size_level.isin([1, 2]))] = 3
# df_final.MPSE[(df_final.title_level == 1) & (df_final.Pclass == 1) & (df_final.Sex == 0) & (df_final.Embarked == 'S') & (df_final.family_size_level.isin([1, 2]))] = 2
# df_final.MPSE.value_counts()

df_final['Alone'] = np.zeros(len(df_final))
df_final['Alone'][df_final.family_size_level == 1] = 10
df_final['EmbkS'] = np.zeros(len(df_final))
df_final['EmbkS'][df_final.Embarked == 'S'] = 10
df_final['MrMale'] = np.zeros(len(df_final))
df_final['MrMale'][df_final.title_level == 1] = 10


# 验证一下，是否有改进？从0.8668，到0.8674， 只有0.001丁点儿的改进，😰

# In[79]:


selection = ['Survived', 'Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Fare', 'Embarked', 'EmbkS', 'MrMale', 'Cabin', 'Alone']
train_set = df_final[df_final.Survived.notna()][selection]
test_set = df_final[df_final.Survived.isna()][selection]
X_train = train_set.drop(columns='Survived')
Y_train = train_set['Survived']
X_test = test_set.drop(columns='Survived')

X_train= pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])
X_test = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'family_size_level', 'title_level', 'age_level', 'Embarked', 'Cabin'])

val_lr = LogisticRegression(random_state=0)
a_score = cross_val_score(val_lr, X_train, Y_train, cv=5, scoring='roc_auc')
a_score.mean()


# # 7 Hyperparameters Tuning
# 我们选取几个表现比较好的进行调参，看看最终结果。如Logistic Regression，KNN，SVM， NW， RF，GBDT

# In[80]:


from sklearn.ensemble import GradientBoostingClassifier

def tune_estimator(name, estimator, params):
    gscv_training = GridSearchCV(estimator=estimator, param_grid=params, scoring='roc_auc', n_jobs=1, cv=5, verbose=False)
    gscv_training.fit(X_train, Y_train)
    return name, gscv_training.best_score_, gscv_training.best_params_


# ## 7.1 LR

# In[81]:


from sklearn.linear_model import LogisticRegression
params = {'C':[0.03, 0.1, 0.3, 1, 3, 10, 20, 30, 50]}
clf = LogisticRegression(random_state=0)
tune_estimator('LR', clf, params)


# In[82]:


# Second fine tuning.
params = {'C':[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
clf = LogisticRegression(random_state=0)

tune_estimator('LR', clf, params)


# ## 7.2 KNN

# In[83]:


params = {'n_neighbors': range(3, 15, 3)}
clf = KNeighborsClassifier()
tune_estimator('KNN', clf, params)


# In[84]:


params = {'n_neighbors': [10, 11, 12, 13, 14]}
clf = KNeighborsClassifier()
tune_estimator('KNN', clf, params)


# ## 7.3 SVM

# In[85]:


params = {'C':[0.01, 0.1, 1, 10], 'gamma':[0.001, 0.01, 0.1], 'kernel':['rbf']}
clf = SVC(random_state=0)
tune_estimator('SVM', clf, params)


# In[86]:


params = {'C':range(8, 14), 'gamma':[0.05, 0.08, 0.01, 0.03, 0.05], 'kernel':['rbf']}
clf = SVC(random_state=0)
tune_estimator('SVM', clf, params)


# ## 7.4 Nerual Network

# In[87]:


params = {'hidden_layer_sizes':[x for x in zip(range(20, 100, 10), range(20, 100, 20))],
          'solver':['lbfgs'], 'alpha': [0.0001, 0.001, 0.01]}
clf = MLPClassifier(random_state = 0)
tune_estimator('NW', clf, params)


# ## 7.5 Random Forest

# In[88]:


params = {'n_estimators':range(10, 40, 5), 'max_depth':[3, 5], 'max_features':range(3, 7)}
clf = RandomForestClassifier(random_state=0)
tune_estimator('RF', clf, params)


# In[89]:


# second round
params = {'n_estimators':range(31, 40, 2), 'max_depth':range(4, 7), 'max_features':range(3, 6)}
clf = RandomForestClassifier(random_state=0)
tune_estimator('RF', clf, params)


# ## 7.6 GBDT

# In[90]:


params = {'learning_rate':[0.001, 0.01, 0.1, 1], 'n_estimators':range(100, 250, 20), 'max_depth':range(2, 5), 'max_features':range(3, 6)}
clf = GradientBoostingClassifier(random_state=0)
tune_estimator('GBDT', clf, params)


# In[91]:


# second tune
params = {'learning_rate':[0.3, 0.5, 0.7, 0.9, 1.2, 1.4, 1.7, 2], 'n_estimators':range(130, 200, 20), 'max_depth':[2, 3, 4], 'max_features':[3, 4, 5]}
clf = GradientBoostingClassifier(random_state=0)
tune_estimator('GBDT', clf, params)


# ## 7.7 Conclusion
# **最终，我们获得最大的得分是采用GBDT，得分0.87741180236289573。**

# # 8. Ensemble Methods
# 知乎文章[【scikit-learn文档解析】集成方法 Ensemble Methods（上）：Bagging与随机森林](https://zhuanlan.zhihu.com/p/26683576)对集成方法进行了一些原理、使用方法的介绍。
# > 在机器学习中，集成方法（ensemble learning）把许多个体预测器（base estimator）组合起来，从而提高整体模型的鲁棒性和泛化能力。
# > 
# > 集成方法有两大类：
# > 
# > Averaging：独立建造多个个体模型，再取它们各自预测值的平均，作为集成模型的最终预测；能够降低variance。例子：随机森林，bagging。
# > Boosting：顺序建立一系列弱模型，每一个弱模型都努力降低bias，从而得到一个强模型。例子：AdaBoost，Gradient Boosting。
# 
# GBDT（gradient boosting decision tree）可以说是最强大的数据挖掘算法之一：它能解决各种分类、回归和排序问题，能优秀地处理定性和定量特征，针对outlier的鲁棒性很强，数值不需要normalize。而且，LightGBM和XGBoost等算法库已经解决了GBDT并行计算的问题（然而scikit-learn暂不支持并行）[XGBoost](https://xgboost.readthedocs.io/en/latest/)。在本质上，决策树类型的算法几乎不对数据的分布做任何统计学假设，这使它们能拟合复杂的非线性函数。
# 
# 其中RandomForest和GBDT我们在上述方法中也已经使用过了，这里我们使用其他的方法，比如Bagging、AdaBoosting、Voting以及Stacking。
# 
# ## 8.1 Bagging
# 因为RandomForest和GBDT都是基于决策树的，这里我们针对表现比较好的LR做一次Bagging。结果0.87594，是稍好于之前单一的LR 0.87440。流程图片来自文章[Ensemble of Weak Learners](http://manish-m.com/?p=794)，以及以下几个ensemble methods的流程图片俱来自于此。
# ![](http://manish-m.com/wp-content/uploads/2012/11/BaggingCropped.png)
# 
# 采用有放回的抽样，如果N趋向于M，则每个训练集将只拥有63%的原始数据集，其他都是重复的。

# In[92]:


from sklearn.ensemble import BaggingClassifier
params = {'n_estimators': range(50, 150, 10)}
bagging = BaggingClassifier(LogisticRegression(C=0.3))
tune_estimator('bagging', bagging, params)


# ## 8.2 AdaBoosting
# 知乎文章[【scikit-learn文档解析】集成方法 Ensemble Methods（下）：AdaBoost，GBDT与投票分类器](https://zhuanlan.zhihu.com/p/26704531)对Adabooting做了介绍，全程为Adaptive Boosting。
# > AdaBoost的核心理念，是按顺序拟合一系列弱预测器，后一个弱预测器建立在前一个弱预测器转换后的数据上。**每一个弱预测器的预测能力，仅仅略好于随机乱猜。**最终的预测结果，是所有预测器的加权平均（或投票结果）。
# > 
# > 最初，原始数据有N个样本（w_1, w_2, w_3, ... , w_N），每个样本的权重是1/N。**在每一个boosting迭代中，一个弱分类器会拟合训练数据。之后，弱分类器分类错误的样本将会得到更多的权重。相应地，正确分类的样本的权重会减少。在下一个的迭代里，一个新的弱分类器会拟合（权重被调整后的）新的训练数据。**因此，越难分类的样本，在一次又一次的迭代中的权重会越来越高。
# > 
# > 因此，AdaBoost的一个缺点就是不免疫数据中的outlier（尤其是分类分不对的outlier）。
# > 
# > AdaBoost可以看作是使用了exponential loss损失函数的Gradient Boosting。
# 
# 有点不明白的是，每次错误样本如何获得更多的权重呢？
# 
# 它组合的是一种叫做decision stump，决策树桩的分类器，也就是单层决策树，单层也就意味着尽可对每一列属性进行一次判断。大概长这样
# ![](https://img-blog.csdn.net/20160325144422445)
# 
# 优化目标，就是使得每一列的错误率判断的错误率最低。
# \begin{equation}
# \underset{1\leq i\leq d}{\arg\min}\frac1N\sum_{n=1}^N1_{y_n\neq g_i(\mathbf x)}
# \end{equation}
# 
# 流程
# ![](http://manish-m.com/wp-content/uploads/2012/11/BoostingCropped2.png)

# In[93]:


from sklearn.ensemble import AdaBoostClassifier
params = {'n_estimators':range(80, 200, 20), 'learning_rate':[0.5, 1, 3, 5]}
adc = AdaBoostClassifier(random_state=0)
tune_estimator('AdaBoosting', adc, params)


# In[94]:


# secound round
params = {'n_estimators':range(150, 180, 10), 'learning_rate':[ 0.7, 0.9, 1.3, 2]}
adc = AdaBoostClassifier(random_state=0)
tune_estimator('AdaBoosting', adc, params)


# ## 8.3 Voting
# > 投票分类器是一种元预测器（meta-estimator），它接收一系列（在概念上可能完全不同的）机器学习分类算法，再将它们各自的结果进行平均，得到最终的预测。
# > 
# > 投票方法有两种：
# > 
# > 硬投票：每个模型输出它自认为最可能的类别，投票模型从其中选出投票模型数量最多的类别，作为最终分类。
# > 软投票：每个模型输出一个所有类别的概率矢量(1 * n_classes)，投票模型取其加权平均，得到一个最终的概率矢量。
# 
# 软投票，是对每个模型的所有类别概率进行加权平均得到的，而不是像硬投票每个类认为最可靠的投票决定。
# 
# ![](http://manish-m.com/wp-content/uploads/2012/11/Voting_Cropped.png)
# 
# 迄今为止，我们已经运用并获得了大部分具有良好性能的算法了，这里我们就对其排个序，以期使用结果排名靠前的算法做一个Voting。

# In[95]:


lr = LogisticRegression(C=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
svc = SVC(C=9, gamma=0.1, random_state=0)
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=(20, 20), solver='lbfgs', random_state=0)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=31, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=3, n_estimators=190, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=140, random_state=0)
abc = AdaBoostClassifier(learning_rate=1.3, n_estimators=150, random_state=0)

names = ['LR', 'KNN', 'SVC', 'MLP', 'RF', 'GBDT', 'Bagging', 'AdaB']
models = [lr, knn, svc, mlp, rf, gbdt, bagging, abc]

result_scores = []
for name, model in zip(names, models):
    scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='roc_auc')
    result_scores.append(scores.mean())
    print('{} has a mean score {:.4f} based on {}'.format(name, scores.mean(), scores))    


# In[96]:


sorted_score = pd.Series(data=result_scores, index = names).sort_values(ascending=False)
ax = plt.subplot(111)
sorted_score.plot(kind='line', ax=ax, title='score order', figsize=(9, 6), colormap='cool')
ax.set_xticks(range(0, 9))
ax.set_xticklabels(sorted_score.index);
sns.despine()


# 我们就选择GBDT、RF、LR、AdaBoosting、Bagging、SVC作为我们的投票base learners。

# In[97]:


from sklearn.ensemble import VotingClassifier
names = ['LR', 'KNN', 'RF', 'GBDT', 'Bagging', 'AdaB']

lr = LogisticRegression(C=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=31, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=3, n_estimators=190, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=140, random_state=0)
abc = AdaBoostClassifier(learning_rate=1.3, n_estimators=150, random_state=0)

# 直接投票，票数多的获胜。
vc_hard = VotingClassifier(estimators=[('LR', lr), ('KNN', knn), ('GBDT', gbdt), ('Bagging', bagging), ('AdaB', abc)], voting='hard')
# 参数里说，soft更加适用于已经调制好的base learners，基于每个learner输出的概率。知乎文章里讲，Soft一般表现的更好。
vc_soft = VotingClassifier(estimators=[('LR', lr), ('KNN', knn), ('RF', rf), ('GBDT', gbdt), ('Bagging', bagging), ('AdaB', abc)], voting='soft')

# 'vc hard:', cross_val_score(vc_hard, X_dummied, Y, cv=5, scoring='roc_auc').mean(),\
'vc soft:', cross_val_score(vc_soft, X_train, Y_train, cv=5, scoring='roc_auc').mean()


# 我们运行VC Hard一直出错，表示没有decision_function可用这类的。
# 另外比起GBDT，VC soft提升了0.002。

# ## 8.4 Stacking
# ![](http://manish-m.com/wp-content/uploads/2012/11/StackingCropped.png)
# 看很多教程，Stacking 都是要把Test Data放进去，直接预测了。

# ## 8.4.1 Stacking
# ### 构建第一层Stacking

# In[98]:


n_train=X_train.shape[0]
n_test=X_test.shape[0]
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)


# 定义Stacking第一层函数，接收未Fit的estimator，然后返回预测后的train_y和test_predict

# In[99]:


def get_oof(clf, X, y, test_X):
    oof_train = np.zeros((n_train, ))
    oof_test_mean = np.zeros((n_test, ))
    # 5 is kf.split
    oof_test_single = np.empty((kf.get_n_splits(), n_test))
    for i, (train_index, val_index) in enumerate(kf.split(X,y)):
        kf_X_train = X.iloc[train_index]
        kf_y_train = y.iloc[train_index]
        kf_X_val = X.iloc[val_index]
        
        clf.fit(kf_X_train, kf_y_train)
        
        oof_train[val_index] = clf.predict(kf_X_val)
        oof_test_single[i,:] = clf.predict(test_X)
    # oof_test_single, 将生成一个5行*n_test列的predict value。那么mean(axis=0), 将对5行，每列的值进行求mean。然后reshape返回   
    oof_test_mean = oof_test_single.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)


# In[100]:


lr = LogisticRegression(C=0.3, random_state=0)
knn = KNeighborsClassifier(n_neighbors=10)
rf = RandomForestClassifier(max_depth=5, max_features=5, n_estimators=31, random_state=0)
gbdt = GradientBoostingClassifier(learning_rate=0.5, max_depth=2, max_features=3, n_estimators=190, random_state=0)
bagging = BaggingClassifier(lr, n_estimators=140, random_state=0)
abc = AdaBoostClassifier(learning_rate=1.3, n_estimators=150, random_state=0)

lr_train, lr_test = get_oof(lr, X_train, Y_train, X_test)
knn_train, knn_test = get_oof(knn, X_train, Y_train, X_test)
rf_train, rf_test = get_oof(rf, X_train, Y_train, X_test)
gbdt_train, gbdt_test=get_oof(gbdt, X_train, Y_train, X_test)
bagging_train, bagging_test = get_oof(bagging,X_train, Y_train, X_test)
abc_train, abc_test = get_oof(abc,X_train, Y_train, X_test)


# In[101]:


y_train_pred_stack = np.concatenate([lr_train, knn_train, rf_train, gbdt_train, bagging_train, abc_train], axis=1)
y_train_stack = Y_train.reset_index(drop=True)
y_test_pred_stack = np.concatenate([lr_test, knn_test, rf_test, gbdt_test, bagging_test, abc_test], axis=1)

y_train_pred_stack.shape, y_train_stack.shape, y_test_pred_stack.shape


# - y_train_pred_stack有891*6，这个6列分别是我们每个estimator在y_train上预测的值y值。
# - y_train_stack，是train上真实的y值。
# - y_test_pred_stack，是每个estimator在df_test上预测的y值，由于每个estimator将预测5次（K-Fold），我们需要将这5次的预测结果求平均，然后得出一个estimator的预测的y值，再把6个estimator再测试集上预测的y值组合起来。
# 
# 然后使用y_train_pred_stack(X_train)和y_train_stack(Y_train)再去做一个训练，进而得出最终的评分。然后再在y_test_pred_stack(X_test)预测最终结果。
# 
# ### 构建第二层Stacking

# In[102]:


# params = {'learning_rate':[0.001, 0.01, 0.1, 1], 'n_estimators':range(100, 250, 20), 'max_depth':[2, 5], 'max_features':range(1, 3)}
# clf = GradientBoostingClassifier(random_state=0)

# params = {'C':[0.05, 0.08, 0.1, 0.2, 0.3]}
# clf = LogisticRegression(random_state=0)

params = {'n_estimators':range(90, 150, 10)}
clf = RandomForestClassifier(random_state=0)

gscv_test= GridSearchCV(estimator=clf, param_grid=params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=False)
gscv_test.fit( y_train_pred_stack, y_train_stack)
gscv_test.best_score_, gscv_test.best_params_


# ### 预测

# In[103]:


y_pred = RandomForestClassifier(random_state=0, n_estimators=100).fit(y_train_pred_stack, y_train_stack).predict(y_test_pred_stack)


# 该Stacking的cross validation 评分最终只有0.8363。我把预测结果与实际结果进行一对比，发现我使用Stacking训练所预测的结果，仍然有138行的是预测失败的，并且预测失败的case几乎是每个estimator都给出同样的结果，这表明，我们需要构造某些feature来分类出这些预测失败的case。这个很像是我们在Misclassification Analysis中所作的事情，我在那个环节并没有做好，构造的feature也没有什么卵用。Andrew Ng说要肉眼看一百行的数据，，，我这肉眼还没修炼成火眼金睛看来。

# In[125]:


c = gscv_test.predict(y_train_pred_stack)

a = pd.DataFrame(y_train_pred_stack)
b = pd.concat([a, pd.Series(c), y_train_stack], axis=1)
b.columns = ['lr', 'knn', 'rf', 'gbdt', 'bagging', 'abc', 'predicted', 'Survived']
b[b.predicted != b.Survived]


# 以上结果，Stacking只有0.8363的评分。所以最终，我们还是选择考虑使用获得最高评分的vote_soft来预测并输出结果。
# # 9. Predict

# In[118]:


y_pred = vc_soft.fit(X_train, Y_train).predict(X_test)
result_df = pd.DataFrame({'PassengerId': X_test.index, 'Survived':y_pred}).set_index('PassengerId')
result_df.Survived = result_df.Survived.astype('int')
result_df.to_csv('predicted_survived.csv')


# In[123]:


pd.read_csv('predicted_survived.csv').head()


# In[124]:


pd.read_csv('../input/gender_submission.csv').head()


# # 格式一致，提交！
# # Right format，Submit！
