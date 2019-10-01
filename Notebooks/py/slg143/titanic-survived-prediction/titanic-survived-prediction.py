#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np    #科学计算包，一矩阵为基础
import pandas as pd   #数据处理，数据分析
import seaborn as sns  #数据可视化
import matplotlib.pyplot as plt   #数据图表


# In[ ]:


#读取数据集
train=pd.read_csv('../input/train.csv')  
test=pd.read_csv('../input/test.csv')


# In[ ]:


train.info()  #查看具体信息，Age、Cabin、Embarked均有缺失值


# In[ ]:


train.describe()    #891个人中，存活率是38.38%，平均年龄是29.7


# In[ ]:


test.info()   # Age Fare Cabin 均存在缺失值


# In[ ]:


#补充Fare的缺失值
test.loc[(test.Fare.isnull()),:]
#求第3等级的票价均值
P3_fare_mean = test.Fare[test.Pclass == 3].mean()
P3_fare_mean

#将缺失值填补为平均值
test.loc[(test.Fare.isnull()), 'Fare'] = P3_fare_mean


# In[ ]:


#存活情况
train.Survived.value_counts().plot(kind="bar") #存活的人数和未成活的人数统计
plt.title("The number of surviving(1 for survived)")
plt.ylabel("Number of passengers")
print(train.Survived.value_counts())


# In[ ]:


#不同等级对是否获救的影响
Survived_0=train.Pclass[train.Survived==0].value_counts()
Survived_1=train.Pclass[train.Survived==1].value_counts()
df=pd.DataFrame({'Survived':Survived_1,'not Survived':Survived_0})
print(df)

df.plot(kind='bar',stacked=True)
plt.title('Survived by different class')
plt.ylabel('numbers')
plt.xlabel('class')

plt.show()

#1，2等级的存活率较高，第3等级的人数最多，但存活率最低，由此可以看出Pclass是结果的影响因素


# In[ ]:


#是否获救人员的性别比例
Survived_m=train.Survived[train.Sex=='male'].value_counts()
Survived_f=train.Survived[train.Sex=='female'].value_counts()
df=pd.DataFrame({'female':Survived_f,'male':Survived_m})
print(df)
df.plot(kind='bar',stacked=True) #堆叠图
plt.title('survived by different sex')
plt.ylabel('numbers')
plt.xlabel('sex')

#性别在是否获救中的占比
Survived_0=train.Sex[train.Survived==0].value_counts()
Survived_1=train.Sex[train.Survived==1].value_counts()
df=pd.DataFrame({'not Survived':Survived_0,'Survived':Survived_1})
print(df)
df.plot(kind='bar',stacked=True) #堆叠图
plt.title('survived in different sex')
plt.ylabel('numbers')
plt.xlabel('Survived')

plt.show()

#在获救的人当中，女性所占的比例较高，女性存活的概率占全部获救人员的68.%
#在所有的女性当中，有74.2%的人获救，所以性别的影响较大


# In[ ]:


#不同等级男女获救情况
fig=plt.figure(figsize=(20,5))
fig.set(alpha=0.65) #设定图表颜色参数
plt.title('survived by class and by sex')

ax1=fig.add_subplot(141)
S1=train.Survived[train.Sex=='female'][train.Pclass!=3].value_counts()
S1.plot(kind='bar',label="female highclass",color='#FA2479')
ax1.set_ylim(0,310)
ax1.set_xticklabels(['survived','not survived'],rotation=0)
ax1.legend(['female/high class'],loc='best')

ax2=fig.add_subplot(142)
S2=train.Survived[train.Sex=='male'][train.Pclass!=3].value_counts()
S2.plot(kind='bar',label='male highclass',color='pink')
ax2.set_ylim(0,310)
ax2.set_xticklabels(['not survived','survived'],rotation=0)
ax2.legend(['male/high class'],loc='best')

ax3=fig.add_subplot(143)
S3=train.Survived[train.Sex=='female'][train.Pclass==3].value_counts()
S3.plot(kind='bar',label='female lowclass',color='lightblue')
ax3.set_ylim(0,310)
ax3.set_xticklabels(['not survived','survived'],rotation=0)
ax3.legend(['female/low class'],loc='best')

ax4=fig.add_subplot(144)
S4=train.Survived[train.Sex=='male'][train.Pclass==3].value_counts()
S4.plot(kind='bar',label='male lowclass',color='steelblue')
ax4.set_ylim(0,310)
ax4.set_xticklabels(['not survived','survived'],rotation=0)
ax4.legend(['male/low class'],loc='best')

plt.show()

#等级和性别的影响 (1，2等级的女性存活率高)


# In[ ]:


#不同等级的小孩子存活情况
fig=plt.figure(figsize=(10,5))
fig.set(alpha=0.2)
plt.title('kid of different class')

axs1=fig.add_subplot(121)
kid_1=train.Survived[train.Age<=15][train.Pclass!=3].value_counts()
kid_1.plot(kind='bar',label='class high',color="blue")
ax1.set_xticklabels(['survived','not survived'],rotation=0)
ax1.legend(['class high'],loc='best')

axs2=fig.add_subplot(122)
kid_2=train.Survived[train.Age<=15][train.Pclass==3].value_counts()
kid_2.plot(kind='bar',label='class low',color='red')
ax2.set_xticklabels([' notsurvived','survived'],rotation=0)
ax2.legend(['class low'],loc='best')

plt.show()

print(kid_1)
print(kid_2)

#1，2等级的小孩几乎全部存活，第3等级存活率则较低(年龄和等级的影响)


# In[ ]:


#不同港口对是否获救的影响
fig=plt.figure()
fig.set(alpha=0.2)

Survived_0=train.Embarked[train.Survived==0].value_counts()
Survived_1=train.Embarked[train.Survived==1].value_counts()
df=pd.DataFrame({'Survived':Survived_1,'not Survived':Survived_0})
df.plot(kind='bar',stacked=True)
plt.title('survived by different harbors')
plt.xlabel('harbors')
plt.ylabel('numbers')

plt.show()
df
#各港口的获救比例没有特别大的差异，所以忽略 Embarked 因素 
train=train.drop(['Embarked'],axis=1)
test=test.drop(['Embarked'],axis=1)


# In[ ]:


#有无船舱号与乘客等级的关系
fig=plt.figure()
fig.set(alpha=0.2)

Pclass_cabin=train.Pclass[train.Cabin.notnull()].value_counts()
Pclass_nocabin=train.Pclass[train.Cabin.isnull()].value_counts()

df=pd.DataFrame({'have cabin':Pclass_cabin,'dont have cabin':Pclass_nocabin}).transpose()

df.plot(kind='bar',stacked=True)
plt.title('relationship between cabin and pclass')
plt.xlabel('cabin')
plt.ylabel('numbers')
plt.xticks(rotation=360)
plt.show()

#拥有船舱号的人绝大部分是第1等级的人，所以考虑等级因素就可以了，删除船舱号 Cabin
train=train.drop(['Cabin'],axis=1)
test=test.drop(['Cabin'],axis=1)


# In[ ]:


df_sibsp=pd.crosstab(train.SibSp,train.Survived).apply(lambda r: r/r.sum(), axis=1)
df_parch=pd.crosstab(train.Parch,train.Survived).apply(lambda r: r/r.sum(), axis=1)

train['family']=train['SibSp']+train['Parch']
df_family=pd.crosstab(train.family,train.Survived).apply(lambda r: r/r.sum(), axis=1)

fig=plt.figure(figsize=(15,5))

ax1=fig.add_subplot(131)
ax1.bar(df_sibsp.index,df_sibsp[1],color = 'darkred',alpha=0.7)
ax1.bar(df_sibsp.index,df_sibsp[0],color = 'darkred',bottom = df_sibsp[1],alpha=0.5)
ax1.grid(True)
ax1.set_title('percentange of survived')
ax1.set_xlabel('sibsp numbers')

ax2=fig.add_subplot(132)
ax2.bar(df_parch.index,df_parch[1],color = 'blue',alpha=0.7)
ax2.bar(df_parch.index,df_parch[0],color = 'blue',bottom = df_parch[1],alpha=0.5)
ax2.grid(True)
ax2.set_title('percentange of survived')
ax2.set_xlabel('parch numbers')

ax3=fig.add_subplot(133)
ax3.bar(df_family.index,df_family[1],color = 'green',alpha=0.7)
ax3.bar(df_family.index,df_family[0],color = 'green',bottom = df_family[1],alpha=0.5)
ax3.grid(True)
ax3.set_title('percentange of survived')
ax3.set_xlabel('family numbers')

#有1-3个家庭成员的存活率较高


# In[ ]:


# 各等级乘客的年龄分布

train.Age[train.Pclass==1].plot(kind="kde")
train.Age[train.Pclass==2].plot(kind="kde")
train.Age[train.Pclass==3].plot(kind="kde") # 关于年龄的不同等级的密度分布图
plt.xlabel("age")
plt.ylabel("density")  
plt.title("passengers's age by Pclass ")
plt.legend(("1","2","3"),loc="best") #设置图例

plt.show()


# In[ ]:


#提取姓名中的称谓
train['Title']=train.Name.str.extract('([A-Za-z]+)\.') 

#train.Title.value_counts()

train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Mr'],inplace=True)
train=train.drop(['Name'],axis=1)
train.head()


# In[ ]:


test['Title']=test.Name.str.extract('([A-Za-z]+)\.')
#test.Title.value_counts()
test['Title'].replace(['Dr','Col','Rev','Dona'],['Mr','Mr','Mr','Mrs'],inplace=True)

test['family']=test['SibSp']+test['Parch']

test=test.drop(['Name'],axis=1)

test.head()


# In[ ]:


#利用线性回归模型填补年龄的缺失值
from sklearn.linear_model import LinearRegression

def set_missing_age(df):
    age_df=df[['Age','Fare', 'Parch', 'SibSp', 'Pclass','family']]
    
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    
    # y即目标年龄
    y = known_age[:, 0] #取所有行的第0列

    # X即特征属性值
    x = known_age[:, 1:] #所有行以及从第1列到最后一列
    
    lr = LinearRegression(fit_intercept = True,normalize = False,copy_X = True,n_jobs = 1 )
    
    lr.fit(x,y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = lr.predict(unknown_age[:, 1:]) #切片

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges  #loc基于标签的，按标签进行选择

    return df, lr

Mr = train[train.Title=='Mr']
Mrs = train[train.Title=='Mrs']
Miss = train[train.Title=='Miss']
Master = train[train.Title=='Master']
Other = train[train.Title=='Other']


Mr,lr = set_missing_age(Mr)
Mrs,lr = set_missing_age(Mrs)
Miss,lr = set_missing_age(Miss)
Master,lr = set_missing_age(Master)

#合并数据集
train=pd.concat([Mr,Mrs,Miss,Master,Other],axis=0).sort_index(ascending=True)
train.head()



# In[ ]:


Mr1 = test[test.Title=='Mr']
Mrs1 = test[test.Title=='Mrs']
Miss1 = test[test.Title=='Miss']
Master1 =test[test.Title=='Master']


Mr1,lr = set_missing_age(Mr1)
Mrs1,lr = set_missing_age(Mrs1)
Miss1,lr = set_missing_age(Miss1)
Master1,lr = set_missing_age(Master1)

#合并数据集
test=pd.concat([Mr1,Mrs1,Miss1,Master1],axis=0).sort_index(ascending=True)
test.head()


# In[ ]:


#将分类变量转换为虚拟/指示符变量
dummies_Pclass = pd.get_dummies(train['Pclass'],prefix='Pclass')  #prefix：前缀
dummies_Sex=pd.get_dummies(train['Sex'],prefix='Sex')
dummies_Title=pd.get_dummies(train['Title'],prefix='Title')

train=pd.concat([train,dummies_Pclass,dummies_Sex,dummies_Title],axis=1)
#沿着特定轴连接pandas对象，并沿着其他轴选择设置逻辑, 要连接的轴0:行，1:列
train.drop(['Pclass','Sex','Ticket','Title'],axis=1,inplace=True)

train.head()


# In[ ]:


#对测试集做相同的处理
dummies_Pclass = pd.get_dummies(test['Pclass'],prefix='Pclass')  #prefix：前缀
dummies_Sex=pd.get_dummies(test['Sex'],prefix='Sex')
dummies_Title=pd.get_dummies(test['Title'],prefix='Title')

test_df=pd.concat([test,dummies_Pclass,dummies_Sex,dummies_Title],axis=1)
#沿着特定轴连接pandas对象，并沿着其他轴选择设置逻辑, 要连接的轴0:行，1:列
test_df.drop(['Pclass','Sex','Ticket','Title'],axis=1,inplace=True)

test_df.head()


# In[ ]:


from sklearn.linear_model import LogisticRegression

train_df=train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Sex_.*|Pclass_.*|Title_.*')
#filter 过滤序列

train_np=train_df.as_matrix() # 将数据转化成矩阵
#train_np

y=train_np[:,0]
x=train_np[:,1::]

#逻辑回归模型
lr=LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
lr.fit(x,y)


# In[ ]:


#调用训练好的模型
test_f= test_df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Sex_.*|Pclass_.*|Title_.*')
predictions = lr.predict(test_f)
result = pd.DataFrame({'PassengerId':test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})


# In[ ]:


from sklearn import cross_validation
lr=LogisticRegression(C=1.0,penalty='l1',tol=1e-6)
all_data = train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.|Sex_.*|Pclass_.*|Title_.*')
x = all_data.values[:,1:]
y = all_data.values[:,0]
print (cross_validation.cross_val_score(lr,x,y,cv=5)) #cv表示不同的交叉验证方法


# In[ ]:


result.to_csv("survived_predictions_1.csv", index=False)


# In[ ]:




