#!/usr/bin/env python
# coding: utf-8

# # Titanic
# https://www.kaggle.com/c/titanic

# 加入numpy, pandas, matplot等库

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt


# 读入数据

# In[ ]:


dataset=pd.read_csv('../input/train.csv')
testset=pd.read_csv('../input/test.csv')


# In[ ]:


dataset.columns


# 0, PassengerId：乘客的数字id
# 
# 1, Survived：幸存(1)、死亡(0)
# 
# 2, Pclass：乘客船层—1st = Upper，2nd = Middle， 3rd = Lower
# 
# 3, Name：名字。
# 
# 4, Sex：性别
# 
# 5, Age：年龄
# 
# 6, SibSp：兄弟姐妹和配偶的数量。
# 
# 7, Parch：父母和孩子的数量。
# 
# 8, Ticket：船票号码。
# 
# 9, Fare：船票价钱。
# 
# 10, Cabin：船舱。
# 
# 11, Embarked：从哪个地方登上泰坦尼克号。 C = Cherbourg, Q = Queenstown, S = Southampton

# # 查看读入数据

# In[ ]:


dataset.head()


# In[ ]:


print(dataset.dtypes)


# In[ ]:


print(dataset.describe())
print(dataset.corr().loc['Survived',])


# In[ ]:





# 从上面数据发现两个有意思的事情
# 
# 1. 数据有NULL元素
# 2. 数据

# # 仔细观察数据

# 查看 cabin： 似乎只有第一个字母有用

# In[ ]:


#too many missing values in cabin, not a good feature
cabin_set=dataset[['Survived','Cabin']].copy(deep=True)
cabin_set['Cabin_new']=cabin_set['Cabin'].str[0]

cabin=pd.DataFrame({'Survived':cabin_set.Cabin_new[cabin_set.Survived==1].value_counts(dropna=True),
                    'Non_survived':cabin_set.Cabin_new[cabin_set.Survived==0].value_counts(dropna=True)});
cabin['%']=cabin['Survived']/(cabin['Survived']+cabin['Non_survived'])*100;
print(cabin)
cabin.plot(y="%", kind='bar')
plt.title("Survival rate by cabin")
plt.xlabel('cabin')
plt.ylabel('Survival Rate(%)')
plt.show()


# 观察性别 

# In[ ]:


sur=dataset.Sex[dataset.Survived==1].value_counts()
nsur=dataset.Sex[dataset.Survived==0].value_counts()
sex_df=pd.DataFrame({'Survived':sur,'Non_Survived':nsur})
sex_df['%']=sex_df.Survived/(sex_df.Survived+sex_df.Non_Survived)*100
print(sex_df)
sex_df.plot(y="%",kind="bar")
plt.title("Survial Rate by Gender")
plt.ylabel("Survival Rate (%)")
plt.show()


# 看看年龄： 按年龄段看跟存活率的关系 

# In[ ]:


dataset['Age'].hist()
plt.show()
#dataset[dataset.Survived==1]['Age'].hist()
#plt.show()
#dataset[dataset.Survived==0]['Age'].hist()
#plt.show()

#surival rate by age group
#duplicate the data set and drop nan
Age_set=dataset[['Survived','Age']].copy(deep=True)
Age_set=Age_set.dropna(axis=0,how="any")

#linear cut the ages into 7 age groups
tmp=pd.factorize(pd.cut(Age_set['Age'],pd.IntervalIndex.from_breaks([0,18,35,55,100]), duplicates='drop'), sort=True)
Age_set['Age_Grp']=tmp[0]

#get suvival rate by age group data frame
sur_by_age=pd.DataFrame({'S':Age_set.Age_Grp[Age_set.Survived==1].value_counts(dropna=True), 
                         'NS':Age_set.Age_Grp[Age_set.Survived==0].value_counts(dropna=True)})

sur_by_age['Age_Grp']=tmp[1].categories[sur_by_age.index]
sur_by_age=sur_by_age.fillna(0)
sur_by_age=sur_by_age.sort_index()
sur_by_age['%']=sur_by_age.S/(sur_by_age.S+sur_by_age.NS)*100

#print and plot
print(sur_by_age)
sur_by_age.plot(x='Age_Grp', y='%', kind='bar')
plt.title("Survival Rate by Age")
plt.ylabel("Survival Rate (%)")
plt.xlabel("Age Group")
plt.show();


# 看看船票价钱

# In[ ]:


dataset['Fare'].hist()
plt.show()
#dataset.Fare[dataset.Survived==1].hist()
#plt.show()
#dataset.Fare[dataset.Survived==0].hist()
#plt.show()


Fare_Set=dataset[['Survived', 'Fare']].copy()

#pd.IntervalIndex.from_breaks([0, 1, 2, 3])
tmp=pd.factorize(pd.cut(Fare_Set['Fare'],pd.IntervalIndex.from_breaks([-1,100,300,1000])), sort=True)
Fare_Set['Fare_Grp']=tmp[0]

sur_by_fare=pd.DataFrame({'S':Fare_Set.Fare_Grp[Fare_Set.Survived==1].value_counts(dropna=True),
                         'NS':Fare_Set.Fare_Grp[Fare_Set.Survived==0].value_counts(dropna=True)})
sur_by_fare['Fare_Grp']=tmp[1].categories[sur_by_fare.index]
sur_by_fare=sur_by_fare.fillna(0)
sur_by_fare['Survival Rate %']=sur_by_fare.S/(sur_by_fare.NS+sur_by_fare.S)*100
print(sur_by_fare)
sur_by_fare.plot(x='Fare_Grp',y='Survival Rate %', kind='bar')
plt.ylabel('Survival Rate %')
plt.show()



# 观察乘客舱层

# In[ ]:


#survival Rate By class
sur_by_class=pd.DataFrame({'S': dataset.Pclass[dataset.Survived==1].value_counts(dropna=True),
                          'NS':dataset.Pclass[dataset.Survived==0].value_counts(dropna=True)});
sur_by_class['%']=sur_by_class.S/(sur_by_class.S+sur_by_class.NS)*100
print(sur_by_class)
sur_by_class.plot(y='%', kind='bar')
plt.xlabel("Pclass")
plt.ylabel("Survival Rate %")
plt.show()


# 观察登船地点

# In[ ]:


#survival rate by embark location
sur_by_emb=pd.DataFrame({'S': dataset.Embarked[dataset.Survived==1].value_counts(dropna=True),
                          'NS':dataset.Embarked[dataset.Survived==0].value_counts(dropna=True)});
sur_by_emb['%']=sur_by_emb.S/(sur_by_emb.S+sur_by_emb.NS)*100
print(sur_by_emb)
sur_by_emb.plot(y='%', kind='bar')
plt.xlabel("Embark Location")
plt.ylabel("Survival Rate %")
plt.show()


# Check SibSp and Parch

# In[ ]:


#checking to see if any relationship with sibsp and parch
sibsq_set=pd.DataFrame({'SibSp=0':dataset.Survived[dataset.SibSp==0].value_counts(dropna=True),
                       'SibSp>0':dataset.Survived[dataset.SibSp>0].value_counts(dropna=True)});
sibsq_set=sibsq_set.T
sibsq_set.columns=['S','NS']
sibsq_set['%']=sibsq_set.S/(sibsq_set.NS+sibsq_set.S)*100
print(sibsq_set)

sibsq_set.plot(y='%', kind='bar')
plt.title('survival rate by sibsp #')
plt.ylabel('survival rate %')
plt.show()


parch_set=pd.DataFrame({'Parch=0':dataset.Survived[dataset.Parch==0].value_counts(dropna=True),
                       'Parch>0':dataset.Survived[dataset.Parch>0].value_counts(dropna=True)});
parch_set=parch_set.T
parch_set.columns=['S','NS']
parch_set['%']=parch_set.S/(parch_set.NS+parch_set.S)*100
print(parch_set)
parch_set.plot(y='%', kind='bar')
plt.title('survival rate by parch #')
plt.ylabel('survival rate %')
plt.show()


# # 保留下有效数据
# pclass, sex, age, fare, embarked

# In[ ]:


#using sibsp and parch does not seem to have improvements
#data=dataset[['Pclass','Sex','Age','Fare','Embarked','SibSp','Parch']]
#test_data=testset[['Pclass','Sex','Age','Fare','Embarked','SibSp','Parch']]

data=dataset[['Pclass','Sex','Age','Fare','Embarked']]
test_data=testset[['Pclass','Sex','Age','Fare','Embarked']]


# # 分离label 和 训练数据

# In[ ]:


label=dataset['Survived']


#大概看一下有多少NAN
print(data.shape)
print(data.isnull().sum(axis=0))
print(test_data.isnull().sum(axis=0))


# 处理空数据

# In[ ]:


def fill_NAN(data):
    data_copy=data.copy(deep=True)
    data_copy['Age']=data_copy['Age'].fillna(data_copy['Age'].median())
    data_copy['Fare']=data_copy['Fare'].fillna(data_copy['Fare'].median())
    data_copy['Pclass']=data_copy['Pclass'].fillna(data_copy['Pclass'].median())
    data_copy['Sex']=data_copy['Sex'].fillna('male')
    data_copy['Embarked']=data_copy['Embarked'].fillna('S')
    return data_copy
data_no_nan=fill_NAN(data)
testdata_no_nan=fill_NAN(test_data)
print(data_no_nan.isnull().sum(axis=0))
print(testdata_no_nan.isnull().sum(axis=0))
    


# 处理Sex 

# In[ ]:


def transfer_sex(data):
    data_copy=data.copy(deep=True)
    data_copy.loc[data_copy['Sex']=='female','Sex']=0
    data_copy.loc[data_copy['Sex']=='male','Sex']=1
    return data_copy
data_after_sex=transfer_sex(data_no_nan)
testdata_after_sex=transfer_sex(testdata_no_nan)


# 处理Embarked

# In[ ]:


def transfer_em(data):
    data_copy=data.copy(deep=True)
    data_copy.loc[data_copy['Embarked']=='S','Embarked']=0
    data_copy.loc[data_copy['Embarked']=='C','Embarked']=1
    data_copy.loc[data_copy['Embarked']=='Q','Embarked']=2
    return data_copy
data_after_em=transfer_em(data_after_sex)
testdata_after_em=transfer_em(testdata_after_sex)


# 归一化 所有的 Feature

# In[ ]:


def normalize(data):
    data_copy=data.copy(deep=True)
    data_copy['Fare']=(data_copy['Fare']-data_copy['Fare'].mean())/data_copy['Fare'].std()
    data_copy['Age']=(data_copy['Age']-data_copy['Age'].mean())/data_copy['Age'].std()
    data_copy['Pclass']=(data_copy['Pclass']-data_copy['Pclass'].mean())/data_copy['Pclass'].std()
    data_copy['Sex']=(data_copy['Sex']-data_copy['Sex'].mean())/data_copy['Sex'].std()
    data_copy['Embarked']=(data_copy['Embarked']-data_copy['Embarked'].mean())/data_copy['Embarked'].std()
    return data_copy
data_after_norm=normalize(data_after_em)
testdata_after_norm=normalize(testdata_after_em)




# 给每个维度加个权重，但似乎好像影响不多，还是加法不对？ （目前是按每个维度 跟Survival 的Corr加的）
# 

# In[ ]:


corr_val=pd.concat([label,data_after_norm],axis=1).corr().iloc[0,:]
print(corr_val)

#applying a weight to each feature that is proportional to the corr to survival
# not sure if this is something correct
def apply_weights(data, corr_val):
    total_val=corr_val.abs().sum(axis=0)-1
    data_copy=data.copy(deep=True)
    data_copy['Fare']=data_copy['Fare']*np.abs(corr_val['Fare'])/total_val
    data_copy['Age']=data_copy['Age']*np.abs(corr_val['Age'])/total_val
    data_copy['Pclass']=data_copy['Pclass']*np.abs(corr_val['Pclass'])/total_val
    data_copy['Sex']=data_copy['Sex']*np.abs(corr_val['Sex'])/total_val
    data_copy['Embarked']=data_copy['Embarked']*np.abs(corr_val['Embarked'])/total_val
    return data_copy

data_after_weight=apply_weights(data_after_norm,corr_val)
testdata_after_weight=apply_weights(testdata_after_norm,corr_val)
    


# 利用KNN训练数据： 这个题似乎改变 KNN 的 distance metrics， 加不加 voting weights 都不大有影响。

# In[ ]:



from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from tqdm import tqdm;

krange=range(1,200)
k_scores=[]
for k in tqdm(krange):
    #tried different distance metrics, does not seem to have much effect
    knn_classifier=KNeighborsClassifier(n_neighbors=k,weights='distance');
    cv_results = cross_validate(knn_classifier, data_after_weight, label, return_train_score=False, cv=7)
    k_scores.append(cv_results['test_score'].mean())
        
plt.plot(krange, k_scores)
print("best k=",np.array(k_scores).argsort()[-1]+1)
print("best_accuracy: ", max(k_scores))


# In[ ]:


# 预测测试实验数据
kbest=68
knn_clf=KNeighborsClassifier(n_neighbors=kbest)
knn_clf.fit(data_after_weight, label)
result=knn_clf.predict(testdata_after_weight)


# 打印输出

# In[ ]:


# kaggle submission score=0.77511
df=pd.DataFrame({'PassengerId':testset['PassengerId'], 'Survived':result})
df.to_csv('submission.csv',header=True, index=False)


# In[ ]:




