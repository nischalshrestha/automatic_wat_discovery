#!/usr/bin/env python
# coding: utf-8

# In[105]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[106]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.shape, test.shape


# In[107]:


train.head(10)


# In[108]:


#sum(train.Survived) / train.shape[0]
train.Survived.value_counts(normalize=True)


# In[109]:


f, ax = plt.subplots(1,2,figsize=(12,5))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# In[110]:


train.Survived.groupby(train.Pclass).agg(['mean', 'count'])


# In[111]:


null_count = train.append(test, sort=True).drop("Survived", axis=1).isnull().sum()
null_count[null_count > 0]


# In[112]:


train.Cabin.dropna().values[:30]


# In[113]:


train.Cabin.str[0].unique()


# In[114]:


fare_median = train.Fare.mean()
fare_median


# In[115]:


embarked_mode = train.Embarked.mode()[0]
embarked_mode


# In[116]:


fill_dict = {'Fare': fare_median, 'Embarked': embarked_mode, 'Cabin': 'Z'}
train.fillna(fill_dict, inplace=True)
test.fillna(fill_dict, inplace=True)


# In[117]:


dataset = train.append(test, sort=True)
dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
pd.crosstab(dataset.Title, dataset.Sex).join(dataset.groupby('Title').mean()['Age'])


# In[118]:


def put_title(data):
    data['Title'] = data.Name.str.extract('([A-Za-z]+)\.', expand=False)
    title_map = {
        'Mlle':'Miss', 
        'Mme':'Miss', 
        'Ms':'Miss',
        'Dr':'Mr', 
        'Major':'Mr', 
        'Capt':'Mr', 
        'Sir':'Mr', 
        'Don':'Mr',
        'Lady':'Mrs', 
        'Countess':'Mrs', 
        'Dona':'Mrs',
        'Jonkheer':'Other', 
        'Col':'Other', 
        'Rev':'Other'}
    data['Title'].replace(title_map, inplace=True)

put_title(train)
put_title(test)


# In[119]:


age_mean = train.append(test, sort=True).groupby('Title')['Age'].mean()
age_mean


# In[120]:


for i in age_mean.index:
    train.loc[train.Age.isnull() & (train.Title == i), 'Age'] = age_mean[i]
    test.loc[test.Age.isnull() & (test.Title == i), 'Age'] = age_mean[i]


# In[121]:


pd.crosstab(train.Cabin.str[0], train.Survived, margins=True)


# In[122]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[123]:


def transDataFrame(data):
    data = data.copy()
    data.Embarked = data.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
    data.Sex = data.Sex.replace(['male', 'female'], [0, 1])
    data.Cabin = le.fit_transform(data.Cabin.str[0])
    data['Title_Code'] = le.fit_transform(data['Title'])
    
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1
    data.loc[(data.SibSp + data.Parch) > 1, 'IsAlone'] = 0
    
    data['Fare_Bin'] = pd.qcut(data['Fare'], 4)
    data['Fare_Code'] = le.fit_transform(data['Fare_Bin'])
    data['Age_Bin'] = pd.cut(data['Age'].astype(int), 5)
    data['Age_Code'] = le.fit_transform(data['Age_Bin'])
    data.drop(['Fare_Bin', 'Age_Bin'], axis=1, inplace=True)
    
    data.drop(['Name','Title','Ticket','Parch','SibSp','Age','Fare'], axis=1, inplace=True)
    return data


# In[124]:


train_1 = transDataFrame(train).drop('PassengerId', axis=1)
train_1.describe()


# In[125]:


corr = train_1.corr()

plt.figure(figsize=(15,6))
plt.title('Correlation of Features for Train Set')
sns.heatmap(corr, vmax=1.0, annot=True, cmap='coolwarm')
plt.show()


# In[126]:


pd.DataFrame(corr.Survived.abs().sort_values(ascending=False)).T


# In[128]:


for x in ['Sex', 'Pclass', 'Cabin', 'Embarked', 'IsAlone', 'Title_Code', 'FamilySize', 'Fare_Code', 'Age_Code']:
        print('Survival Correlation by:', x)
        print(train_1[[x, 'Survived']].groupby(x, as_index=False).mean())
        print('-'*10, '\n')


# In[129]:


a = sns.pairplot(train_1[[u'Survived', u'Pclass', u'Sex', u'Age_Code', u'IsAlone', u'Fare_Code', u'Embarked', u'FamilySize', u'Title_Code']], 
                 hue='Survived', size=1.3, palette='seismic')
a.set(xticklabels=[])


# In[95]:


#pd.crosstab(train_1.Age // 5 * 5, train_1.Survived).plot.area(stacked=False)
a = sns.FacetGrid(train, hue='Survived', aspect=4)
a.map(sns.kdeplot, 'Age', shade=True)
a.set(xlim=(0 , train['Age'].max()))
a.add_legend()


# In[96]:


_train = transDataFrame(train)
_train.drop(['PassengerId'], axis=1, inplace=True)
_train.head()


# In[97]:


_train.isnull().any()


# In[98]:


_test = transDataFrame(test)
_test.head()


# In[99]:


_test.isnull().any()


# In[100]:


_train.head()


# In[101]:


X_train = _train.drop("Survived", axis=1)
Y_train = _train["Survived"]
X_test  = _test.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape


# In[130]:


from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(criterion='entropy', max_leaf_nodes=12, max_depth=5, n_estimators=100, random_state=0)
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

#import xgboost as xgb
#clf = xgb.XGBClassifier(n_estimators=25, max_depth=12, learning_rate=0.1, subsample=1, colsample_bytree=1, random_state=42)

clf.fit(X_train, Y_train)
result = clf.predict(X_test)


# In[103]:


submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": result
})
submission.to_csv("submission.csv", index=False)


# In[104]:


submission.head(20)


# In[ ]:




