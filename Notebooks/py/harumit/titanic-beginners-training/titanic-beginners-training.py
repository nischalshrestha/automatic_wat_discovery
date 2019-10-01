#!/usr/bin/env python
# coding: utf-8

# ## Titanic
# - 1 データ取り込み
# - 2 前処理
# - 3 可視化／基礎分析
# - 4 学習　モデル構築
# - 5 検証
# - 6 予測

# In[1]:


import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[3]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[4]:


train.head()


# In[5]:


train.tail()


# In[6]:


train.info()


# ## 欠損値確認

# In[7]:


train.isnull().sum()


# In[8]:


train.describe()


# ## 前処理

# In[9]:


train.Age = train.Age.fillna(train.Age.median()) ##mean()?
train.Sex = train.Sex.replace(['male', 'female'], [0, 1])
train.Embarked = train.Embarked.fillna("S")
train.Embarked = train.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])


# In[12]:


train.info()


# In[11]:


train.describe()


# ### memo
# - 敬称を利用した処理を追加する
# - Ticket, Cabin の利用
# - 家族、一緒に乗船した人がいるかなど

# # 可視化

# In[13]:


train.corr()


# In[14]:


train.corr().style.background_gradient().format('{:.2f}')


# ### memo
# - 可視化手法の検討？？

# # モデル構築

# In[15]:


from sklearn.model_selection import train_test_split

train = train.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
train_X = train.drop('Survived', axis=1)
train_y = train.Survived
(train_X, test_X ,train_y, test_y) = train_test_split(train_X, train_y, test_size = 0.3, random_state = 0)


# ### decisionTree

# In[16]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)


# In[17]:


from sklearn.metrics import (roc_curve, auc, accuracy_score)

pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
auc(fpr, tpr)
accuracy_score(pred, test_y)


# ### RandomForest

# In[18]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf = clf.fit(train_X, train_y)
pred = clf.predict(test_X)
fpr, tpr, thresholds = roc_curve(test_y, pred, pos_label=1)
auc(fpr, tpr)
accuracy_score(pred, test_y)


# RandomForestを採用

# ### GridSearch

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters = {
        'max_depth'         : [5, 10, 15, 20, 25],
        'random_state'      : [0],
        'n_estimators'      : [100, 150, 200, 250, 300, 400],
        'min_samples_split' : [15, 20, 25, 30, 35, 40, 50],
        'min_samples_leaf'  : [1, 2, 3],
        'bootstrap'         : [False],
        'criterion'         : ["entropy"]
}

gsc = GridSearchCV(RandomForestClassifier(), parameters,cv=3)
gsc.fit(train_X, train_y)

model = gsc.best_estimator_


# In[ ]:


gsc.best_params_


# ## test.csvの予測

# In[17]:


test.head()


# In[18]:


test.tail()


# In[19]:


test.info()


# In[20]:


test.isnull().sum()


# In[21]:


test.Age = test.Age.fillna(test.Age.median()) ##mean()?
test.Sex = test.Sex.replace(['male', 'female'], [0, 1])
test.Embarked = test.Embarked.fillna("S")
test.Embarked = test.Embarked.replace(['C', 'S', 'Q'], [0, 1, 2])
test.Fare = test.Fare.fillna(test.Fare.median()) ##mean()?


# In[22]:


test.isnull().sum()


# In[23]:


test.describe()


# In[25]:


test_data = test.drop(['Cabin','Name','PassengerId','Ticket'],axis=1)
pred3 = model.predict(test_data)


# In[26]:


pred3


# In[29]:


submission = pd.DataFrame({
        "PassengerId": test.PassengerId,
        "Survived": pred3
    })


# In[30]:


submission.to_csv('titanic_pre.csv', index=False)


# ## 1回目提出
# - Score:0.69377 上位94％

# ### memo
# - 特徴量エンジニアリング実施
# - パラメータの調整
# - 手法を変える

# In[ ]:




