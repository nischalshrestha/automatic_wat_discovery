#!/usr/bin/env python
# coding: utf-8

# titanic data analysis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import numpy as np
import sklearn as  sk
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn import svm
import xgboost as xgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score,KFold,train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:



train_df.dtypes


# In[ ]:


train_df.head()


# In[ ]:


combin = pd.concat([train_df.drop("Survived",1),test_df])


# In[ ]:


combin.head(10)


# In[ ]:


train_df.describe()


# In[ ]:


#missing value
train_df.isnull().sum()


# In[ ]:


#check the test data missing value
test_df.info()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


surv = train_df[train_df["Survived"]==1]
nosurv = train_df[train_df["Survived"]==0]
print(len(surv)/train_df.shape[0],len(nosurv)/train_df.shape[0])


# In[ ]:


#plot dist about Age,直方图
sns.set(font_scale=3)
surv_col = "blue"
nosurv_col = "red"
plt.figure(figsize=(30,28))
plt.subplot(331)
dist_min  = int(train_df["Age"].min())
dist_max = int(train_df['Age'].max())
surv_axe = sns.distplot(surv['Age'].dropna(),bins=range(dist_min,dist_max+1,1),kde=False,color=surv_col)
#surv_axe.set_axis_label("survived Age")
surv_axe.set(xlabel="survived Age")
plt.subplot(332)
nosurv_axe = sns.distplot(nosurv["Age"].dropna(),bins=range(dist_min,dist_max+1,1),kde=False,color=nosurv_col)
#nosurv_axe.set_axis_label("no survived Age")
nosurv_axe.set(xlabel="no survived Age")
plt.subplot(333)
surv_axe = sns.distplot(surv['Age'].dropna(),bins=range(dist_min,dist_max+1,1),kde=False,color=surv_col)
nosurv_axe = sns.distplot(nosurv["Age"].dropna(),bins=range(dist_min,dist_max+1,1),kde=False,color=nosurv_col)


#每个类别存活人数的比率
oridinal_plot = {334:"Sex",335:"Pclass",336:"SibSp",337:"Parch",338:"Embarked"}
for splot,colum in oridinal_plot.items():
    plt.subplot(splot)
    sns.barplot(colum,"Survived",data=train_df)

    
plt.subplot(339)
#数值类型，取对数，起到归一化的作用
sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')


# In[ ]:




