#!/usr/bin/env python
# coding: utf-8

# In machine learning, we can use the random forest algorithm to judge the relative importance of each feature, which will help us to make feature selection. I used the random forest algorithm in sklearn to judge the importance of each feature and visualize the data, which allows us to understand the importance of each feature more intuitively. First let's import some necessary libraries and do some initial setup：

# In[ ]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().magic(u'matplotlib inline')
import pandas as pd
import seaborn as sns
sns.set(style="white",color_codes=True)
plt.rcParams['figure.figsize'] = (15,9.27)
# Set the font set of the latex code to computer modern
matplotlib.rcParams['mathtext.fontset'] = "cm"


# Next we import the classic Titanic shipwreck data and perform some necessary feature engineering. The final result is as follows:

# In[ ]:


df = pd.read_csv('../input/train.csv')

titanic = df.drop('Name',axis=1)

titanic.drop(['Ticket','Cabin','Embarked','PassengerId'],axis=1,inplace=True)

def encode(x):
    if x == 'male':
        return 1
    else:
        return 0

titanic['ismale'] = titanic.Sex.apply(encode)
titanic.drop('Sex',axis=1,inplace=True)
titanic.dropna(inplace=True)

titanic.head()


# Then we define a function that uses the sklearn, seaborn, and matplotlib libraries to visualize the importance of features：

# In[ ]:


def feature_importance_plot(df,target):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    y = df[target]
    x = df.drop(target,axis=1)
    model.fit(x,y)
    res_df = pd.DataFrame({'feature':x.columns,'importance':model.feature_importances_})
    res = res_df.sort_values('importance',ascending=False)
    res['cum_importance'] = res.importance.cumsum()
    plt.subplot(211)
    sns.barplot(res.feature,res.importance)
    plt.subplot(212)
    plt.plot(np.arange(1,res.shape[0]+1),res.cum_importance,linewidth=2)
    return(res)


# we can find out: 

# In[ ]:


feature_importance_plot(titanic,'Survived')


# Obviously, in the Titanic shipwreck data, the three characteristics of age, fare and whether it is male are more important, accounting for 82.14% of the importance of all features.

# Similarly, let's take a look at the classic iris data set.

# In[ ]:


iris = sns.load_dataset('iris')

feature_importance_plot(iris,'species')


# Obviously, the two features of petal_width and petal_lenght are more important.

# In[ ]:




