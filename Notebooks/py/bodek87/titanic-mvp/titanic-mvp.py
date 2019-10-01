#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_raw = df_train.copy
df_train.head()


# In[3]:


df_train.describe()


# In[4]:


df_train.isnull().sum().sort_values(ascending = False)


# In[5]:


df_train.drop('Cabin',axis =1, inplace=True)
df_test.drop('Cabin',axis =1, inplace=True)
df_train.head()


# In[6]:


df_train.drop('PassengerId',axis =1, inplace=True)
df_test.drop('PassengerId',axis =1, inplace=True)

df_train.head()


# In[7]:


def name_split(row):
    try: return row.split(",")[0]
    except:return None


# In[8]:


df_train["surname"] = df_train["Name"].apply(lambda x: name_split(x))
df_test["surname"] = df_test["Name"].apply(lambda x: name_split(x))
df_train.head()


# In[9]:


df_test.head()


# In[10]:


df_train["title"] = df_train["Name"].apply(lambda x: x.split(",")[1].split(".")[0])
df_test["title"] = df_test["Name"].apply(lambda x: x.split(",")[1].split(".")[0])

df_train.head()


# In[11]:


df_train.surname.value_counts()


# In[12]:


df_train[df_train["surname"]=="Sage"]


# In[13]:


df_train[df_train["surname"]=="Fortune"]


# In[14]:


#df_train[df_train["Cabin"]=="C23 C25 C27"]


# In[15]:


df_train[df_train["surname"]=="Panula"]


# * parch = # of parents / children aboard the Titanic
# * sibsp = # of siblings / spouses aboard the Titanic
# 
# >sibsp: The dataset defines family relations in this way...
# Sibling = brother, sister, stepbrother, stepsister
# Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# >parch: The dataset defines family relations in this way...
# Parent = mother, father
# Child = daughter, son, stepdaughter, stepson
# Some children travelled only with a nanny, therefore parch=0 for them.

# In[16]:


df_train.drop('Ticket',axis =1, inplace=True)
df_train.drop('Name',axis =1, inplace=True)
df_test.drop('Ticket',axis =1, inplace=True)
df_test.drop('Name',axis =1, inplace=True)



df_train.head()


# In[17]:


df_train.isnull().sum()


# In[18]:


df_train[df_train["Embarked"].isnull()].index.tolist()


# In[19]:


df_train.loc[61]


# In[20]:


df_train.loc[829]


# From which port fare was about 80? Barchart sns

# In[21]:


sns.barplot(df_train["Pclass"],df_train["Fare"])


# In[22]:


sns.barplot(df_train["Embarked"],df_train["Fare"])


# DataFrame.set_value(index, col, value, takeable=False)

# In[23]:


df_train.describe()


# In[24]:


df_test[df_test["Fare"].isnull()].index.tolist()


# In[25]:


df_test.isnull().sum()


# In[26]:


df_test.loc[152]


# In[27]:


df_test.set_value(152,"Fare","60")


# In[28]:


df_train.set_value(61,"Embarked","C")
df_train.set_value(829,"Embarked","C")
df_train.isnull().sum()


# In[29]:


df_test.isnull().sum()


# In[30]:


df_train.describe()


# In[31]:


df_train.loc[df_train["Age"].isnull()]


# In[32]:


df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean())


# In[33]:


df_train.describe()


# class pandas.Categorical(values, categories=None, ordered=None, dtype=None, fastpath=False)

# In[34]:


df_train['Sex'] = pd.Categorical(df_train["Sex"])
df_train['Embarked'] = pd.Categorical(df_train["Embarked"])
df_train['Pclass'] = pd.Categorical(df_train["Pclass"])

df_test['Sex'] = pd.Categorical(df_test["Sex"])
df_test['Embarked'] = pd.Categorical(df_test["Embarked"])
df_test['Pclass'] = pd.Categorical(df_test["Pclass"])


# pandas.get_dummies(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)

# In[35]:


#df_train = pd.get_dummies(df_train["Embarked"], drop_first=True)
df_train.head()


# In[36]:


df_train.columns.tolist()


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split


# In[38]:


df_train.head()


# In[39]:


logreg = LogisticRegression()


# In[40]:


y = df_train.Survived.copy() 


# In[41]:


df_train = df_train.drop("Survived", axis=1)


# In[42]:


X = df_train


# In[43]:


X.head()


# In[44]:


X.dropna()


# In[45]:


df = pd.concat([df_train,pd.get_dummies(df_train.title)], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Embarked)], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Sex)], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Pclass)], axis = 1)
df = df.drop("Embarked",axis = 1)
df= df.drop("title",axis = 1)
df =df.drop("surname",axis = 1)
df = df.drop("Sex", axis = 1)
df = df.drop("Pclass", axis = 1)


# In[46]:


df.head()


# In[47]:


X = df


# In[48]:


X.dtypes


# Split data to: train data, validate data. Test data is from CSV

# In[49]:


from sklearn.model_selection import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(X, y, random_state=2018, test_size=0.4)


# In[50]:


import numpy as np
from sklearn.svm import SVC
clf = SVC()
clf.fit(x_tr, y_tr) 
model = SVC()
model.fit(x_tr, y_tr)


# In[51]:


print(round(model.score(x_te, y_te),4))


# My best submission was 0.75598 on test data. 

# In[52]:


from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(x_tr, y_tr, verbose=False)


# In[53]:


my_model.score


# In[54]:


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(x_tr, y_tr)


# In[55]:


xgb.score(x_te, y_te)


# In[56]:


df = pd.concat([df_test,pd.get_dummies(df_test.title)], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Embarked)], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Sex)], axis = 1)
df = pd.concat([df,pd.get_dummies(df.Pclass)], axis = 1)
df = df.drop("Embarked",axis = 1)
df= df.drop("title",axis = 1)
df =df.drop("surname",axis = 1)
df = df.drop("Sex", axis = 1)
df = df.drop("Pclass", axis = 1)

df.columns
df = df.drop(" Dona", axis = 1)


# In[57]:


x_tr.columns


# In[58]:


df.columns


# In[59]:


x_tr = x_tr[df.columns]


# In[60]:


x_tr.columns


# In[61]:


xgb = XGBClassifier()
xgb.fit(x_tr, y_tr)


# In[62]:


x_te = x_te[df.columns]


# In[63]:


xgb.score(x_te, y_te)


# In[64]:


y_predict = xgb.predict(df)


# In[65]:


len(y_predict)


# In[66]:


df.shape


# In[67]:


df_test = pd.read_csv('../input/test.csv')


# In[68]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('titanic_mvp_1_19_04_2018.csv', index=False)


# Your submission scored 0.76076, which is not an improvement of your best score. Keep trying!

# In[69]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_tr, y_tr)


# In[70]:


clf.score(x_te, y_te)


# In[71]:


y_predict = clf.predict(df)


# In[72]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('titanic_mvp_2_19_04_2018.csv', index=False)


# Your submission scored 0.77033, which is not an improvement of your best score. Keep trying!
# 
# 

# In[73]:


for i in range (2,10):
    clf = RandomForestClassifier(max_depth=i, random_state=2018)
    clf.fit(x_tr, y_tr)
    print("For i = {}, score is {}".format(i,clf.score(x_te, y_te)))


# In[74]:


clf = RandomForestClassifier(max_depth=4, random_state=2018)
clf.fit(x_tr, y_tr)
y_predict = clf.predict(df)


# In[75]:


submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('titanic_mvp_3_19_04_2018.csv', index=False)


# Your Best Entry 
# You advanced 528 places on the leaderboard!
# Your submission scored 0.78947, which is an improvement of your previous score of 0.78468. Great job!

# In[76]:


from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[77]:


xgb_model = XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.05], #so called `eta` value
              'max_depth': [7],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(x_tr, y_tr)


# In[78]:


clf.score(x_te, y_te)


# In[79]:


y_predict = clf.predict(df)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('titanic_mvp_4_19_04_2018.csv', index=False)


# ![Mapa](https://s3.amazonaws.com/MLMastery/MachineLearningAlgorithms.png?__s=dprexsnpfokqfnwxhrgr)

# In[80]:


Scores = {}


# In[81]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(x_tr,y_tr)

Scores[clf] = clf.score(x_te, y_te)


# In[82]:


Scores


# In[83]:


from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=4, random_state=0).fit(x_tr,y_tr)
clf.score(x_te, y_te)


# In[84]:


for i in range (2,10):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=i, random_state=0).fit(x_tr,y_tr)
    print("For i = {}, score is {}".format(i,clf.score(x_te, y_te)))


# In[85]:


for i in range (2,10):
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=i, random_state=0).fit(x_tr,y_tr)
    print("For i = {}, score is {}".format(i,clf.score(x_te, y_te)))


# In[86]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier as tree


# In[87]:



params_gs = {'criterion':('entropy', 'gini'),
'splitter':('best','random'),
'max_depth':np.arange(1,6),
'min_samples_split':np.arange(3,8),
'min_samples_leaf':np.arange(1,5)}
 
params_rs = {'criterion':('entropy', 'gini'),
'splitter':('best','random'),
'max_depth':randint(1,6),
'min_samples_split':randint(3,8),
'min_samples_leaf':randint(1,5)}


# In[88]:


model = tree()
gs = GridSearchCV(tree(), cv = 10, param_grid = params_gs, scoring = 'accuracy')
gs.fit(x_tr, y_tr) 


# In[89]:


cv_score_gs = []
final_score_gs = []

for i in range(0, 100):
    print('Iteracja: ' + str(i))
    gs = GridSearchCV(tree(), cv = 10, param_grid = params_gs, scoring = 'accuracy', n_jobs = -1)
    gs.fit(x_tr, y_tr)
    cv_score_gs.append(gs.best_score_)
    # test modelu - parametry GridSearchCV
    model_1 = tree(**gs.best_params_)
    model_1.fit(x_tr, y_tr)
    final_score_gs.append(model_1.score(x_te, y_te))


# In[90]:


print(np.mean(cv_score_gs)) # 0.873
print(np.mean(final_score_gs)) # 0.88
 
# Współczynnik zmienności
print(np.std(cv_score_gs)/np.mean(cv_score_gs) * 100) # 0.34
print(np.std(final_score_gs)/np.mean(final_score_gs) * 100) # 1.338


# In[91]:


cv_score_rs = []
final_score_rs = []

for i in range(0, 100):
   print('Iteracja: ' + str(i))
   rs = RandomizedSearchCV(tree(), cv = 10, n_iter = 20, param_distributions = params_rs, n_jobs = -1)
   rs.fit(x_tr, y_tr)
   cv_score_rs.append(rs.best_score_)
   # test modelu - parametry RandomizedSearchCV
   model_2 = tree(**rs.best_params_)
   model_2.fit(x_tr, y_tr)
   final_score_rs.append(model_2.score(x_te, y_te))


# In[92]:


# Średnie cv score
np.mean(cv_score_rs) # 0.864
np.mean(final_score_rs) # 0.883

# Współczynnik zmienności
np.std(cv_score_rs)/np.mean(cv_score_rs) * 100 # 0.526
np.std(final_score_rs)/np.mean(final_score_rs) * 100 # 1.224


# In[94]:


np.mean(cv_score_rs) 


# In[95]:


np.mean(final_score_rs) # 0.883


# In[96]:


y_predict = rs.predict(df)


# In[97]:


y_predict


# In[98]:


y_predict = clf.predict(df)
submission = pd.DataFrame({
        "PassengerId": df_test["PassengerId"],
        "Survived": y_predict
    })
submission.to_csv('titanic_mvp_5_26_04_2018.csv', index=False)


# In[ ]:




