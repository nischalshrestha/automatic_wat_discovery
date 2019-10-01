#!/usr/bin/env python
# coding: utf-8

# ### useful libraries.

# In[ ]:


get_ipython().magic(u'matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ### Data Preparation
# 

# In[ ]:


train = pd.read_csv('../input/train.csv', header=0)
test = pd.read_csv('../input/test.csv', header=0)
print(train.shape)
print(test.shape)
#train.describe()


# In[ ]:


train.describe()


# In[ ]:


train.head(2)


# In[ ]:


test.head(2)


# ### data preparation

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# In[ ]:


train['source'] = 'train'
test['source']= 'test'
fulldata = pd.concat([train, test], ignore_index=True)
fulldata.shape


# In[ ]:


fulldata[fulldata.Embarked.isnull()][['Fare', 'Pclass', 'Embarked']]


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax = fulldata.boxplot(column='Fare', by=['Embarked','Pclass'], ax=ax)
plt.axhline(y=80, color='green')


# In[ ]:


_ = fulldata.set_value(fulldata.Embarked.isnull(), 'Embarked', 'C')


# In[ ]:


fulldata[fulldata.Fare.isnull()][['Fare', 'Pclass', 'Embarked']]


# In[ ]:


fulldata[(fulldata.Pclass==3)&(fulldata.Embarked=='S')].Fare.value_counts().head()


# In[ ]:


_ = fulldata.set_value(fulldata.Fare.isnull(), 'Fare', 8.05)
_ = fulldata.drop(labels=['PassengerId', 'Name', 'Cabin','Ticket'], axis=1, inplace=True)


# In[ ]:


fulldata['family_num'] = fulldata.Parch + fulldata.SibSp + 1
_ = fulldata.drop(labels=['Parch','SibSp'], axis=1, inplace=True)
fulldata['Family_size'] = pd.Series('M', index=fulldata.index)
_ = fulldata.set_value(fulldata.family_num > 4, 'Family_size', 'L')
_ = fulldata.set_value(fulldata.family_num < 3, 'Family_size', 'S')
_ = fulldata.drop(labels=['family_num'], axis=1, inplace=True)


# ### Normalize the fare

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
fulldata['Fare'] = pd.Series(scaler.fit_transform(fulldata.Fare.reshape(-1, 1)).reshape(-1), index=fulldata.index)


# In[ ]:


fulldata.Sex = np.where(fulldata.Sex == 'female', 0, 1)
fulldata = pd.get_dummies(fulldata, columns=['Embarked', 'Pclass', 'Family_size'])


# In[ ]:


fulldata.head(2)


# ## predict age by random forest

# In[ ]:


age_data = fulldata.drop(['source', 'Survived'],  axis=1)
age_data.head(2)


# In[ ]:


from sklearn.model_selection import train_test_split
X = age_data[~age_data.Age.isnull()].drop(['Age'], axis=1)
y = age_data[~age_data.Age.isnull()].Age
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[500, 1000, 2000], 'max_depth':[4,8,10], 'min_samples_leaf': [8, 10, 12], 'max_features': ['sqrt', 'log2', .5, .9]}
clf = GridSearchCV(RandomForestRegressor(random_state=40), parameters, cv=5, 
                   n_jobs=-1, verbose=1)
clf.fit(X_train, y_train)
rfr = clf.best_estimator_


# In[ ]:


pred_age = rfr.predict(age_data[age_data.Age.isnull()].drop(['Age'], axis=1))
_ = fulldata.set_value(fulldata.Age.isnull(), 'Age', pred_age)


# In[ ]:


fulldata.head(2)


# In[ ]:


fulldata['Age'] = pd.Series(scaler.fit_transform(fulldata.Age.reshape(-1,1)).reshape(-1), index=fulldata.index)


# In[ ]:


from sklearn.model_selection import learning_curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



# In[ ]:


from sklearn.metrics import roc_curve, auc
def plot_roc_curve(estimator, X, y, title):
    # Determine the false positive and true positive rates
    fpr, tpr, _ = roc_curve(y, estimator.predict_proba(X)[:,1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    print ('ROC AUC: %0.2f' % roc_auc)

    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - {}'.format(title))
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

def build_model(estimator, parameters, X_train, y_train, scoring):
    model = GridSearchCV(estimator, param_grid = parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_


# In[ ]:


from sklearn.metrics import accuracy_score
scoring = make_scorer(accuracy_score, greater_is_better=True)


# In[ ]:


Train_data = fulldata.loc[fulldata['source']=='train']
Test_data = fulldata.loc[fulldata['source']=='test']
_ = Train_data.drop('source',axis=1,inplace=True)
_ = Test_data.drop(['source', 'Survived'],axis=1,inplace=True)
X = Train_data.drop('Survived',axis=1)
y = Train_data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42, criterion='entropy', min_samples_split=5, oob_score=True)
parameters = {'n_estimators':[300, 400, 500], 'min_samples_leaf':[12], 'max_depth':[3,4,5], 'max_features': ['sqrt', 'log2', .5, .9, 1]}
clf_rfc = build_model(rfc, parameters, X_train, y_train, scoring)


# In[ ]:


print (accuracy_score(y_test, clf_rfc.predict(X_test)))


# In[ ]:


plt.figure(figsize=(10,6))
plt.barh(np.arange(X_train.columns.shape[0]), clf_rfc.feature_importances_, 0.5)
plt.yticks(np.arange(X_train.columns.shape[0]), X_train.columns)
plt.grid()
plt.xticks(np.arange(0,0.2,0.02));


# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier(seed=42,objective='binary:logistic')
params = {'max_depth':[3,4,5], 'n_estimators': [400,600,800], 'subsample': [0.7,0.8,0.9], }
clf_xgb = build_model(xgb, params, X_train, y_train, scoring)
print (accuracy_score(y_test, clf_xgb.predict(X_test)))


# In[ ]:


clf_xgb


# In[ ]:


cols = X_train.columns[clf_rfc.feature_importances_>=0.016]
xgb1 = XGBClassifier(seed=42,objective='binary:logistic', random_state=42)
params = {'max_depth':[3,4,5], 'n_estimators': [300,400,500], 'subsample': [0.8, 0.75, 0.7], 'colsample_bytree':[0.5, 0.6, 0.8],'learning_rate':[0.05, 0.08, 0.1]}
clf_xgb1 = build_model(xgb, params, X_train[cols], y_train, scoring)
xgb1_predicted = clf_xgb1.predict(X_test[cols])
print(accuracy_score(y_test, xgb1_predicted))


# In[ ]:


clf_xgb1


# In[ ]:


Test_data.head(2)


# In[ ]:


submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission.PassengerId = test['PassengerId']
submission.Survived = pd.Series(clf_xgb1.predict(Test_data[cols]), index=submission.index)
submission.Survived = np.where(submission.Survived == 0, 0, 1)
submission.to_csv("submission.csv", index=False)

