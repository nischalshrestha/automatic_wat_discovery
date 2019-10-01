#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# This is one of many competitions that Kaggle provided. <br>
# As we know, the sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. <br>
# The purpose of this challenge is  to complete the analysis of what sorts of people were likely to survive or in particular is to apply the tools of machine learning to predict which passengers survived the tragedy. <br>
# Link from Kaggle: [Titanic](https://www.kaggle.com/c/titanic)

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
sns.set_style('whitegrid')


# ##  Exploratory Data Analysis

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head(5)


# | Variable | Definition | Key   |
# |------|------|------|
# |   Survival  | Survival| 0 = No, 1 = Yes|
# |   pclass  | Ticket class| 1 = 1st, 2 = 2nd, 3 = 3rd|
# |   sex  | sex|  |
# |   Age  | Age in years|  |	
# |   sibsp  | # of siblings / spouses aboard the Titanic|  |	
# |   parch  | # of parents / children aboard the Titanic|  |	
# |   ticket  | Ticket number|  |	
# |   fare	| Passenger fare |  |			
# |   cabin	| Cabin number |  |			
# |   embarked | Port of Embarkation | C = Cherbourg, Q = Queenstown, S = Southampton |	

# $\textbf{Variable Notes}$ <br>
# - $\textbf{pclass}$ : A proxy for socio-economic status (SES) <br>
# 1st = Upper <br>
# 2nd = Middle <br>
# 3rd = Lower <br>
# 
# - $\textbf{age}$: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5 <br>
# 
# - $\textbf{sibsp}$ : The dataset defines family relations in this way... <br>
# Sibling = brother, sister, stepbrother, stepsister <br> 
# Spouse = husband, wife (mistresses and fiancés were ignored) <br>
# 
# - $\textbf{parch}$: The dataset defines family relations in this way... <br>
# Parent = mother, father<br>
# Child = daughter, son, stepdaughter, stepson <br>
# Some children travelled only with a nanny, therefore parch=0 for them. <br>

# #### Is there any null data?

# In[ ]:


df.isnull().sum(axis = 0)


# In[ ]:


df = df.drop(df.columns[[8, 10]], axis = 1)
df = df.fillna(df.mean())


# In order to deal with missing values (NaN), we fill those empty data with mean for each colums.

# #### How many survivors?

# In[ ]:


df['Survived'].value_counts()


# In[ ]:


df.pivot_table(['Survived'], ['Sex', 'Pclass']).sort_values(by = ['Survived'], ascending = False)


# As we can see, women in the first class are the most survivors of titanic accidents. There is a priority to rescue women first and class sequences also influence the probability level of passengers survived.

# In[ ]:


g = sns.PairGrid(df, y_vars="Survived",
                 x_vars=["Pclass", "Sex"],
                 size=5, aspect=.5)

g.map(sns.pointplot, color=sns.xkcd_rgb["green"])
g.set(ylim=(0, 1))
sns.despine(fig=g.fig, left=True)


# In[ ]:


df[['SibSp', 'Parch']].hist(figsize=(16, 10), xlabelsize=8, ylabelsize=8);


# #### How many passengers that travelled alone?

# In[ ]:


Alone = [0 for k in range(len(df))]
for p in range(len(df)):
    if df['SibSp'][p] == 0 and df['Parch'][p] == 0:
        Alone[p] = 1


# In[ ]:


df = df.assign(IsAlone =Alone)
df['IsAlone'].value_counts()


# In[ ]:


plt.rcParams['figure.figsize'] = (10, 8)
sns.countplot(x='Survived', hue='IsAlone', data=df);


# There are variables of SibSp (Siblings or Spouse) and Parch (Parent or Children) that shows passengers who travelled alone and not. About 60% of total passengers travelled alone, and 50% of those passengers were not survived.

# #### What ticket is the most expensive?

# In[ ]:


expensive_ = df.sort_values(by='Fare', ascending = False)
expensive_.head(10)


# The most expensive fare for Titanic was 512.329 which embarked at Cherbourg and had first Class. And apparently, all passengers that had most expensive ticket got survived from the accident.

# ## Correlation

# In[ ]:


df = df.drop(df[["PassengerId", "Name"]], axis=1)
df = df.dropna(axis = 0, how = 'any')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df['Sex'] = lb.fit_transform(df['Sex'])


# In[ ]:


embarked_ = pd.get_dummies(df['Embarked'],prefix = 'Embarked' )
df = df.assign(C=embarked_['Embarked_C'], Q=embarked_['Embarked_Q'], S=embarked_['Embarked_S'])
df = df.drop(['Embarked'], axis=1)
df.head(5)


# In[ ]:


df['Age'] = np.round(df['Age'])


# In[ ]:


corr_ = df.corr()
corr_[abs(corr_) < 0.5] = 0


# In[ ]:


plt.figure(figsize=(16,10))
sns.heatmap(corr_, annot=True)
plt.show()


# In[ ]:


df_corr = df.corr()['Survived'][1:]
goldlist = df_corr[abs(df_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Survived:\n{}".format(len(goldlist), goldlist))


# In this section, we want to look correlation between variables, but to make it easy, we only want to look high correlation (coef > 0.5). It's clear that each embarked has high negative correlation between each others because whenever one variable "embarked" is filled, then another variable "embarked" will be zero. Variable "Is Alone" has high correlation between "SibSp" and "Parch" because we build the data from those two variables. "Fare" and "Pclass" have high negative correlation, because the smaller Pclass get the higher fare and vice versa. Only variable "Sex" that has high correlation to "Survived", and it is negative because we made 'male' is equal to 1 and 'female' equal to 0, and passengers 'survived' is equal to 1, even though we know that most survivors is female, thus there will be negative correlation. 

# ## Identify Key Features

# We want to find the key features that affect the process of classification of the variable "survived". In this case, we used logistic regression.

# In[ ]:


X = df.drop('Survived', axis=1)
y = df['Survived']


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)
X_test_std = sc.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_std, y)


# In[ ]:


result = pd.DataFrame(model.coef_, columns = X.columns)
result = result.T
result.columns = ['coefficient']


# In[ ]:


np.abs(result).sort_values(by='coefficient', ascending=False)


# This is the list that shows the variable that gives the most influence on the classification of the "survived".

# In[ ]:


import statsmodels.formula.api as smf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# #### All Variables

# In[ ]:


log_reg = smf.ols(formula = 'y ~ Pclass+Sex+Age+SibSp+Parch+Fare+IsAlone+C+Q+S', data=X)
benchmark = log_reg.fit()
print('r2 score : \t', r2_score(y, benchmark.predict(X)))
print('mse : \t', mean_squared_error(y, benchmark.predict(X)))


# #### Without Variable 'Sex'

# In[ ]:


log_reg = smf.ols(formula = 'y ~ Pclass+Age+SibSp+Parch+Fare+IsAlone+C+Q+S', data=X)
benchmark = log_reg.fit()
print('r2 score : \t', r2_score(y, benchmark.predict(X)))
print('mse : \t', mean_squared_error(y, benchmark.predict(X)))


# Now we see, if we do not input variable "Sex" into classification, then we get lower result of $R^2$ and higher error

# #### Without Variable 'C'

# In[ ]:


log_reg = smf.ols(formula = 'y ~ Pclass+Sex+Age+SibSp+Parch+Fare+IsAlone+Q+S', data=X)
benchmark = log_reg.fit()
print('r2 score : \t', r2_score(y, benchmark.predict(X)))
print('mse : \t', mean_squared_error(y, benchmark.predict(X)))


# But, if we do not input variable "Embarked_C", the results did not change significantly.

# ## Making Model

# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict


# We build some models to seek the best model we can provided.
# - [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
# <br> Support Vector Machine will optimize and locate a separating line, which is the line that allows for largest gap between the two classes. New data are then mapped into that same space and predicted to belong to a class based on which side of the gap they fall. <br>
# <br>
# - [Gradient Boosting](https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/)
# <br> Gradient Boosting generally use decision tree as "weak learners". The concept is to add predictors, and each predictor will correct its predecessor, thus finally fit new predictor to the residual errors. <br>
# <br>
# - Random Forest
# <br> Random Forest is Ensemble of Decision Trees. Random forests are a way of averaging multiple deep decision trees, trained on different parts of the same training set, with the goal of reducing the variance. [Friedman, J., Hastie, T., & Tibshirani, R. (2001). The elements of statistical learning (Vol. 1, No. 10). New York, NY, USA:: Springer series in statistics}]. <br>
# <br>
# - [Logistic Regression](https://machinelearningmastery.com/logistic-regression-for-machine-learning/)
# <br> Logistic regression is the go-to linear classification algorithm for two-class problems. Logistic Regression use sigmoid function that allows such real-valued number and map it into a value between 0 and 1. <br>
# <br>
# - [K-Nearest Neighbor (KNN)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
# <br> An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small) <br>
# <br>
# - [AdaBoost](http://rob.schapire.net/papers/explaining-adaboost.pdf)
# <br> Adaptive Boosting is one of Boosting algortihm, which combine "weak learners" into a weighted sum to produce the final prediction. The difference between gradient boosting, AdaBoost alter instance weights at every iteration.  <br>
# <br>
# - [XGboost](http://xgboost.readthedocs.io/en/latest/)
# <br> Both XGBoost and Gradient Boosting Machine follows the principle of gradient boosting. XGBoost used a more regularized model formalization to control over-fitting, which gives it better performance. <br>
# <br>
# - [Artificial Neural Networks](http://metalab.uniten.edu.my/~abdrahim/mitm613/Jain1996_ANN%20-%20A%20Tutorial.pdf)
# <br> ANNs are computing system that inspired by neural networks in brain. The system learn from some examples that we provided, generally without being programmed with any task-specific rules.<br>
# <br>
# 
# To find the most significant models, we use [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics), then use the score to compare those models. 

# ### SVM Classifier

# In[ ]:


from sklearn import svm
clf_rbf = svm.SVC(kernel='rbf', degree = 10, C = 1)


# In[ ]:


clf_rbf.fit(X_std, y)


# In[ ]:


res_clf_rbf = cross_val_score(clf_rbf, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_rbf)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_rbf)))


# ### Gradient Boosting

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
clf_gb = GradientBoostingClassifier()
clf_gb.fit(X_std, y)


# In[ ]:


res_clf_gb = cross_val_score(clf_gb, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_gb)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_gb)))


# ### Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(min_samples_leaf = 10, n_estimators = 10)
rf.fit(X_std, y)


# In[ ]:


res_rf = cross_val_score(rf, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_rf)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_rf)))


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
clf_lr = LogisticRegression()
clf_lr.fit(X_std, y)


# In[ ]:


res_clf_lr = cross_val_score(clf_lr, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_lr)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_lr)))


# ### KNN Classifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(X_std, y)


# In[ ]:


res_knn = cross_val_score(knn, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_knn)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_knn)))


# ### AdaBoost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier()
clf_ada.fit(X_std, y)


# In[ ]:


res_clf_ada = cross_val_score(clf_ada, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_ada)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_ada)))


# ### XGBoost

# In[ ]:


import xgboost as xgb
clf_xgb = xgb.XGBClassifier(random_state = 42)
clf_xgb.fit(X_std, y)


# In[ ]:


res_clf_xgb = cross_val_score(clf_xgb, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_xgb)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_xgb)))


# ### All Classifier

# In[ ]:


classifier = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Gradient Boosting', 'Random Forest', 
            'Logistic Regression', 'KNN', 
              'Adaboost', 'XGBoost'],
    'Score': [np.mean(res_clf_rbf), np.mean(res_clf_gb), np.mean(res_rf), 
              np.mean(res_clf_lr), np.mean(res_knn), 
              np.mean(res_clf_ada), np.mean(res_clf_xgb)]})
classifier.sort_values(by='Score', ascending=False)


# So far, gradient boosting has the best score of cross validation. However, we have not try some improvements to those algorithm and ANNs as well.

# ## Grid Search for XGBoost

# Now we want to do Grid Search, it search some parameters on an algorithm that produce maximum score.

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


params = {'max_depth':(5, 10, 25, 50), 
          'n_estimators':(50, 200, 500, 1000)} 


# We seek through two parameters,
# - max_depth : Maximum tree depth for base learners
# - n_estimators : Number of boosted trees to fit

# In[ ]:


clf_xgb_grid = GridSearchCV(clf_xgb, params, n_jobs=-1,
                            cv=3, verbose=1, scoring='accuracy')
clf_xgb_grid.fit(X_std, y)


# In[ ]:


clf_xgb_grid.best_estimator_.get_params


# In[ ]:


clf_xgb_ = xgb.XGBClassifier(random_state = 42, learning_rate = 0.1, max_depth = 5, n_estimators=50, n_jobs=1)
clf_xgb_.fit(X_std, y)
res_clf_xgb_ = cross_val_score(clf_xgb_, X_std, y, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_xgb_)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_xgb_)))


# It is improved, the result of average accuracy is higher than before (about 0.81)

# ## Importance of Key Features

# Now we want to see if some variables cause interference on the classifier, thus we just want the key features to be inputted, we choose the top4 and see what happen.

# In[ ]:


columns_to_show = ['Pclass', 'Sex', 'Age', 'SibSp']
X_ = df[columns_to_show]
y_ = df['Survived']


# In[ ]:


X_std_ = sc.fit_transform(X_)


# In[ ]:


clf_xgb_.fit(X_std_, y_)


# In[ ]:


res_clf_xgb_t = cross_val_score(clf_xgb_, X_std_, y_, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res_clf_xgb_t)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(res_clf_xgb_t)))


# So, it is improved, but not significant. The average accuracy only increases 0.002 from the previous but it's worth a try.

# ## Artificial Neural Networks

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# This is the last classifier that we want to try, we use [Keras](https://keras.io/). Keras is a high-level neural networks API written in Python.

# In[ ]:


classifier_ = Sequential()
classifier_.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu', input_dim = 10))
classifier_.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'relu'))
classifier_.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid'))
classifier_.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[ ]:


classifier_.fit(X_std, y, batch_size = 20, epochs = 200)


# In[ ]:


metr_ = classifier_.evaluate(X_std, y_)
print("Loss metric: \t {0:.4f}".format(metr_[0]))
print("Accuracy metric: \t{0:.4f}".format(metr_[1]))


# ## Conclusion

# In[ ]:


classifier_update = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Gradient Boosting', 'Random Forest', 
            'Logistic Regression', 'KNN', 
              'Adaboost', 'XGBoost', 'ANN'],
    'Score': [np.mean(res_clf_rbf), np.mean(res_clf_gb), np.mean(res_rf), 
              np.mean(res_clf_lr), np.mean(res_knn), 
              np.mean(res_clf_ada), np.mean(res_clf_xgb_), metr_[1]]})
classifier_update['Score'] = np.round(classifier_update['Score'], decimals = 4)
classifier_update.sort_values(by='Score', ascending=False)


# In[ ]:


plt.figure(figsize=(16, 8))

x = classifier_update['Model']
y = classifier_update['Score']
sns.barplot(x, y, palette="Set3")
plt.ylabel("Score")
plt.xlabel("Models")
plt.ylim(0.75, 0.86);


# ### Test Data

# In[ ]:


df2 = pd.read_csv('../input/test.csv')
df2 = df2.drop(df2.columns[[2,7,9]], axis = 1)
df2 = df2.fillna(df2.mean())
df2 = df2.dropna(axis = 0, how = 'any')


# In[ ]:


alone = [0 for k in range(len(df2))]
for p in range(len(df2)):
    if df2['SibSp'][p] == 0 and df2['Parch'][p] == 0:
        alone[p] = 1


# In[ ]:


df2 = df2.assign(IsAlone=alone)
df2 = df2.dropna(axis = 0, how = 'any')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lb_t = LabelEncoder()
df2['Sex'] = lb_t.fit_transform(df2['Sex'])

embarked_ = pd.get_dummies(df2['Embarked'],prefix = 'Embarked' )
df2 = df2.assign(C=embarked_['Embarked_C'], Q=embarked_['Embarked_Q'], S=embarked_['Embarked_S'])
df2 = df2.drop(['Embarked'], axis=1)

df2['Age'] = np.round(df2['Age'])
df2.head(3)


# In[ ]:


X_test = df2.drop('PassengerId', axis = 1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_std = sc.fit_transform(X)
X_test_std = sc.transform(X_test)


# In[ ]:


y_pred = classifier_.predict(X_test_std)
y_pred = (y_pred > 0.5)
y_pred = pd.Series(list(y_pred))

submission = pd.DataFrame({"PassengerId": df2["PassengerId"], "Survived": y_pred})
submission['Survived'] = submission['Survived'].astype(int)
submission.head()
#submission.to_csv('..\submission.csv', index = False)

