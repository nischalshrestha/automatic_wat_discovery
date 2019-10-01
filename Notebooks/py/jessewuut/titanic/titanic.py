#!/usr/bin/env python
# coding: utf-8

# In[216]:


get_ipython().magic(u'matplotlib inline')


# In[217]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# # Data Exploration

# In[218]:


train = pd.read_csv("../input/train.csv", index_col="PassengerId")


# In[219]:


train.head()


# ### Histograms for Obvious Potential Predicters

# In[220]:


sns.pairplot(train.dropna(how='any'), hue='Survived')


# # Feature Engineering
# ## "Title", within Name

# In[221]:


train['Title'] = train.Name.apply(lambda x: x[x.find(',')+2:x.find('.')])
del train['Name'] # Remove the original name, which is no longer useful


# ## Isolating Cabin
# Passengers can't be spread out across cabins

# In[222]:


train.loc[train.Cabin.str.len() > 5, 'Cabin'] # Checking whether a passenger can have multiple rooms in different cabins


# In[223]:


# Add the cabin letter as a factor, NaN if no data is available
train['Cabin'] = train['Cabin'].fillna(value=' ')
train['Cabin'] = train.Cabin.map(lambda x: x[0]).replace(' ', np.nan)


# ## Removing Non-Relevant Columns
# Ticket Number is the only one

# In[224]:


del train['Ticket']


# # Factor Plots and Linear Relationships

# In[225]:


train.head()


# ## Survived

# In[226]:


train['Survived'].value_counts()


# In[227]:


sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
plt.show()


# ### Descriptive Statistics by Survival

# In[228]:


train.groupby('Survived').mean()


# We see, as expected, that the mean SES and Fare paid of passengers that survived was higher. Average age of survivors was lower, which may go along with the "women and children first" story.  SibSp and Parch I am unsure of just by viewing means.

# ### Categorical Means for Factor Variables

# In[229]:


train.head()


# In[230]:


train.groupby('Sex').mean()


# Sex seems to be a huge predictor for survival.  This continues to subscribe to the "Women and Children" story.

# In[231]:


train.groupby('Embarked').mean()


# Port of embarkment seems to be tied up with Socioeconomic class and Fare, but we see a higher rate of survival from Queenstown than Southampton, despite them having lower SES and fares.  Maybe they have a higher F:M sex ratio?

# In[232]:


train.groupby(['Embarked', 'Sex']).count()


# Bingo, Queenstown has nearly a 1:1 F/M ratio while Southamtpon has < 1:2.

# In[233]:


train.groupby('Cabin').mean()


# In[234]:


train.groupby(['Cabin', 'Sex']).count()


# Cabin might be a good predictor, judging by the higher survival in cabins D and E as compared to other cabins with higher average SES and Fair, despite having similar M/F ratios.  The amount of missing cabin data could make things difficult though.

# In[235]:


train.groupby('Title').mean()


#  I'll have to be careful of overfitting based on these titles, as a bunch fo them only have a few or one persons to them.  It seemed that "Mrs"  did better than "Miss", which seems to suggest that old women would survive more than young women.

# ## Visualizations

# ### Sex

# In[236]:


pd.crosstab(train.Sex, train.Survived).plot(kind='bar')


# There definitely seems to be an effect

# ### Embarked

# In[237]:


table = pd.crosstab(train.Embarked, train.Survived)


# In[238]:


table.plot(kind='bar', stacked=True)


# In[239]:


table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Port of Embarkment Vs Survival')
plt.ylabel('Proportion of Passengers')


# ### Cabin

# In[240]:


table_cabin = pd.crosstab(train.Cabin, train.Survived)
table_cabin.plot(kind='bar', stacked=True)


# In[241]:


table_cabin.div(table_cabin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)


# Cabin seems like it could have an effect

# ### Title

# In[242]:


table_title = pd.crosstab(train.Title, train.Survived)
table_title.plot(kind='bar', stacked=True)


# In[243]:


table_title.div(table_title.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)


# Definitely seems to be an effect for title, but we need to be careful of overfitting.  I'll probably take most of them out when converting to binary variables.

# ### Age

# In[244]:


sns.distplot(train.Age.dropna(), rug=True)


# Most passengers are aged between 20-40

# # Create Dummy Variables

# In[245]:


train.head()


# In[246]:


cat_vars=['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']
train_dummy = train.copy()
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(train[var], prefix=var)
    train_dummy=train_dummy.join(cat_list)
    
train_dummy.drop(columns=cat_vars, inplace=True)


# We want to get rid of a bunch of these titles

# In[247]:


train.Title.value_counts()


# In[248]:


bad_title = ['Dr', 'Rev', 'Col', 'Mlle', 'Major', 'Mme', 'Ms', 'Capt', 'the Countess', 'Don', 'Lady', 'Sir', 'Jonkheer']
bad_title = ['Title_' + title for title in bad_title]
train_dummy.drop(columns=bad_title, inplace=True)


# In[250]:


response = ['Survived']
predictor = [x for x in train_dummy.columns if x != 'Survived']


# In[251]:


#imputing
from sklearn.preprocessing import Imputer
my_imputer = Imputer()

age_imputed = my_imputer.fit_transform(train_dummy.Age.values.reshape(-1,1))
train_dummy['Age'] = age_imputed


# In[252]:


train_dummy.head()


# # Feature Selection

# In[253]:


from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(train_dummy[predictor], train_dummy[response].as_matrix().ravel())
print(rfe.support_)
print(rfe.ranking_)


# In[254]:


rfe_cols = [x for x,y in zip(predictor, rfe.support_) if y==True]
print(rfe_cols)


# In[255]:


predictor


# # Implement the Model

# In[256]:


import statsmodels.api as sm
from scipy import stats
#d from a weird error
#from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


logit_model=sm.Logit(train_dummy[response], train_dummy[predictor])
result=logit_model.fit()
result.summary()


# # Evaluate the Model

# ## Accuracy

# In[257]:


X_train, X_test, y_train, y_test = train_test_split(train_dummy[predictor].as_matrix(), np.ravel(train_dummy[response]), test_size=0.3, random_state=0)
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[258]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# ### Cross Validation

# In[259]:


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, train_dummy[predictor].as_matrix(), train_dummy[response].as_matrix().ravel(), cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


# ## Confusion Matrix
# 

# In[260]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[261]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[262]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Evaluation with Full Training Set

# In[263]:


test = pd.read_csv("../input/test.csv", index_col="PassengerId")


# In[264]:


# Add the cabin letter as a factor, NaN if no data is available
test['Cabin'] = test['Cabin'].fillna(value=' ')
test['Cabin'] = test.Cabin.map(lambda x: x[0]).replace(' ', np.nan)


# In[265]:


test['Title'] = test.Name.apply(lambda x: x[x.find(',')+2:x.find('.')])
del test['Name'] # Remove the original name, which is no longer useful
del test['Ticket']


# In[266]:


cat_vars=['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']
test_dummy = test.copy()
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(test[var], prefix=var)
    test_dummy = test_dummy.join(cat_list)
    
test_dummy.drop(columns=cat_vars, inplace=True)


# In[267]:


test.Title.value_counts()


# In[268]:


bad_title = ['Col', 'Rev', 'Dr', 'Ms', 'Dona']
bad_title = ['Title_' + title for title in bad_title]
test_dummy.drop(columns=bad_title, inplace=True)


# In[297]:


my_imputer = Imputer()

age_imputed = my_imputer.fit_transform(test_dummy.Age.values.reshape(-1,1))
test_dummy['Age'] = age_imputed

# 1 value needs fare imputing for the test set
fare_imputed = my_imputer.fit_transform(test_dummy.Fare.values.reshape(-1,1))
test_dummy['Fare'] = fare_imputed


# In[278]:


train_dummy.drop(columns='Survived').columns


# In[277]:


test_dummy.columns


# Columns are not equivalent, there were no passengers in Cabin T in the test set

# In[281]:


test_dummy['Cabin_T'] = 0
test_dummy = test_dummy[train_dummy.drop(columns='Survived').columns.tolist()]


# # Train Model and Make Predictions

# In[300]:



X_train = train_dummy[predictor].as_matrix()
X_test = test_dummy.as_matrix()
y_train = np.ravel(train_dummy[response])

from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[301]:


# Predictions
y_pred = logreg.predict(X_test)


# In[305]:


#format for writing to csv
submission = pd.DataFrame(data={'PassengerId': test_dummy.index.values, 'Survived': y_pred})


# In[309]:


submission.head()


# In[311]:


submission.to_csv("solution_logit.csv", index=False)


# In[ ]:




