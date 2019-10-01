#!/usr/bin/env python
# coding: utf-8

# # 1. Imports

# In[ ]:


import numpy as np 
import pandas as pd
import scipy.stats as stats

from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from statsmodels.stats.outliers_influence import variance_inflation_factor   

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
combined = pd.concat((train,test))


# # 2. Diving Into The Data

# In[ ]:


print(train.shape)
print(test.shape)
print(combined.shape)


# In[ ]:


train.head()


# In[ ]:


train.describe()


# In[ ]:


print(train.isnull().sum())
print(test.isnull().sum())
print(combined.isnull().sum())


# In[ ]:


train.info()


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 4))
mosaic(train,["Survived",'Sex','Pclass'], axes_label = False, ax=ax)

plt.figure(figsize=[12,8])
plt.subplot(231)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(232)
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot(233)
sns.barplot('Pclass', 'Survived', hue = 'Sex', data=train)
plt.subplot(234)
sns.barplot('Parch', 'Survived', data=train)
plt.subplot(235)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(236)
sns.barplot('Embarked', 'Survived', data=train)

fig, axes = plt.subplots(1,3, figsize=(12, 4))
tab = pd.crosstab(combined['Embarked'], combined['Pclass'])
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=axes[0])
tab = pd.crosstab(combined['Embarked'], combined['Sex'])
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=axes[1])
tab = pd.crosstab(combined['Pclass'], combined['Sex'])
tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, ax=axes[2])

plt.figure(figsize=[12,4])
plt.hist([train.loc[(train['Survived'])==1, 'Age'].dropna(),           train.loc[(train['Survived'])==0, 'Age'].dropna()],           bins = 40, histtype='stepfilled', stacked=True, label=['Survived','Died'])
plt.legend()

age_bin = pd.cut(train['Age'], np.linspace(train['Age'].min(), train['Age'].max(), 41))
age_grouped = train['Survived'].groupby(age_bin).mean()

plt.figure(figsize=[12,4])
plt.plot(np.linspace(0,80,40), age_grouped, color='purple', label='Total')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Survival Percentage')

male_age_bin = pd.cut(train.loc[(train['Sex'] == 'male'),'Age'],	np.linspace(train['Age'].min(), train['Age'].max(), 41))
male_age_grouped = train.loc[(train['Sex'] == 'male'),'Survived'].groupby(male_age_bin).mean()

female_age_bin = pd.cut(train.loc[(train['Sex'] == 'female'),'Age'],	np.linspace(train['Age'].min(), train['Age'].max(), 41))
female_age_grouped = train.loc[(train['Sex'] == 'female'),'Survived'].groupby(female_age_bin).mean()

plt.figure(figsize=[12,4])
ax = plt.axes()
ax.plot(np.linspace(0,80,40),female_age_grouped, color='red', label='Female')
ax.plot(np.linspace(0,80,40),male_age_grouped, color='blue', label='Male')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Survival Percentage')


# In[ ]:


survival_percentage = round(train.pivot_table('Survived', index='Sex', columns='Pclass', 
                                              margins=True),3) * 100
print(f'Total Survival Percentage: {survival_percentage.iloc[2,3]}% \n'
	f'Female Survival Percentage: {survival_percentage.iloc[0,3]}% \n'
	f'Male Survival Percentage: {survival_percentage.iloc[1,3]}%')
print(survival_percentage)


# # 3. Feature Extraction

# In[ ]:


# Changed the Sex category from two different strings to a binary number
# Replaced Sex with Male to be more clear
# Created a list of dummy variables in order to represent the categorical Embarked data
# Dropped 'Cabin' and 'Ticket' as they were very incomplete, difficult to estimate, and most likely \
# very collinear with the other features. That being said, it would have been interesting to model \
# the survival rate by level in the ship that the cabin resided in.
# Lastly the PassengerId was made into an explicit Index in order to better manipulate the data \
# across the three different sets of data; train, test, combined.

le = LabelEncoder()
combined['Sex'] = le.fit_transform(combined['Sex'])
combined = combined.rename(columns={'Sex':'Male'})
combined = pd.concat([combined, pd.get_dummies(combined['Embarked'])],axis=1)
combined = combined.drop(['Cabin', 'Ticket', 'Embarked'], 1)
combined.set_index('PassengerId',drop=True,inplace=True)
train.set_index('PassengerId',drop=True,inplace=True)
train = combined.loc[train.index]


# In[ ]:


f, ax = plt.subplots(figsize=(11,9))
plt.title("Pearson Correlation of Features", y=1.02, size=15)
sns.heatmap(train.drop(['Name'],1).corr(),vmax=.6,cmap="RdBu_r",annot=True, square=True)


# In[ ]:


# Vectorized the string data in the Name category in order to extract the titles of the different \
# passengers.
# Then the data was sorted and manually inspected to pick titles out of the top 60 words

vec = CountVectorizer()
words = vec.fit_transform(combined['Name'])
names = pd.DataFrame(words.toarray(), columns=vec.get_feature_names())
print(names.sum().sort_values(ascending=False).head(60))
names.set_index(combined.index, inplace=True)


# In[ ]:


# Used a boolean mask to remove the most common titles from the data 
# Inspected the remaining data to extract the lesser known titles

mask = (names['master']==0) & (names['rev']==0) & (names['dr']==0) & (names['mrs']==0) &        (names['miss']==0) & (names['mr']==0)
print(combined[mask]['Name'])
print(combined[mask]['Name'].count())
print(combined[~(mask)]['Name'].count())


# In[ ]:


# Removed all the columns that were not passenger titles
# Searched for all passengers with more than one title (due to pseudonyms). If the title was two of \
# the same, it was corrected to be one. If the titles were differed, the real title was chosen to be \
# the official and the pseudonym was discarded

names = names[['master','mr','miss','mrs','dr','rev','don','mme','ms','major','mlle','col','capt',
 'countess','jonkheer','dona']]
print(names.sum().sum())
print(names[names.sum(1)>1])
names.loc[(names['mr']>1),'mr'] = 1
names.loc[(names.sum(1)>1),'miss'] = 1
names.loc[(names.sum(1)>1),'mlle'] = 0
names.loc[(names.sum(1)>1),'mrs'] = 0
print(names[names.sum(1)>1])
print(names.sum().sum())


# In[ ]:


# Since there were 16 titles, many of which encompassed only a single digit amount of passengers, \
# the titles were changed to their closest equivalent. The remaining titles were \
# miss, mrs, master, and mr

print(names.sum())
names.loc[(names['ms'])==1,'miss'] = 1
names.loc[(names['ms'])==1,'ms'] = 0
names.loc[(names['mme'])==1,'mrs'] = 1                         
names.loc[(names['mme'])==1,'mme'] = 0
names.loc[(names['mlle'])==1,'miss'] = 1
names.loc[(names['mlle'])==1,'mlle'] = 0
names.loc[(names['jonkheer'])==1,'mr'] = 1
names.loc[(names['jonkheer'])==1,'jonkheer'] = 0
names.loc[(names['countess'])==1,'mrs'] = 1
names.loc[(names['countess'])==1,'countess'] = 0
names.loc[(names['don'])==1,'mr'] = 1
names.loc[(names['don'])==1,'don'] = 0
names.loc[(names['dona'])==1,'mrs'] = 1
names.loc[(names['dona'])==1,'dona'] = 0
names.loc[(names['col'])==1,'mr'] = 1
names.loc[(names['col'])==1,'col'] = 0
names.loc[(names['major'])==1,'mr'] = 1
names.loc[(names['major'])==1,'major'] = 0
names.loc[(names['capt'])==1,'mr'] = 1
names.loc[(names['capt'])==1,'capt'] = 0
names.loc[(names['dr'])==1,'mr'] = 1
names.loc[(names['dr'])==1,'dr'] = 0
names.loc[(names['rev'])==1,'mr'] = 1
names.loc[(names['rev'])==1,'rev'] = 0
print(names.sum())
print(names.sum().sum())


# In[ ]:


names = names[['master','mr','miss','mrs']]
combined = combined.drop(['Name'],1)
combined = pd.concat([combined,names],axis=1)
train = combined.loc[train.index]


# In[ ]:


f, ax = plt.subplots(figsize=(12,10))
plt.title("Pearson Correlation of Features", y=1.02, size=15)
sns.heatmap(train.corr(),vmax=.6,cmap="RdBu_r",annot=True,fmt='.2f',square=True)


# In[ ]:


print(train.groupby([train['mr'],train['miss'],train['mrs'],train['master']])['Survived'].mean())


# In[ ]:


# Empty Age entries were estimated by median age of the Pclass that the passenger belonged to.
# This assumption/step could have been even more precise but the expected differences made the payoff \
# low

print(train.isnull().any())
print(train.isnull().sum())

combined.loc[(combined['Pclass'])==3,'Age'] = combined.loc[(combined['Pclass'])==3,'Age'].fillna(combined.loc[(combined['Pclass'])==3,'Age'].median())
combined.loc[(combined['Pclass'])==2,'Age'] = combined.loc[(combined['Pclass'])==2,'Age'].fillna(combined.loc[(combined['Pclass'])==2,'Age'].median())
combined.loc[(combined['Pclass'])==1,'Age'] = combined.loc[(combined['Pclass'])==1,'Age'].fillna(combined.loc[(combined['Pclass'])==1,'Age'].median())
train = combined.loc[train.index]

print(train.isnull().any())
print(train.head())


# # 4. Modeling

# In[ ]:


X = train.drop(['Survived'],1)
y = train.Survived


# In[ ]:


lreg = LogisticRegression()
lreg_yhat= lreg.fit(X, y).predict(X)

lreg_sas = accuracy_score(y, lreg_yhat)
lreg_cv5s = cross_val_score(lreg, X, y, cv=5, n_jobs=-1).mean()
lreg_l1os = cross_val_score(lreg, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(lreg_sas))
print('CV5 Score : {}'.format(lreg_cv5s))
print('CVLeave1Out Score : {}'.format(lreg_l1os))

lreg_pvsa_survival = np.column_stack((cross_val_predict(lreg, X, y, cv=5, n_jobs=-1), y))
print('Predicted Survival : {}'.format(lreg_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(lreg_pvsa_survival[:,1].mean()))
print(classification_report(y, lreg_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,lreg_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('LREG - Survival - Confusion Matrix')


# In[ ]:


# Logistic regression is susceptible to multicolinearity and since I wanted to study the weight of \
# the coefficients I decided to remove all features that had high VIF scores
# The new coefficients more accuratly display the true weights

print(pd.DataFrame(list(zip(X.columns, np.transpose(lreg.fit(X, y).coef_)))))

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in                range(X[variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[variables]
#credit to SpanishBoy & Prashant on stackexchange

Xdelta = calculate_vif_(X)
print(pd.DataFrame(list(zip(Xdelta.columns, np.transpose(lreg.fit(Xdelta, y).coef_)))))


# In[ ]:


gnb = GaussianNB()
gnb_yhat = gnb.fit(X, y).predict(X)

gnb_sas = accuracy_score(y, gnb_yhat)
gnb_cv5s = cross_val_score(gnb, X, y, cv=5, n_jobs=-1).mean()
gnb_l1os = cross_val_score(gnb, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(gnb_sas))
print('CV5 Score : {}'.format(gnb_cv5s))
print('CVLeave1Out Score : {}'.format(gnb_l1os))

gnb_pvsa_survival = np.column_stack((cross_val_predict(gnb, X, y, cv=5 , n_jobs=-1), y))
print('Predicted Survival : {}'.format(gnb_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(gnb_pvsa_survival[:,1].mean()))
print(classification_report(y, gnb_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,gnb_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('GNB - Survival - Confusion Matrix')


# In[ ]:


lsvc = SVC(kernel='linear', C=1)
lsvc_yhat= lsvc.fit(X, y).predict(X)

lsvc_sas = accuracy_score(y, lsvc_yhat)
lsvc_cv5s = cross_val_score(lsvc, X, y, cv=5, n_jobs=-1).mean()
lsvc_l1os = cross_val_score(lsvc, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(lsvc_sas))
print('CV5 Score : {}'.format(lsvc_cv5s))
print('CVLeave1Out Score : {}'.format(lsvc_l1os))

lsvc_pvsa_survival = np.column_stack((cross_val_predict(lsvc, X, y, cv=5, n_jobs=-1), y))
print('Predicted Survival : {}'.format(lsvc_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(lsvc_pvsa_survival[:,1].mean()))
print(classification_report(y, lsvc_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,lsvc_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('LSVC - Survival - Confusion Matrix')


# In[ ]:


rfc = RandomForestClassifier()
rfc_yhat = rfc.fit(X, y).predict(X)

rfc_sas = accuracy_score(y, rfc_yhat)
rfc_cv5s = cross_val_score(rfc, X, y, cv=5, n_jobs=-1).mean()
rfc_l1os = cross_val_score(rfc, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(rfc_sas))
print('CV5 Score : {}'.format(rfc_cv5s))
print('CVLeave1Out Score : {}'.format(rfc_l1os))

rfc_pvsa_survival = np.column_stack((cross_val_predict(rfc, X, y, cv=5, n_jobs=-1), y))
print('Predicted Survival : {}'.format(rfc_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(rfc_pvsa_survival[:,1].mean()))
print(classification_report(y, rfc_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,rfc_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('RFC - Survival - Confusion Matrix')


# In[ ]:


bc = BaggingClassifier()
bc_yhat = bc.fit(X,y).predict(X)

bc_sas = accuracy_score(y, bc_yhat)
bc_cv5s = cross_val_score(bc, X, y, cv=5, n_jobs=-1).mean()
bc_l1os = cross_val_score(bc, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(bc_sas))
print('CV5 Score : {}'.format(bc_cv5s))
print('CVLeave1Out Score : {}'.format(bc_l1os))

bc_pvsa_survival = np.column_stack((cross_val_predict(bc, X, y, cv=5, n_jobs=-1), y))
print('Predicted Survival : {}'.format(bc_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(bc_pvsa_survival[:,1].mean()))
print(classification_report(y, bc_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,bc_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('BC - Survival - Confusion Matrix')


# In[ ]:


gbc = GradientBoostingClassifier()
gbc_yhat = gbc.fit(X,y).predict(X)

gbc_sas = accuracy_score(y, gbc_yhat)
gbc_cv5s = cross_val_score(gbc, X, y, cv=5, n_jobs=-1).mean()
gbc_l1os = cross_val_score(gbc, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(gbc_sas))
print('CV5 Score : {}'.format(gbc_cv5s))
print('CVLeave1Out Score : {}'.format(gbc_l1os))

gbc_pvsa_survival = np.column_stack((cross_val_predict(gbc, X, y, cv=5, n_jobs=-1), y))
print('Predicted Survival : {}'.format(gbc_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(gbc_pvsa_survival[:,1].mean()))
print(classification_report(y, gbc_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,gbc_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('GBC - Survival - Confusion Matrix')


# In[ ]:


vote = VotingClassifier(estimators=[('lreg', lreg), ('gbc', gbc), ('bc', bc), ('rfc', rfc), 
	('lsvc', lsvc), ('gnb', gnb)], voting='hard')
vote_yhat = vote.fit(X,y).predict(X)

vote_sas = accuracy_score(y, vote_yhat)
vote_cv5s = cross_val_score(vote, X, y, cv=5, n_jobs=-1).mean()
vote_l1os = cross_val_score(vote, X, y, cv=LeaveOneOut().split(X), n_jobs=-1).mean()
print('Self Accuracy Score : {}'.format(vote_sas))
print('CV5 Score : {}'.format(vote_cv5s))
print('CVLeave1Out Score : {}'.format(vote_l1os))

vote_pvsa_survival = np.column_stack((cross_val_predict(vote, X, y, cv=5, n_jobs=-1), y))
print('Predicted Survival : {}'.format(vote_pvsa_survival[:,0].mean()))
print('Actual Survival : {}'.format(vote_pvsa_survival[:,1].mean()))
print(classification_report(y, vote_pvsa_survival[:,0], target_names=['dead','notdead']))

cm = confusion_matrix(y,vote_pvsa_survival[:,0])
ax = plt.axes()
sns.heatmap(cm, ax=ax, fmt='d', square=True, annot=True, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('GBC - Survival - Confusion Matrix')


# # 5. Prediction and Submission

# In[ ]:


gbc_scores = ['gbc', gbc_sas, gbc_cv5s, gbc_l1os, gbc_pvsa_survival[:,0].mean()]
bc_scores = ['bc', bc_sas, bc_cv5s, bc_l1os, bc_pvsa_survival[:,0].mean()]
lreg_scores = ['lreg', lreg_sas, lreg_cv5s, lreg_l1os, lreg_pvsa_survival[:,0].mean()]
rfc_scores = ['rfc', rfc_sas, rfc_cv5s, rfc_l1os, rfc_pvsa_survival[:,0].mean()]
lsvc_scores = ['lsvc', lsvc_sas, lsvc_cv5s, lsvc_l1os, lsvc_pvsa_survival[:,0].mean()]
gnb_scores = ['gnb', gnb_sas, gnb_cv5s, gnb_l1os, gnb_pvsa_survival[:,0].mean()]
vote_scores = ['vote', vote_sas, vote_cv5s, vote_l1os, vote_pvsa_survival[:,0].mean()]

classifier_comparison = pd.DataFrame([gbc_scores, bc_scores, lreg_scores, rfc_scores, 
	lsvc_scores, gnb_scores, vote_scores], columns=['Classifier','SAS','CV5S','L1OS',
	'Predicted Survival'])
print(classifier_comparison)


# In[ ]:


# Reindexed the test dataframe, and used the mean fare for the Pclass that the missing fare value \
# was in. 

test.set_index('PassengerId',drop=True,inplace=True)
test = combined.loc[test.index]
Xtest = test.drop(['Survived'],1)
Xtest.loc[(Xtest['Fare'].isnull()), 'Fare'] = combined.loc[(Xtest.loc[(Xtest['Fare'].isnull()), 'Pclass']), 'Fare'].mean()
test_gbc_yhat = gbc.fit(X,y).predict(Xtest)


# In[ ]:


submit = pd.DataFrame(list(zip(test.index, test_gbc_yhat)), columns = ['PassengerId', 'Survived'])
submit.to_csv("../working/submit.csv", index=False)
print(submit.tail())
print(submit.Survived.mean())

