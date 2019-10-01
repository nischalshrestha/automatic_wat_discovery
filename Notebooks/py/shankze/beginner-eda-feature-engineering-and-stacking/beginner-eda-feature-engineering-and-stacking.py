#!/usr/bin/env python
# coding: utf-8

# This is my first submission here. Any kind of feedback will be highly appreciated. I have used stacking here. Stacking is an ensemble technique where predictions from multiple models are used to generate a second level model called meta-model. This second-layer algorithm is trained to optimally combine the model predictions to form a new set of predictions. There are many good resources online that explain this concept in detail.
# 
# In this kernel, I have tried multiple classification algorithms. These models are called base models. The predictions from these base models serve as the input to the second-level model called the meta-model. I have used XGBoost for fitting my meta-model. To predict the results, I used the base models to generate first-level predictions on test set. These first level predictions from the base models were used as the input to the meta model. 
# 
# This kernel is a combination of multiple approaches that I have learned through various online courses and books.  
# 
# First, we will import the libraries: 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


import os
print(os.listdir("../input"))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


print(len(train))
print(len(test))


# ## Missing Values

# In[ ]:


print(train.isnull().sum())


# In[ ]:


print(test.isnull().sum())


# In[ ]:


plt.figure(figsize=(12,5))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


plt.figure(figsize=(12,5))
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# ## Impute Missing Values

# Age and Cabin columns contain many missing values. 77% of the observations for Cabin do not contain a value. ML algorithms will not be able to handle such a large number of missing values. We will drop the Cabin column.

# In[ ]:


train.drop(['Cabin'], axis =1, inplace=True)
test.drop(['Cabin'], axis = 1, inplace=True)

combined = pd.concat([train,test])

sns.heatmap(combined.corr(),annot=True)


# There is a high correlation between Age and PClass. I will use PClass to impute Age, where it is misisng.

# In[ ]:


combined.groupby('Pclass').mean()['Age']


# In[ ]:


def setAge(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass == 1:
            return 39
        elif pclass == 2:
            return 30
        else:
            return 25
    else:
        return age


# In[ ]:


train['Age'] = train[['Age','Pclass']].apply(setAge,axis=1)
test['Age'] = test[['Age','Pclass']].apply(setAge,axis=1)

combined['Embarked'].value_counts()


# As Southampton is the most common value for port emabrked, I will replace the misisng values with 'S'

# In[ ]:


train['Embarked'] = train['Embarked'].replace(np.NaN, 'S') 
test['Embarked'] = test['Embarked'].replace(np.NaN, 'S') 


# There is one observation in the Train data set that is misisng Fare information. I will set the value based on the mean  of 3rd class fare.

# In[ ]:


combined.groupby('Pclass').mean()['Fare']


# In[ ]:


test[test['Fare'].isnull()]


# In[ ]:


test["Fare"].fillna(13.30, inplace=True)


# Verifying there are no missing values

# In[ ]:



print(train.isnull().sum())


# In[ ]:


print(test.isnull().sum())


# The above steps resolve all the missing values. We no longer have any missing values.

# ## EDA & Feature Engineering

# This is my favorite phase of any data science project. I feel it is important to analyze all variables in the data set. Most of my analysis is limited to the target variable 'Survived' but I have included some analysis on relationsips between other columns. 

# ### Sex

# In[ ]:


sns.countplot(train['Survived'],palette= {1: "#1ab188", 0: "#c22250"})


# In[ ]:


sns.barplot(data=train,x='Sex',y='Survived',palette= {'male': "#3498db", 'female': "#ffe1ff"})


# In[ ]:


sns.countplot(data=train,x='Sex',hue='Survived',palette= {1: "#1ab188", 0: "#c22250"})


# ### Parch, SibSp and Family Size

# In[ ]:


sns.barplot(data=train,x='Parch',y='Survived')


# In[ ]:


sns.barplot(data=train,x='SibSp',y='Survived')


# As the SibSp and Parch columns are not very different, I will combine them into a single column called Family Size

# In[ ]:


train['FamilySize'] = train['SibSp'] + train ['Parch']
test['FamilySize'] = test['SibSp'] + test ['Parch']

sns.barplot(data=train,x='FamilySize',y='Survived')


# Survival probability is low for single travelers. So I will create a column to identify the solo travellers.

# In[ ]:


def isAlone(cols):
    if (cols[0]==0) & (cols[1]==0):
        return 1
    else:
        return 0
train['IsAlone'] = train[['SibSp','Parch']].apply(isAlone,axis=1)
test['IsAlone'] = test[['SibSp','Parch']].apply(isAlone,axis=1)

sns.barplot(data=train,x='IsAlone',y='Survived')


# In[ ]:


sns.countplot(data=train,x='IsAlone',hue='Survived',palette= {1: "#1ab188", 0: "#c22250"})


# In[ ]:


combined = pd.concat([train,test])
sns.countplot(data=combined,x='Sex',hue='IsAlone')


# ### Age

# In[ ]:


f = sns.FacetGrid(combined,hue='IsAlone',size=5,aspect=4)
f.map(sns.kdeplot,'Age',shade= True)
f.add_legend()


# In[ ]:


combined[(combined['IsAlone'] == True)].sort_values(['Age']).head()


# The 2 plots above may not be of much significance to this challenge but I find them interesting. We observe that most female travelers had a family member travelling with them. Most of the travelers sailing alone were in the age group of 17-40. The youngest solo traveler was just 5 years and she survived. This piqued my curiosity and I found that she was travelling with a nursemaid. More information about Virginia Ethel: https://www.encyclopedia-titanica.org/titanic-survivor/virginia-ethel-emanuel.html. 
# The second youngest solo traveler was not that fortunate. He was travelling with a relative. That's it for now, lets continue with the challenge.

# In[ ]:


sns.distplot(train['Age'].dropna(),kde=False,bins=60)


# In[ ]:


f = sns.FacetGrid(train,hue='Survived',size=5,aspect=4)
f.map(sns.kdeplot,'Age',shade= True)
f.add_legend()


# ### P Class and Fare

# The chance of survival increased with class. Around 60% of first class passengers survived.Only 23% of third class passengers survived.

# In[ ]:


sns.barplot(data=train,x='Pclass',y='Survived',palette= {1: "#117A65", 2: "#52BE80",3: "#ABEBC6"})


# In[ ]:


f = sns.FacetGrid(combined,hue='Pclass',size=5,aspect=4)
f.map(sns.kdeplot,'Age',shade= True)
f.add_legend()


# This above graph is interesting but expected.

# In[ ]:


f = sns.FacetGrid(combined,hue='Pclass',size=5,aspect=4)
plt.xlim(0, 300)
f.map(sns.kdeplot,'Fare',shade= True)
f.add_legend()


# In[ ]:


f = sns.FacetGrid(combined,hue='Pclass',size=5,aspect=4)
plt.xlim(0, 50)
f.map(sns.kdeplot,'Fare',shade= True)
f.add_legend()


# In[ ]:


plt.figure(figsize=(12,5))
combined['FareBucket'] = (combined['Fare']/50).astype(int)*50
sns.countplot(data=combined,x='FareBucket',hue='Pclass',palette= {1: "#117A65", 2: "#52BE80",3: "#ABEBC6"})


# We see from the above 3 graphs that there were a few first class passengers/tickets who did not pay for their tickets.

# In[ ]:


combined[combined['Pclass']==1].sort_values('Fare').head()


# 1. This is interesting. I looked it up online and found that many passengers were given complimentary tickets. So let's find the lowest revenue first class ticket.

# In[ ]:


combined[(combined['Pclass']==1)&(combined['Fare']>0)].sort_values('Fare').head()


# Frans Olof paid just 5 pounds for his first class ticket. The encyclopedia-titanica article about him says that his company bought his ticket. I wish I knew his travel agent. Many first class passengers paid around 25 pounds which is considerably cheaper than the most expensive 3rd class ticket!
# 
# I am curious about the most expensive 3rd class ticket. Let's have a look.

# In[ ]:


combined[combined['Pclass']==3].sort_values('Fare',ascending=False).head(5)


# The highest 3rd class fares are associated with Ticket No CA. 2343.  11 passengers traveled on this ticket and we know that  7 of them did not survive. We do not know the fate of 4 who are the test set. This is unfortunate but reveals a very important charateristic about our data. I had ignored the 'Ticket' column but I should not ignore it. Looks like they are from the same family. I hope someone from the Sage family survived. 

# In[ ]:


combined[combined['Ticket']=='CA. 2343']


# I did some researching online for more information about their expensive 3rd class fare but I was not able to find anything concrete. There is a mention of the family changing plans (to sail aboard the Titanic instead of Philadelphia) due to the coal strike. Maybe they ended up buying these expensive tickets due to the late change of plans.
# 
# I am curious about what happened to the remaining 4 members of the Sage family. So I will create a temporaty dataframe to store their info including Passenger ID

# In[ ]:


tst_sageFamily = test[test['Ticket']=='CA. 2343']
tst_sageFamily


# ### Ticket Number

# There are 40 groups with more than 2 passengers in the trainset. I will first find the survival probability for each ticket number.

# In[ ]:


ticCount = train.groupby('Ticket')['Sex'].count()
ticSurN = train.groupby('Ticket')['Survived'].sum()
ticCount = pd.DataFrame(ticCount)
ticSurN = pd.DataFrame(ticSurN)
ticSur = ticCount.join(ticSurN)
ticSur['TicSurvProb'] = ticSur['Survived']*(100) /(ticSur['Sex'])

ticSur.rename(index=str, columns={"Sex": "PassengerCount", "Survived": "PassengersSurvived"},inplace=True)
ticSur.reset_index(level=0, inplace=True)
ticSur.head()


# I then create a column called TicSurvProb to store the probability of survival of passengers who are in groups and have these ticket numbers. Passengers who do not posess these ticket numbers are assigned the mean value.

# In[ ]:


ticSur = ticSur[ticSur['PassengerCount'] > 2]
train = pd.merge(train, ticSur, on=['Ticket', 'Ticket'],how='left')
train['TicSurvProb'] = train['TicSurvProb'].replace(np.NaN, 38.38)


# In[ ]:


test = pd.merge(test, ticSur, on=['Ticket', 'Ticket'],how='left')
test['TicSurvProb'] = test['TicSurvProb'].replace(np.NaN, 38.38)


# In[ ]:


train.drop(['PassengerCount','PassengersSurvived','Ticket'],axis=1,inplace=True)
test.drop(['PassengerCount','PassengersSurvived','Ticket'],axis=1,inplace=True)


# ## Categorical Columns

# In[ ]:


train = pd.get_dummies(train, columns=['Embarked'])
test = pd.get_dummies(test, columns=['Embarked'])  

train['Sex'] = train['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
test['Sex'] = test['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# ## Binning

# Here, I split the Fare and Age columns to bins based on value. I can also create quantile bins using the pd.qcut function but I got better results by binning based on value.

# In[ ]:


train['FareBucket'] = (train['Fare']/50).astype(int)
test['FareBucket'] = (test['Fare']/50).astype(int)


# In[ ]:


train['AgeBand'] = (train['Age']/5).astype(int)
test['AgeBand'] = (test['Age']/5).astype(int)


# ## Correlation

# In[ ]:


train.drop(['Age','Fare','PassengerId','Name','SibSp','Parch'], axis =1, inplace=True)
test.drop(['Age','Fare','Name','SibSp','Parch'], axis = 1, inplace=True)


# In[ ]:


p = sns.pairplot(train[['Survived', 'Pclass', 'Sex', 'FamilySize', 'FareBucket','AgeBand', 'IsAlone']],hue='Survived', diag_kind = 'kde',palette= {1: "#1ab188", 0: "#c22250"} )
p.set(xticklabels=[])


# In[ ]:


plt.figure(figsize=(14,10))
sns.heatmap(train.corr(),annot=True)


# The column 'Sex' has the highest correlation with Survived followed by 'TicSurvProb'.

# In[ ]:


pd.DataFrame(train.corr()['Survived']).abs().sort_values('Survived',ascending=False)


# In[ ]:


X = train.drop(['Survived'],axis=1)
y = train['Survived']


# ## Model Data

# I will first get the accuracy of each model separately using Cross Validation. I will use the entire train data to perform Cross Validation. I may get better results with KNN and SVC if I scale the data but I have skipped that step.

# In[ ]:


from sklearn.model_selection import KFold 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
num_of_estimators = 500
rfClass = RandomForestClassifier(n_estimators=200,max_depth=3,
                                 min_samples_leaf= 1,max_features=5,min_samples_split=3,criterion='entropy')
logClass = LogisticRegression(penalty='l1',C=21.544346900318832)
svcClass = SVC(gamma=0.001,C=10)
knnClass = KNeighborsClassifier(n_neighbors=9)
xgbClass = xgb.XGBClassifier(n_estimators=100,colsample_bytree= 0.8, gamma=1, max_depth=5, min_child_weight=1, subsample=1.0)
nbClass = MultinomialNB()
adaClass = AdaBoostClassifier(n_estimators=20,learning_rate=0.2)
extraTreesClass = ExtraTreesClassifier(n_estimators=50,bootstrap=False,criterion='entropy',max_features=3,min_samples_leaf=3,
                                        min_samples_split=10,max_depth=None)
gradientBClass = GradientBoostingClassifier(n_estimators=20,max_depth=3,max_features= 5,min_samples_leaf=3,min_samples_split=2)
res_alg = ['Random Forest','Logistic Regression','SVC','KNN','XG Boost','Naive Bayes','ADA Boost','Extra Trees','Gradient Boost']
res_acc = []
res_acc.append(cross_val_score(rfClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(logClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(svcClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(knnClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(xgbClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(nbClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(adaClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(extraTreesClass,X,y,scoring='accuracy',cv=10).mean()*100)
res_acc.append(cross_val_score(gradientBClass,X,y,scoring='accuracy',cv=10).mean()*100)


# In[ ]:


cv_results = pd.DataFrame({'Algorithm':res_alg,'Accuracy':res_acc})
cv_results.sort_values('Accuracy',ascending=False)


# In[ ]:


plt.figure(figsize=(12,8))
cv_results = cv_results.sort_values(['Accuracy'],ascending=False).reset_index(drop=True)
sns.set(style="whitegrid")
sns.barplot(data=cv_results,x='Accuracy',y='Algorithm')


# ## Stacking

# In[ ]:


from sklearn.model_selection import train_test_split


# I will divide the training data set into 3 subsets - s_train, s_valid and s_test. s_train will be used to train the base models. I will use these base models to make predictions on the s_valid dataset. I will then make a data frame of all the predictions and this will server as the training data to the meta model. s_test will be used to test the model.

# Divide into train and a temporary test set

# In[ ]:


X_s_train, X_s_test2, y_s_train, y_s_test2 = train_test_split(X, y, test_size=0.6,random_state=101)


# Divide the temporary test set into validate and test

# In[ ]:


X_s_valid, X_s_test, y_s_valid, y_s_test = train_test_split(X_s_test2, y_s_test2, test_size=0.2,random_state=101)


# In[ ]:


print(len(X_s_train))
print(len(X_s_valid)) 
print(len(X_s_test)) 


# Train the base models with s_train

# In[ ]:


rfClass.fit(X_s_train,y_s_train)
adaClass.fit(X_s_train,y_s_train)
extraTreesClass.fit(X_s_train,y_s_train)
logClass.fit(X_s_train,y_s_train)
xgbClass.fit(X_s_train,y_s_train)
svcClass.fit(X_s_train,y_s_train)
knnClass.fit(X_s_train,y_s_train)
gradientBClass.fit(X_s_train,y_s_train)

#predict these models on validate data
vld_rfPred = rfClass.predict(X_s_valid)
vld_adaPred = adaClass.predict(X_s_valid)
vld_extPred = extraTreesClass.predict(X_s_valid)
vld_logPred = logClass.predict(X_s_valid)
vld_xgbPred = xgbClass.predict(X_s_valid)
vld_svcPred = svcClass.predict(X_s_valid)
vld_knnPred = knnClass.predict(X_s_valid)
vld_gbPred =  gradientBClass.predict(X_s_valid)


# In[ ]:


base_predictions_train = pd.DataFrame( {
    'RandomForest': vld_rfPred,
    'AdaptiveBoost': vld_adaPred,
    'ExtraTrees': vld_extPred,
    'Log': vld_logPred,
    'XGB': vld_xgbPred,  
    'SVC': vld_svcPred,
    'KNN': vld_knnPred,
    'GB' : vld_gbPred,
    'Y': y_s_valid,
    })
base_predictions_train.head()
sns.heatmap(base_predictions_train.corr(),annot=True)


# In[ ]:


#Concatenate all predictions on Validate
stacked_valid_predictions = np.column_stack((vld_rfPred, vld_adaPred, vld_extPred,vld_logPred,vld_xgbPred,
                                             vld_svcPred,vld_knnPred,vld_gbPred))


# Create meta model

# In[ ]:


meta_model = xgb.XGBClassifier(n_estimators=90,colsample_bytree=0.8, gamma=5, max_depth=3,
                                      min_child_weight=10, subsample=0.6)


# Fit meta model on Validate subset

# In[ ]:


meta_model.fit(stacked_valid_predictions,y_s_valid)

feature_importances = pd.DataFrame(meta_model.feature_importances_,index = ['vld_rfPred', 'vld_adaPred', 'vld_extPred','vld_logPred','vld_xgbPred',
                                             'vld_svcPred','vld_knnPred','vld_gbPred'],columns=['importance']).sort_values('importance',   ascending=False)
feature_importances


# Use Base Models to predict on s_test set

# In[ ]:


tst_rfPred = rfClass.predict(X_s_test)
tst_adaPred = adaClass.predict(X_s_test)
tst_extPred = extraTreesClass.predict(X_s_test)
tst_logPred = logClass.predict(X_s_test)
tst_xgbPred = xgbClass.predict(X_s_test)
tst_svcPred = svcClass.predict(X_s_test)
tst_knnPred = knnClass.predict(X_s_test)
tst_gbPred = gradientBClass.predict(X_s_test)

#Concatenate base model predictions on Test
stacked_test_predictions = np.column_stack((tst_rfPred, tst_adaPred, tst_extPred,tst_logPred,tst_xgbPred,
                                           tst_svcPred,tst_knnPred,tst_gbPred))


# Use the predictions from the above step as input to the meta model

# In[ ]:


s_test_pred = meta_model.predict(stacked_test_predictions)

from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_s_test,s_test_pred))
print(confusion_matrix(y_s_test,s_test_pred))


# This shows that stacking provides better performance than separate models.

#  Predictions on Test Data for submission

# In[ ]:


X_test = test.drop(['PassengerId'],axis=1)

#Use the base models to make predictions on test set
t_rfPred = rfClass.predict(X_test)
t_adaPred = adaClass.predict(X_test)
t_extPred = extraTreesClass.predict(X_test)
t_logPred = logClass.predict(X_test)
t_xgbPred = xgbClass.predict(X_test)
t_svcPred = svcClass.predict(X_test)
t_knnPred = knnClass.predict(X_test)
t_gbPred = gradientBClass.predict(X_test)

#Concatenate base model predictions on Test
stacked_t_predictions = np.column_stack((t_rfPred, t_adaPred, t_extPred,t_logPred,t_xgbPred,t_svcPred,t_knnPred,t_gbPred))
#Use the meta model to make predictions on test set
final_pred = meta_model.predict(stacked_t_predictions)

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": final_pred
    })


# In[ ]:


tst_sageFamily.merge(submission,how='left',on='PassengerId')


# According to this model, no one from the Sage family survived and this is what actually happened. This page contains more information: https://www.bbc.com/news/uk-england-cambridgeshire-17596264

# In[ ]:


submission.to_csv('titanic_output.csv', index=False)


# In[ ]:




