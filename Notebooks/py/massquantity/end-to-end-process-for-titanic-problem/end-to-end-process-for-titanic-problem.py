#!/usr/bin/env python
# coding: utf-8

# # Content

# + Data Cleaning
# + Exploratory Visualization
# + Feature Engineering
# + Basic Modeling & Evaluation
# + Hyperparameters tuning
# + Ensemble Methods

# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[ ]:


plt.style.use('ggplot')


# In[ ]:


# combine train and test set.
train = pd.read_csv( '../input/train.csv')
test = pd.read_csv( '../input/test.csv')
full=pd.concat([train,test],ignore_index=True)


# In[ ]:


full.head()


# # Data Cleaning

# In[ ]:


full.isnull().sum()


# __The 'Age', 'Cabin', 'Embarked', 'Fare' columns have missing values. First we fill the missing 'Embarked' with the mode.__

# In[ ]:


full.Embarked.mode()


# In[ ]:


full['Embarked'].fillna('S',inplace=True)


# __Since 'Fare' is mainly related to 'Pclass', we should check which class this person belongs to.__

# In[ ]:


full[full.Fare.isnull()]


# __It's a passenger from Pclass 3, so we'll fill the missing value with the median fare of Pclass 3.__

# In[ ]:


full.Fare.fillna(full[full.Pclass==3]['Fare'].median(),inplace=True)


# **There are a lot of missing values in 'Cabin', maybe there is difference between the survival rate of people who has Cabin number and those who hasn't.**

# In[ ]:


full.loc[full.Cabin.notnull(),'Cabin']=1
full.loc[full.Cabin.isnull(),'Cabin']=0


# In[ ]:


full.Cabin.isnull().sum()


# In[ ]:


pd.pivot_table(full,index=['Cabin'],values=['Survived']).plot.bar(figsize=(8,5))
plt.title('Survival Rate')


# __We can also plot the count of 'Cabin' to see some patterns.__

# In[ ]:


cabin=pd.crosstab(full.Cabin,full.Survived)
cabin.rename(index={0:'no cabin',1:'cabin'},columns={0.0:'Dead',1.0:'Survived'},inplace=True)
cabin


# In[ ]:


cabin.plot.bar(figsize=(8,5))
plt.xticks(rotation=0,size='xx-large')
plt.title('Survived Count')
plt.xlabel('')
plt.legend()


# __From the plot, we can conclude that there is far more chance for a passenger to survive if we know his/her 'Cabin'.__

# ### Extract Title from 'Name'

# In[ ]:


full['Title']=full['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())


# In[ ]:


full.Title.value_counts()


# In[ ]:


pd.crosstab(full.Title,full.Sex)


# __All the 'Title' belongs to one kind of gender except for 'Dr'.__

# In[ ]:


full[(full.Title=='Dr')&(full.Sex=='female')]


# __So the PassengerId of the female 'Dr' is '797'. Then we map the 'Title'.__

# In[ ]:


nn={'Capt':'Rareman', 'Col':'Rareman','Don':'Rareman','Dona':'Rarewoman',
    'Dr':'Rareman','Jonkheer':'Rareman','Lady':'Rarewoman','Major':'Rareman',
    'Master':'Master','Miss':'Miss','Mlle':'Rarewoman','Mme':'Rarewoman',
    'Mr':'Mr','Mrs':'Mrs','Ms':'Rarewoman','Rev':'Mr','Sir':'Rareman',
    'the Countess':'Rarewoman'}


# In[ ]:


full.Title=full.Title.map(nn)


# In[ ]:


# assign the female 'Dr' to 'Rarewoman'
full.loc[full.PassengerId==797,'Title']='Rarewoman'


# In[ ]:


full.Title.value_counts()


# In[ ]:


full[full.Title=='Master']['Sex'].value_counts()


# In[ ]:


full[full.Title=='Master']['Age'].describe()


# In[ ]:


full[full.Title=='Miss']['Age'].describe()


# + __'Master' mainly stands for little boy, but we also want to find little girl. Because children tend to have higher survival rate.__

# + __For the 'Miss' with a Age record, we can simply determine whether a 'Miss' is a little girl by her age.__

# + __For the 'Miss' with no Age record, we use (Parch!=0). Since if it's a little girl, she was very likely to be accompanied by parents.__

# We'll create a function to filter the girls. The function can't be used if 'Age' is Nan, so first we fill the missing values with '999'.

# In[ ]:


full.Age.fillna(999,inplace=True)


# In[ ]:


def girl(aa):
    if (aa.Age!=999)&(aa.Title=='Miss')&(aa.Age<=14):
        return 'Girl'
    elif (aa.Age==999)&(aa.Title=='Miss')&(aa.Parch!=0):
        return 'Girl'
    else:
        return aa.Title


# In[ ]:


full['Title']=full.apply(girl,axis=1)


# In[ ]:


full.Title.value_counts()


# __Next we fill the missing 'Age' according to their 'Title'.__

# In[ ]:


full[full.Age==999]['Age'].value_counts()


# In[ ]:


Tit=['Mr','Miss','Mrs','Master','Girl','Rareman','Rarewoman']
for i in Tit:
    full.loc[(full.Age==999)&(full.Title==i),'Age']=full.loc[full.Title==i,'Age'].median()


# In[ ]:


full.info()


# ### Finally, there is no missing value now!!!

# # Exploratory Visualization

# In[ ]:


full.head()


# __Let's first check whether the Age of each Title is reasonable.__

# In[ ]:


full.groupby(['Title'])[['Age','Title']].mean().plot(kind='bar',figsize=(8,5))
plt.xticks(rotation=0)
plt.show()


# __As we can see, female has a much larger survival rate than male.__

# In[ ]:


pd.crosstab(full.Sex,full.Survived).plot.bar(stacked=True,figsize=(8,5),color=['#4169E1','#FF00FF'])
plt.xticks(rotation=0,size='large')
plt.legend(bbox_to_anchor=(0.55,0.9))


# __ We can also check the relationship between 'Age' and 'Survived'.__

# In[ ]:


agehist=pd.concat([full[full.Survived==1]['Age'],full[full.Survived==0]['Age']],axis=1)
agehist.columns=['Survived','Dead']
agehist.head()


# In[ ]:


agehist.plot(kind='hist',bins=30,figsize=(15,8),alpha=0.3)


# In[ ]:


farehist=pd.concat([full[full.Survived==1]['Fare'],full[full.Survived==0]['Fare']],axis=1)
farehist.columns=['Survived','Dead']
farehist.head()


# In[ ]:


farehist.plot.hist(bins=30,figsize=(15,8),alpha=0.3,stacked=True,color=['blue','red'])


# __People with high 'Fare' are more likely to survive, though most 'Fare' are under 100.__

# In[ ]:


full.groupby(['Title'])[['Title','Survived']].mean().plot(kind='bar',figsize=(10,7))
plt.xticks(rotation=0)


# __The 'Rarewoman' has 100% survival rate, that's amazing!!__

# __It's natural to assume that 'Pclass' also plays a big part, as we can see from the plot below. The females in class 3 have a survival rate of about 50%, while survival rateof females from class1/2 are much higher.__

# In[ ]:


fig,axes=plt.subplots(2,3,figsize=(15,8))
Sex1=['male','female']
for i,ax in zip(Sex1,axes):
    for j,pp in zip(range(1,4),ax):
        PclassSex=full[(full.Sex==i)&(full.Pclass==j)]['Survived'].value_counts().sort_index(ascending=False)
        pp.bar(range(len(PclassSex)),PclassSex,label=(i,'Class'+str(j)))
        pp.set_xticks((0,1))
        pp.set_xticklabels(('Survived','Dead'))
        pp.legend(bbox_to_anchor=(0.6,1.1))


# # Feature Engeneering

# In[ ]:


# create age bands
full.AgeCut=pd.cut(full.Age,5)


# In[ ]:


# create fare bands
full.FareCut=pd.qcut(full.Fare,5)


# In[ ]:


full.AgeCut.value_counts().sort_index()


# In[ ]:


full.FareCut.value_counts().sort_index()


# In[ ]:


# replace agebands with ordinals
full.loc[full.Age<=16.136,'AgeCut']=1
full.loc[(full.Age>16.136)&(full.Age<=32.102),'AgeCut']=2
full.loc[(full.Age>32.102)&(full.Age<=48.068),'AgeCut']=3
full.loc[(full.Age>48.068)&(full.Age<=64.034),'AgeCut']=4
full.loc[full.Age>64.034,'AgeCut']=5


# In[ ]:


full.loc[full.Fare<=7.854,'FareCut']=1
full.loc[(full.Fare>7.854)&(full.Fare<=10.5),'FareCut']=2
full.loc[(full.Fare>10.5)&(full.Fare<=21.558),'FareCut']=3
full.loc[(full.Fare>21.558)&(full.Fare<=41.579),'FareCut']=4
full.loc[full.Fare>41.579,'FareCut']=5


# __We can see from the plot that 'FareCut' has a big impact on survial rate.__

# In[ ]:


full[['FareCut','Survived']].groupby(['FareCut']).mean().plot.bar(figsize=(8,5))


# In[ ]:


full.corr()


# __We haven't gererate any feature from 'Parch','Pclass','SibSp','Title', so let's do this by using pivot table.__

# In[ ]:


full[full.Survived.notnull()].pivot_table(index=['Title','Pclass'],values=['Survived']).sort_values('Survived',ascending=False)


# In[ ]:


full[full.Survived.notnull()].pivot_table(index=['Title','Parch'],values=['Survived']).sort_values('Survived',ascending=False)


# #### _From the pivot tables above, there is definitely a relationship among 'Survived','Title','Pclass','Parch'. So we can combine them together._

# In[ ]:


# only choose the object with not null 'Survived'.
TPP=full[full.Survived.notnull()].pivot_table(index=['Title','Pclass','Parch'],values=['Survived']).sort_values('Survived',ascending=False)
TPP


# In[ ]:


TPP.plot(kind='bar',figsize=(16,10))
plt.xticks(rotation=40)
plt.axhline(0.8,color='#BA55D3')
plt.axhline(0.5,color='#BA55D3')
plt.annotate('80% survival rate',xy=(30,0.81),xytext=(32,0.85),arrowprops=dict(facecolor='#BA55D3',shrink=0.05))
plt.annotate('50% survival rate',xy=(32,0.51),xytext=(34,0.54),arrowprops=dict(facecolor='#BA55D3',shrink=0.05))


# __From the plot, we can draw some horizontal lines and make some classification. I only choose 80% and 50%, because I'm so afraid of overfitting.__

# In[ ]:


# use 'Title','Pclass','Parch' to generate feature 'TPP'.
Tit=['Girl','Master','Mr','Miss','Mrs','Rareman','Rarewoman']
for i in Tit:
    for j in range(1,4):
        for g in range(0,10):
            if full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g)&(full.Survived.notnull()),'Survived'].mean()>=0.8:
                full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g),'TPP']=1
            elif full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g)&(full.Survived.notnull()),'Survived'].mean()>=0.5:
                full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g),'TPP']=2
            elif full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g)&(full.Survived.notnull()),'Survived'].mean()>=0:
                full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g),'TPP']=3
            else: 
                full.loc[(full.Title==i)&(full.Pclass==j)&(full.Parch==g),'TPP']=4


# + __'TPP=1' means highest probability to survive, and 'TPP=3' means the lowest.__
# + __'TPP=4' means there is no such combination of (Title&Pclass&Pclass) in train set. Let's see what kind of combination it contains.__

# In[ ]:


full[full.TPP==4]


# __ We can simply classify them by 'Sex'&'Pclass'.__

# In[ ]:


full.ix[(full.TPP==4)&(full.Sex=='female')&(full.Pclass!=3),'TPP']=1
full.ix[(full.TPP==4)&(full.Sex=='female')&(full.Pclass==3),'TPP']=2
full.ix[(full.TPP==4)&(full.Sex=='male')&(full.Pclass!=3),'TPP']=2
full.ix[(full.TPP==4)&(full.Sex=='male')&(full.Pclass==3),'TPP']=3


# In[ ]:


full.TPP.value_counts()


# In[ ]:


full.info()


# # Basic Modeling & Evaluation

# In[ ]:


predictors=['Cabin','Embarked','Parch','Pclass','Sex','SibSp','Title','AgeCut','TPP','FareCut','Age','Fare']


# In[ ]:


# convert categorical variables to numerical variables
full_dummies=pd.get_dummies(full[predictors])


# In[ ]:


full_dummies.head()


# __We choose 7 models and use 5-folds cross-calidation to evaluate these models.__

# Models include:
# 
# + k-Nearest Neighbors
# + Logistic Regression
# + Naive Bayes classifier
# + Decision Tree
# + Random Forrest
# + Gradient Boosting Decision Tree
# + Support Vector Machine

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC


# In[ ]:


models=[KNeighborsClassifier(),LogisticRegression(),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(),
       GradientBoostingClassifier(),SVC()]


# In[ ]:


full.shape,full_dummies.shape


# In[ ]:


X=full_dummies[:891]
y=full.Survived[:891]
test_X=full_dummies[891:]


# __As some algorithms such as KNN and SVM are sensitive to the scaling of the data, here we also apply standard-scaling to the data.__

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler=StandardScaler()
X_scaled=scaler.fit(X).transform(X)
test_X_scaled=scaler.fit(X).transform(test_X)


# In[ ]:


# evaluate models by using cross-validation
names=['KNN','LR','NB','Tree','RF','GDBT','SVM']
for name, model in zip(names,models):
    score=cross_val_score(model,X,y,cv=5)
    print("{}:{},{}".format(name,score.mean(),score))


# In[ ]:


# used scaled data
names=['KNN','LR','NB','Tree','RF','GDBT','SVM']
for name, model in zip(names,models):
    score=cross_val_score(model,X_scaled,y,cv=5)
    print("{}:{},{}".format(name,score.mean(),score))


# __'k-Nearest Neighbors', 'Support Vector Machine' perform much better on scaled data__

# **Then we use (feature importances) in GradientBoostingClassifier to see which features are important.**

# In[ ]:


model=GradientBoostingClassifier()


# In[ ]:


model.fit(X,y)


# In[ ]:


model.feature_importances_


# In[ ]:


X.columns


# In[ ]:


fi=pd.DataFrame({'importance':model.feature_importances_},index=X.columns)


# In[ ]:


fi.sort_values('importance',ascending=False)


# In[ ]:


fi.sort_values('importance',ascending=False).plot.bar(figsize=(11,7))
plt.xticks(rotation=30)
plt.title('Feature Importance',size='x-large')


# __Based on the bar plot, 'TPP','Fare','Age' are the most important.__

# **Now let's think through this problem in another way. Our goal here is to improve the overall accuracy. This is equivalent to minimizing the misclassified observations. So if all the misclassified observations are found, maybe we can see the pattern and generate some new features.**

# **Again we use cross-validation to search for the miscalssified observations**

# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


kf=KFold(n_splits=10,random_state=1)


# In[ ]:


kf.get_n_splits(X)


# In[ ]:


print(kf)


# In[ ]:


# extract the indices of misclassified observations
rr=[]
for train_index, val_index in kf.split(X):
    pred=model.fit(X.ix[train_index],y[train_index]).predict(X.ix[val_index])
    rr.append(y[val_index][pred!=y[val_index]].index.values)


# In[ ]:


rr


# In[ ]:


# combine all the indices
whole_index=np.concatenate(rr)
len(whole_index)


# In[ ]:


full.ix[whole_index].head()


# In[ ]:


diff=full.ix[whole_index]


# In[ ]:


diff.describe()


# In[ ]:


diff.describe(include=['O'])


# In[ ]:


# both mean and count of 'survived' should be considered.
diff.groupby(['Title'])['Survived'].agg([('average','mean'),('number','count')])


# In[ ]:


diff.groupby(['Title','Pclass'])['Survived'].agg([('average','mean'),('number','count')])


# **It seems mainly the third class 'Miss'/'Mrs' and the first/third class 'Mr' are missclassified.**

# In[ ]:


diff.groupby(['Title','Pclass','Parch','SibSp'])['Survived'].agg([('average','mean'),('number','count')])


# Gererally, we should only pick the categories with relatively large numbers. That is:

# 1. **'Mr','Pclass 1','Parch 0','SibSp 0', 17**
# 2. **'Mr','Pclass 1','Parch 0','SibSp 1', 8**
# 3. **'Mr','Pclass 2/3','Parch 0','SibSp 0', 32+7**
# 4. **'Miss','Pclass 3','Parch 0','SibSp 0', 21**

# __Then we add new feature 'MPPS'.__

# In[ ]:


full.loc[(full.Title=='Mr')&(full.Pclass==1)&(full.Parch==0)&((full.SibSp==0)|(full.SibSp==1)),'MPPS']=1
full.loc[(full.Title=='Mr')&(full.Pclass!=1)&(full.Parch==0)&(full.SibSp==0),'MPPS']=2
full.loc[(full.Title=='Miss')&(full.Pclass==3)&(full.Parch==0)&(full.SibSp==0),'MPPS']=3
full.MPPS.fillna(4,inplace=True)


# In[ ]:


full.MPPS.value_counts()


# From the __feature-Importance__ plot we can see the 'Fare' is the most important feature, let's explore whether we can generate some new feature.

# In[ ]:


diff[(diff.Title=='Mr')|(diff.Title=='Miss')].groupby(['Title','Survived','Pclass'])[['Fare']].describe().unstack()


# In[ ]:


full[(full.Title=='Mr')|(full.Title=='Miss')].groupby(['Title','Survived','Pclass'])[['Fare']].describe().unstack()


# But there seems no big difference between the 'Fare' of 'diff' and 'full'.

# __Finally we could draw a corrlelation heatmap__

# In[ ]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=20)
sns.heatmap(full[['Cabin','Parch','Pclass','SibSp','AgeCut','TPP','FareCut','Age','Fare','MPPS','Survived']].astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# # Hyperparameters Tuning

# __Now let's do grid search for some algorithms. Since many algorithms performs better in scaled data, we will use scaled data.__

# In[ ]:


predictors=['Cabin','Embarked','Parch','Pclass','Sex','SibSp','Title','AgeCut','TPP','FareCut','Age','Fare','MPPS']
full_dummies=pd.get_dummies(full[predictors])
X=full_dummies[:891]
y=full.Survived[:891]
test_X=full_dummies[891:]

scaler=StandardScaler()
X_scaled=scaler.fit(X).transform(X)
test_X_scaled=scaler.fit(X).transform(test_X)


# In[ ]:


from sklearn.model_selection import GridSearchCV


# ### k-Nearest Neighbors

# In[ ]:


param_grid={'n_neighbors':[1,2,3,4,5,6,7,8,9]}
grid_search=GridSearchCV(KNeighborsClassifier(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# ### Logistic Regression

# In[ ]:


param_grid={'C':[0.01,0.1,1,10]}
grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# In[ ]:


# second round grid search
param_grid={'C':[0.04,0.06,0.08,0.1,0.12,0.14]}
grid_search=GridSearchCV(LogisticRegression(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# ### Support Vector Machine

# In[ ]:


param_grid={'C':[0.01,0.1,1,10],'gamma':[0.01,0.1,1,10]}
grid_search=GridSearchCV(SVC(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# In[ ]:


#second round grid search
param_grid={'C':[2,4,6,8,10,12,14],'gamma':[0.008,0.01,0.012,0.015,0.02]}
grid_search=GridSearchCV(SVC(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# ### Gradient Boosting Decision Tree

# In[ ]:


param_grid={'n_estimators':[30,50,80,120,200],'learning_rate':[0.05,0.1,0.5,1],'max_depth':[1,2,3,4,5]}
grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# In[ ]:


#second round search
param_grid={'n_estimators':[100,120,140,160],'learning_rate':[0.05,0.08,0.1,0.12],'max_depth':[3,4]}
grid_search=GridSearchCV(GradientBoostingClassifier(),param_grid,cv=5)

grid_search.fit(X_scaled,y)

grid_search.best_params_,grid_search.best_score_


# # Ensemble Methods 

# ## Bagging

# __We use logistic regression with the parameter we just tuned to apply bagging.__

# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


bagging=BaggingClassifier(LogisticRegression(C=0.06),n_estimators=100)


# ## VotingClassifier

# __We use five models to apply votingclassifier, namely logistic regression, random forest, gradient boosting decision,support vector machine and k-nearest neighbors.__

# In[ ]:


from sklearn.ensemble import VotingClassifier


# In[ ]:


clf1=LogisticRegression(C=0.06)
clf2=RandomForestClassifier(n_estimators=500)
clf3=GradientBoostingClassifier(n_estimators=100,learning_rate=0.12,max_depth=4)
clf4=SVC(C=4,gamma=0.015,probability=True)
clf5=KNeighborsClassifier(n_neighbors=8)


# In[ ]:


eclf_hard=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('GDBT',clf3),('SVM',clf4),('KNN',clf5)])


# In[ ]:


# add weights
eclfW_hard=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('GDBT',clf3),('SVM',clf4),('KNN',clf5)],weights=[1,1,2,2,1])


# In[ ]:


# soft voting
eclf_soft=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('GDBT',clf3),('SVM',clf4),('KNN',clf5)],voting='soft')


# In[ ]:


# add weights
eclfW_soft=VotingClassifier(estimators=[('LR',clf1),('RF',clf2),('GDBT',clf3),('SVM',clf4),('KNN',clf5)],voting='soft',weights=[1,1,2,2,1])


# __Finally we can evaluate all the models we just used.__

# In[ ]:


models=[KNeighborsClassifier(n_neighbors=8),LogisticRegression(C=0.06),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=500),
        GradientBoostingClassifier(n_estimators=100,learning_rate=0.12,max_depth=4),SVC(C=4,gamma=0.015),
        eclf_hard,eclf_soft,eclfW_hard,eclfW_soft,bagging]


# In[ ]:


names=['KNN','LR','NB','CART','RF','GBT','SVM','VC_hard','VC_soft','VCW_hard','VCW_soft','Bagging']
for name,model in zip(names,models):
    score=cross_val_score(model,X_scaled,y,cv=5)
    print("{}: {},{}".format(name,score.mean(),score))


# ## Stacking

# __We use logistic regression, k-nearest neighbors, support vector machine, Gradient Boosting Decision Tree as first-level models, and use random forest as second-level model.__

# In[ ]:


from sklearn.model_selection import StratifiedKFold
n_train=train.shape[0]
n_test=test.shape[0]
kf=StratifiedKFold(n_splits=5,random_state=1,shuffle=True)  


# In[ ]:


def get_oof(clf,X,y,test_X):
    oof_train=np.zeros((n_train,))
    oof_test_mean=np.zeros((n_test,))
    oof_test_single=np.empty((5,n_test))
    for i, (train_index,val_index) in enumerate(kf.split(X,y)):
        kf_X_train=X[train_index]
        kf_y_train=y[train_index]
        kf_X_val=X[val_index]
        
        clf.fit(kf_X_train,kf_y_train)
        
        oof_train[val_index]=clf.predict(kf_X_val)
        oof_test_single[i,:]=clf.predict(test_X)
    oof_test_mean=oof_test_single.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)


# In[ ]:


LR_train,LR_test=get_oof(LogisticRegression(C=0.06),X_scaled,y,test_X_scaled)
KNN_train,KNN_test=get_oof(KNeighborsClassifier(n_neighbors=8),X_scaled,y,test_X_scaled)
SVM_train,SVM_test=get_oof(SVC(C=4,gamma=0.015),X_scaled,y,test_X_scaled)
GBDT_train,GBDT_test=get_oof(GradientBoostingClassifier(n_estimators=100,learning_rate=0.12,max_depth=4),X_scaled,y,test_X_scaled)


# In[ ]:


X_stack=np.concatenate((LR_train,KNN_train,SVM_train,GBDT_train),axis=1)
y_stack=y
X_test_stack=np.concatenate((LR_test,KNN_test,SVM_test,GBDT_test),axis=1)


# In[ ]:


X_stack.shape,y_stack.shape,X_test_stack.shape


# In[ ]:


stack_score=cross_val_score(RandomForestClassifier(n_estimators=1000),X_stack,y_stack,cv=5)


# In[ ]:


# cross-validation score of stacking
stack_score.mean(),stack_score


# In[ ]:


pred=RandomForestClassifier(n_estimators=500).fit(X_stack,y_stack).predict(X_test_stack)


# In[ ]:


tt=pd.DataFrame({'PassengerId':test.PassengerId,'Survived':pred})


# In[ ]:


tt.to_csv('G.csv',index=False)


# In[ ]:




