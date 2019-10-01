#!/usr/bin/env python
# coding: utf-8

# ### Titanic - Data Exploration and Feature Engineering  For Beginner

# #### Importing libraries

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'import numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns')


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"df_train=pd.read_csv('../input/train.csv')")


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"df_test=pd.read_csv('../input/test.csv')")


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train.head(3)


# In[ ]:


df_train.dtypes


# In[ ]:


df_train['Survived'].value_counts()


# Bar chart for categorical features
# *Pclass
# *Sex
# *SibSp
# *Parch
# *Embarked
# 

# In[ ]:


def bar_chart(feature):
    survived = df_train[df_train['Survived']==1][feature].value_counts()
    dead = df_train[df_train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index=['Survived','Dead'] # x-axis
    df.plot(kind='bar', stacked=True, figsize=(10,7))


# In[ ]:


bar_chart('Pclass')


# The chart confirms 1st class mostly survived and 3rd class mostly dead

# In[ ]:


bar_chart('Sex')


# The chart confirms women more likely survived than Men

# In[ ]:


bar_chart('Embarked')


# the chart confirms person from C are more likely survived and person from S are mostly dead and from Q also mostly dead 

# In[ ]:


bar_chart('SibSp')


# *The chart confirms a person aboarded with more than 2 siblings r spouse more likely survived
# *The chart a person aboarded without siblings r spouse more likely dead 

# In[ ]:


bar_chart('Parch')


# The chart confirms a person aboarded with more than 2 parents r children more likely survived and person aboarded without parents r child more likely dead

# In[ ]:


bar_chart('Embarked')


# ### Feature importance- used to detect the priority of the features

# In[ ]:


df=pd.read_csv('../input/train.csv')


# In[ ]:


df.dtypes


# In[ ]:


df.sample(2)


# In[ ]:


df['PassengerId']=df['PassengerId'].astype('category')
df['Survived']=df['Survived'].astype('category')
df['Pclass']=df['Pclass'].astype('category')
df['Name']=df['Name'].astype('category')
df['Sex']=df['Sex'].astype('category')
df['Age']=df['Age'].astype('category')
df['SibSp']=df['SibSp'].astype('category')
df['Parch']=df['Parch'].astype('category')
df['Ticket']=df['Ticket'].astype('category')
df['Cabin']=df['Cabin'].astype('category')
df['Fare']=df['Fare'].astype('category')
df['Embarked']=df['Embarked'].astype('category')


# In[ ]:


df.dtypes


# In[ ]:


cat_columns=df.select_dtypes(['category']).columns
print(cat_columns)
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


# In[ ]:


df.dtypes


# In[ ]:


X=df.iloc[:,2:12]
Y=df.iloc[:,1:2]


# In[ ]:


df.head(2)


# In[ ]:


X.head(2)


# In[ ]:


Y.head(2)


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'from sklearn.ensemble import RandomForestClassifier\nrnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=50)\nrnd_clf.fit(X, Y)\nfor name, importance in zip(df.columns, rnd_clf.feature_importances_):\n     print(name, "=", importance)')


# #### graph for feature importance

# In[ ]:


features = df.columns
importances = rnd_clf.feature_importances_
indices = np.argsort(importances)


# In[ ]:


print('Features:',features)
print('importances:',importances)
print('indices:',indices)
print("range:",range(len(indices)))
print("imp[indi] :",importances[indices])


# In[ ]:


plt.title('Feature Importances of Titanic Dataset')
plt.barh(range(len(indices)), importances[indices],color='purple',align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])


# In[ ]:


plt.title('Feature Importances of Titanic Dataset')
plt.barh(range(len(indices)), importances[indices],color='purple',align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# #### Feature Engineering
# Name

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u"df_train_test=[df_train, df_test]#combining train and test dataset\nfor data in df_train_test:\n    data['Title'] = data['Name'].str.extract('([A-za-z]+)\\.', expand=False)")


# In[ ]:


df_train['Title'].value_counts()


# In[ ]:


df_test['Title'].value_counts()


# ### Title map
# 
mr: 0
miss: 1
mrs: 2
others: 3
# In[ ]:


title_mapping={"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Dr":3, "Rev":3, "Col":3, "Ms":3, "Dona":3, "Major":3, "Mme":3, "Don":3,
             "Sir":3, "Jonkheer":3, "Capt":3, "Lady":3, "Dona":3, "Mlle":3, "Countess":3 }


# In[ ]:


for data in df_train_test:
    data['Title']=data['Title'].map(title_mapping)
    


# In[ ]:


df_train.head()


# In[ ]:


bar_chart('Title')


# In[ ]:


df_test.head(4)


# In[ ]:


#delete unnecessary feature from dataset
df_train.drop('Name',axis=1, inplace=True)
df_test.drop('Name',axis=1, inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### sex
# 
male: 0 
female: 1
# In[ ]:


gender_mapping={"male":0, "female":1}
for data in df_train_test:
    data['Sex']=data['Sex'].map(gender_mapping)


# In[ ]:


bar_chart('Sex')
plt.xticks(rotation='horizontal')


# ### Age
# some fields in age is missing

# In[ ]:


df_train.head(10)


# In[ ]:


df_train.tail(10)


# In[ ]:


df_train["Age"].fillna(df_train.groupby("Title")["Age"].transform("median"), inplace=True)
df_test["Age"].fillna(df_test.groupby("Title")["Age"].transform("median"), inplace=True)


# In[ ]:


facet=sns.FacetGrid(df_train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, df_train["Age"].max()))
facet.add_legend()
plt.show()


# In[ ]:


facet=sns.FacetGrid(df_train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0, df_train["Age"].max()))
facet.add_legend()
plt.xlim(0,40)


# ### converting numerical age to categorical variable 
# 
child : 0
young : 1
adult : 2
mid-age : 3
senior : 4
# In[ ]:


for data in df_train_test:
    data.loc[data['Age'] <= 16, 'Age'] = 0,
    data.loc[(data['Age'] > 16) & (data['Age'] <= 26),'Age'] = 1,
    data.loc[(data['Age'] > 26) & (data['Age'] <= 36),'Age'] = 2,
    data.loc[(data['Age'] > 36) & (data['Age'] <= 62),'Age'] = 3,
    data.loc[data['Age'] > 62,'Age']= 4


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


bar_chart('Age')
plt.xticks(rotation='horizontal')


# ### Embarked

# In[ ]:


Pclass1 = df_train[df_train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = df_train[df_train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = df_train[df_train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,6))
plt.xticks(rotation='horizontal')


# In[ ]:


for data in df_train_test:
    data['Embarked'] = data['Embarked'].fillna('S')


# In[ ]:


df_train.head()


# In[ ]:


embarked_mapping = {"S" : 0, "C" : 1, "Q" :2}
for data in df_train_test:
    data['Embarked'] = data['Embarked'].map(embarked_mapping)


# ### Fare

# In[ ]:


#fill missing fare with median fare for each Pclass
df_train["Fare"].fillna(df_train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
df_test["Fare"].fillna(df_test.groupby("Pclass")["Fare"].transform("median"), inplace=True)


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Fare', shade=True)
facet.set(xlim=(0, df_train['Fare'].max()))
facet.add_legend()
plt.show()


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Fare', shade=True)
facet.set(xlim=(0, df_train['Fare'].max()))
facet.add_legend()
plt.xlim(0,100)


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Fare', shade=True)
facet.set(xlim=(0, df_train['Fare'].max()))
facet.add_legend()
plt.xlim(100,200)


# In[ ]:


for data in df_train_test:
    data.loc[data['Fare'] <= 17, 'Fare'] = 0,
    data.loc[(data['Fare'] > 17) & (data['Fare'] <= 30),'Fare'] = 1,
    data.loc[(data['Fare'] > 30) & (data['Fare'] <= 100),'Fare'] = 2,
    data.loc[data['Fare'] > 100,'Fare']= 3


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# ### Cabin

# In[ ]:


for data in df_train_test:
    data['Cabin'] = data['Cabin'].str[:1]


# In[ ]:


Pclass1 = df_train[df_train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = df_train[df_train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = df_train[df_train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class','3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,6))
plt.xticks(rotation='horizontal')


# In[ ]:


cabin_mapping = { "A":0, "B": 0.4, "C":0.8, "D": 1.2, "E":1.6, "F":2, "G":2.4, "T":2.8 }
for data in df_train_test:
    data['Cabin'] = data['Cabin'].map(cabin_mapping)


# In[ ]:


#fill missing fare with median fare for each pclass
df_train['Cabin'].fillna(df_train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
df_test['Cabin'].fillna(df_test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# ### Family Size

# In[ ]:


df_train['Family'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['Family'] = df_test['SibSp'] + df_test['Parch'] + 1


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Family', shade=True)
facet.set(xlim=(0, df_train['Family'].max()))
facet.add_legend()
plt.show()


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Family', shade=True)
facet.set(xlim=(0, df_train['Family'].max()))
facet.add_legend()
plt.show()


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Family', shade=True)
facet.set(xlim=(0, df_train['Family'].max()))
facet.add_legend()
plt.xlim(0,4)


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Family', shade=True)
facet.set(xlim=(0, df_train['Family'].max()))
facet.add_legend()
plt.xlim(4,7)


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Family', shade=True)
facet.set(xlim=(0, df_train['Family'].max()))
facet.add_legend()
plt.xlim(7,10)


# In[ ]:


facet = sns.FacetGrid(df_train, hue="Survived", aspect=4)
facet.map(sns.kdeplot,'Family', shade=True)
facet.set(xlim=(0, df_train['Family'].max()))
facet.add_legend()
plt.xlim(8,10)


# In[ ]:


df_train['Family'].value_counts()


# In[ ]:


family_mapping = {1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4}
for data in df_train_test:
    data['Family'] = data['Family'].map(family_mapping)


# In[ ]:


df_train.head()


# In[ ]:


df_test.head()


# In[ ]:


features_remove = ['Ticket', 'SibSp','Parch']
df_train = df_train.drop(features_remove, axis=1)
df_test = df_test.drop(features_remove, axis=1)
df_train = df_train.drop(['PassengerId'], axis=1)


# In[ ]:


print("Train Set","\n",df_train.head(),"\n")
print("Test Set","\n",df_test.head())


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_target = df_train['Survived'] 


# In[ ]:


df_target.head()


# In[ ]:


df_train = df_train.drop(['Survived'], axis=1)


# In[ ]:


df_train.shape,df_target.shape


# In[ ]:


df_train.columns


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# ## Modelling 

# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'# Importing Classifier Modules\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.svm import SVC')


# ### cross validation (K-fold)

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold= KFold(n_splits=10, shuffle=True, random_state=0)


# ### DecisionTree

# In[ ]:


clf = DecisionTreeClassifier()
clf


# In[ ]:


scoring = 'accuracy'
score = cross_val_score(clf, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ### RandomForest Classifier

# In[ ]:


rf=RandomForestClassifier(n_estimators=10)
rf


# In[ ]:


scoring = 'accuracy'
score = cross_val_score(rf, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# ## svc

# In[ ]:


svc=SVC()
svc


# In[ ]:


scoring = 'accuracy'
score = cross_val_score(svc, df_train, df_target, cv=k_fold, scoring = scoring)
print(score)


# In[ ]:


round(np.mean(score)*100, 2)


# #### Test data

# In[ ]:


df_test.shape


# In[ ]:


df_test.columns


# In[ ]:


get_ipython().run_cell_magic(u'time', u'', u'svc1=SVC()\nsvc1.fit(df_train, df_target)\ntest_data = df_test.drop("PassengerId", axis=1).copy()\nprediction = svc1.predict(test_data)')


# In[ ]:


submission = pd.DataFrame({
    "PassengerId" : df_test["PassengerId"],
    "Survived"  : prediction
})


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:


submission = pd.read_csv('submission.csv')
submission.head()


# In[ ]:





# In[ ]:





# In[ ]:




