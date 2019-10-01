#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('../input/train.csv')
train.head()



# In[2]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[3]:


Passenger = train.iloc[:,0]


# In[4]:


Passenger.head()


# In[5]:


combined = [train,test]


# In[6]:


train.describe()


# In[7]:


train.info()


# # age, cabin and embarked have missing values

# In[8]:


test.dtypes


# name,sex,cabin,embarked are categorical columns while rest are numerical

# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


train.groupby(['Cabin'])['Cabin'].count().sort_values(ascending = False).index[0]


# In[12]:


def impute_cat(df_train,df_test,variable):
    most_frequent_category = train.groupby([variable])[variable].count().sort_values(ascending = False).index[0]
    df_train[variable].fillna(most_frequent_category, inplace = True)
    df_test[variable].fillna(most_frequent_category, inplace = True)
    


# In[13]:


def impute_cat_missing(df_train,df_test,variable):
    df_train[variable].fillna('Missing', inplace = True)
    df_test[variable].fillna('Missing', inplace = True)
    
impute_cat_missing(train,test,'Cabin')


# In[14]:


impute_cat(train,test,'Embarked')


# In[15]:


def impute_num(train,test, variable):
    train[variable].fillna(train[variable].median(), inplace = True)
    test[variable].fillna(test[variable].median(), inplace = True)


# In[16]:


for variable in ['Age', 'Fare']:
    impute_num(train,test,variable)


# In[17]:


test.isnull().sum()


# In[18]:


submission = pd.read_csv('../input/gender_submission.csv')


# In[19]:


submission.head()


# In[20]:


submission.isnull().sum()


# let's determine outliers

# In[21]:


train.dtypes != 'O'


# In[22]:


train.columns


# In[ ]:





# In[23]:


print('number of unique values for Age : ' , train.Age.nunique())
print('number of unique values for Fare : ' , train.Fare.nunique())


# In[24]:


train.dtypes


# In[25]:


train['Ticket'].head()


# In[26]:


# lets convert SibSp and Parch into a single column
train.head()


# In[27]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test.SibSp + test.Parch +1


# In[28]:


train['IsAlone'] = np.where(train.FamilySize >1, 0, 1)


# In[29]:


train.head()


# In[30]:


test['IsAlone'] = np.where(test.FamilySize >1, 0, 1)


# In[31]:


train['Title'] = train.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0]


# In[32]:


test['Title'] = test.Name.str.split(',', expand=True)[1].str.split('.', expand=True)[0]


# In[33]:


train.Ticket.nunique()


# In[34]:


train.info()


# In[35]:


train['Ticket_Num'] = train.Ticket.apply( lambda s : s.split(' ')[-1])


# In[36]:


train['Ticket_Num'] = np.where(train.Ticket_Num.str.isdigit(), train.Ticket_Num, np.nan)
train['Ticket_Cat'] = train.Ticket.apply(lambda s: s.split(' ')[0])
train['Ticket_Cat'] = np.where(train.Ticket_Cat.str.isdigit(),  np.nan, train['Ticket_Cat'])


# In[37]:


train.head()


# In[38]:


train.Ticket_Num.isnull().sum()


# In[39]:


train.Ticket_Cat.isnull().sum()


# In[40]:


import re


# In[41]:


text = train.Ticket_Cat.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))


# In[42]:


text = text.str.upper()


# In[43]:


text.unique()


# In[44]:


train.Ticket_Cat = train.Ticket_Cat.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
train.Ticket_Cat = train.Ticket_Cat.str.upper()


# In[45]:


test['Ticket_Num'] = test.Ticket.apply( lambda s : s.split(' ')[-1])
test['Ticket_Num'] = np.where(test.Ticket_Num.str.isdigit(), test.Ticket_Num, np.nan)
test['Ticket_Cat'] = test.Ticket.apply(lambda s: s.split(' ')[0])
test['Ticket_Cat'] = np.where(test.Ticket_Cat.str.isdigit(),  np.nan, test['Ticket_Cat'])


# In[46]:


test.Ticket_Cat = test.Ticket_Cat.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
test.Ticket_Cat = test.Ticket_Cat.str.upper()


# In[47]:


train['Ticket_Cat'] = np.where(train['Ticket_Cat'] == 'NAN', np.nan, train['Ticket_Cat'])
test['Ticket_Cat'] = np.where(test['Ticket_Cat'] == 'NAN', np.nan, test['Ticket_Cat'])


# In[48]:


train.Ticket_Cat.isnull().sum()


# In[49]:


train['Cabin_Cat'] = train['Cabin'].str[0]


# In[50]:


train['Cabin_Num'] = train['Cabin'].str.extract('(\d+)')


# In[51]:


test['Cabin_Cat'] = test['Cabin'].str[0]
test['Cabin_Num'] = test['Cabin'].str.extract('(\d+)')


# In[52]:


train.head()


# In[53]:


test.head()


# In[54]:


train['Cabin_Num'] = train['Cabin_Num'].astype('float')


# In[55]:


def conv_to_float(data):
    for var in ['Ticket_Num',  'Cabin_Num']:
        data[var] = data[var].astype('float')


# In[56]:


conv_to_float(train)
conv_to_float(test)


# In[57]:


train['FareBin'] = pd.qcut(train.Fare,  5)


# In[58]:


test['FareBin'] = pd.qcut(test.Fare, 5)


# In[59]:


test['AgeBin']= pd.cut(test.Age.astype(int), 5)
train['AgeBin']= pd.cut(train.Age.astype(int), 5)


# In[60]:


train.head()


# In[61]:


train.drop(labels= ['PassengerId', 'Name', 'Cabin','Ticket'], inplace= True, axis =1)


# In[62]:


train.head()


# In[63]:


test.drop(labels= ['PassengerId', 'Name', 'Cabin','Ticket'], inplace= True, axis =1)


# In[64]:


def impute_cat_missing(df_train,df_test,variable):
    df_train[variable].fillna('Missing', inplace = True)
    df_test[variable].fillna('Missing', inplace = True)
  


# In[65]:


impute_num(train,test,'Cabin_Num')


# In[66]:


impute_num(train,test,'Ticket_Num')


# In[67]:


impute_cat_missing(train,test,'Cabin_Cat')


# In[68]:


impute_cat(train,test,'Ticket_Cat')


# In[69]:


#finding outlier in numerical columns
 
for var in ['Age', 'Fare', 'FamilySize']:
    sns.boxplot(y = var, data = train)
    plt.show()


# In[70]:


# all of these have outliers
# let's find out their distribution
for var in ['Age', 'Fare', 'FamilySize']:
    sns.distplot(train[var], bins = 30)
    plt.show()


# In[71]:


train.columns


# In[72]:


# all of these do not follow gaussian distribution , hence , we will follow inter-quartile range to find out outliers
def removing_outliers(data):
    for var in ['Age', 'Fare']:
        IQR = data[var].quantile(.75) - data[var].quantile(.25)
        upper_bound = round(data[var].quantile(.75) + (IQR*3))
        lower_bound = round(data[var].quantile(.25) - (IQR*3))
        print('Extreme outliers are values for {variable} < {lowerboundary} or > {upperboundary}'.format(variable=var, lowerboundary=lower_bound, upperboundary=upper_bound))
        print('-**************************-')
        print('Removing outlier values')
        data[var] = np.where(data[var] > upper_bound, upper_bound, data[var])


# In[73]:


removing_outliers(train)


# In[74]:


# let's check whether top-coding worked
for var in ['Age',  'Fare']:
    print(var, ' max value: ', train[var].max())


# In[75]:


# removing outliers in categorical columns
for var in ['Title', 'Ticket_Cat', 'Embarked', 'Cabin_Cat']:
    print(var, train[var].value_counts()/np.float(len(train)))
    print()


# In[76]:


train.dtypes


# In[77]:


temp = train.groupby(['Cabin_Cat'])['Cabin_Cat'].count()/np.float(len(train))


# In[78]:


frequent_cat = [x for x in temp.loc[temp>0.01].index.values]


# In[79]:


test.dtypes
   


# In[80]:


frequent_cat


# In[81]:


def rare_imputation(variable, which='rare'): 
    # find frequent labels
    temp = train.groupby([variable])[variable].count()/np.float(len(train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    # create new variables, with Rare labels imputed
    if which=='frequent':
        # find the most frequent category
        mode_label = train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        train[variable] = np.where(train[variable].isin(frequent_cat), train[variable], mode_label)
        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], mode_label)
            
    else:
        train[variable] = np.where(train[variable].isin(frequent_cat), train[variable], 'Rare')
        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], 'Rare')
       


# In[82]:


rare_imputation('Cabin_Cat', 'frequent')
rare_imputation('Ticket_Cat', 'rare')
rare_imputation('Title', 'frequent')


# In[83]:


for var in ['Title', 'Ticket_Cat', 'Embarked', 'Cabin_Cat']:
    print(var, train[var].value_counts()/np.float(len(train)))
    print()
    print(train.dtypes)



# In[84]:


print(train.head())
print(train.dtypes)


# In[85]:


def conv_to_cat(data):
    for var in ['Sex', 'Embarked', 'Title', 'Ticket_Cat', 'Cabin_Cat', 'FareBin', 'AgeBin']:
        data[var] = data[var].astype('category')


# In[86]:


conv_to_cat(train)


# In[87]:


conv_to_cat(test)


# In[88]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()


# In[89]:


def label_encode(data):
    for var in ['Sex', 'Embarked', 'Title', 'Ticket_Cat', 'Cabin_Cat', 'FareBin', 'AgeBin']:
        data[var] = label.fit_transform(data[var])
           


# In[90]:


train_label = train.copy()


# In[91]:


test_label = test.copy()


# In[92]:


label_encode(train_label)


# In[93]:


train.head()


# In[94]:


label_encode(test_label)


# In[95]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[2,7,10,12,13,15,16])


# In[96]:


train_ohe= train_label.copy()
test_ohe = test_label.copy()


# In[97]:


train_ohe = ohe.fit_transform(train_ohe).toarray()


# In[98]:



ohe_test = OneHotEncoder(categorical_features=[1,6,9,11,12,14,15])


# In[99]:


test_ohe = ohe_test.fit_transform(test_ohe).toarray()


# In[100]:


train_ohe


# In[101]:


test_ohe


# In[102]:


train_label.head()


# In[103]:


test_label.head()


# In[104]:


train.head()


# In[105]:


train.groupby(['Survived'], as_index=False).mean()


# In[106]:


train_label.groupby(['Survived'], as_index=False).mean()


# In[107]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[108]:


train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[109]:


train[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[110]:


train[['FareBin', 'Survived']].groupby(['FareBin'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[111]:


pd.crosstab(train['Title'], train['Survived'])


# In[112]:


train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[113]:


train[['Ticket_Cat', 'Survived']].groupby(['Ticket_Cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[114]:


train[['Cabin_Cat', 'Survived']].groupby(['Cabin_Cat'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[115]:


train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[116]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).sum().sort_values(by='Survived', ascending=False)


# In[117]:


train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[118]:


train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[119]:


train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[120]:


train.columns


# In[121]:


for var in ['Pclass', 'Sex', 'SibSp', 'Parch','Embarked', 'FamilySize', 'IsAlone', 'Title','Ticket_Cat', 'Cabin_Cat', 'FareBin', 'AgeBin']:
    print(train[[var, 'Survived']].groupby([var], as_index=False).sum().sort_values(by='Survived', ascending=False))
    print('*'*40)


# In[122]:


plt.figure(figsize=(15,9))
i=1
colour = {1 : 'magenta', 2 : 'green', 3: 'orange'}
for var in ['Age', 'Fare', 'FamilySize']:
    plt.subplot(1,3,i)
    sns.boxplot(y=var, data=train, color= colour[i])
    plt.title(var)
    i=i+1
    


# In[123]:


plt.figure(figsize=(15,6))
i=1
colour = {1 : 'magenta', 2 : 'green', 3: 'orange'}
for var in ['Age', 'Fare', 'FamilySize']:
    plt.subplot(1,3,i)
    sns.boxplot(x='Sex', y=var, data=train, color= colour[i], hue='Survived')
    plt.title(var)
    i=i+1


# In[124]:


plt.figure(figsize=(15,6))
i=1
j=1
colour = {1 : 'pink', 2 : 'green', 3: 'white', 4: 'black', 5:'orange', 6:'blue'}
for var in ['Age', 'Fare', 'FamilySize']:
    plt.subplot(1,3,i)
    plt.hist(x=[train[train['Survived']==1][var],train[train['Survived']==0][var]], bins =20, stacked = True, color= [colour[j], colour[j+1]], label= ['Survived', 'Dead'])
    plt.title(var)
    plt.legend()
    i=i+1
    j=j+2


# In[125]:


train.columns


# In[126]:


plt.figure(figsize=(19,25))
i=1
for var in ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin', 'AgeBin']:
    plt.subplot(5,2,i)
    sns.barplot(x=var, y='Survived', data=train)
    plt.title(var)
    i=i+1
      


# In[127]:


plt.figure(figsize=(19,25))
i=1
for var in ['Pclass',  'SibSp', 'Parch', 'Embarked', 'FamilySize', 'IsAlone', 'Title', 'FareBin', 'AgeBin']:
    plt.subplot(5,2,i)
    sns.barplot(x=var, y='Survived', data=train, hue ='Sex')
    plt.title(var)
    i=i+1


# In[128]:


g = sns.FacetGrid(data= train, col='Survived', row='Pclass')
g.map(plt.hist, 'Age')
g.map(plt.legend)


# In[129]:


train.columns


# In[130]:


f = sns.FacetGrid(data=train, col = 'Embarked', row= 'Survived')
f.map(sns.barplot, 'Sex', 'Fare')


# In[131]:


plt.scatter(x = train.Age, y=train.Fare)


# In[132]:


sns.regplot(x='Age', y='Fare', data=train)


# In[133]:


sns.kdeplot(train['Age'], train['Fare'])


# In[134]:


train.columns


# In[135]:


h = sns.FacetGrid(data=train, row='Embarked', col='Pclass')
h.map(sns.barplot, 'Survived', 'Age')


# In[136]:


sns.pairplot(data=train)


# In[137]:


sns.lmplot('Age', 'Fare', data=train)


# In[138]:


train.head()


# In[139]:


train.corr() >.8


# In[140]:


train_label.corr() >.8


# In[141]:


# feature Selection
# lets find out the constant features


# In[142]:


from sklearn.feature_selection import VarianceThreshold


# In[143]:


sel = VarianceThreshold(threshold=0)


# In[144]:


sel.fit(train_label)


# In[145]:


train.head()


# In[146]:


train_label.head()


# In[147]:


sum(sel.get_support())


# In[148]:


train_label.shape


# In[149]:


# let's find the quasi constant features
sel_quasi = VarianceThreshold(threshold=0.01) # for 99% constant features


# In[150]:


sel_quasi.fit(train_label)


# In[151]:


sum(sel_quasi.get_support())


# In[152]:


# hence, none of the columns are constant and quasi constant .
# let's find out the duplicated columns


# In[153]:


dup_train_label = train_label.T


# In[154]:


dup_train_label.head()


# In[155]:


dup_train_label.duplicated().sum()


# In[156]:


# so, this shows that none of the columns are duplicated


# In[157]:


train_label.corr()


# In[158]:


plt.figure(figsize=(15,8))
sns.heatmap(data = train_label.corr().abs(), cmap = 'magma_r')


# In[159]:


def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[160]:


correlation(train_label, 0.8)


# In[161]:


len(set(correlation(train_label, 0.8)))


# In[162]:


train_label.corr()


# In[163]:


# {'AgeBin', 'FamilySize', 'FareBin'} are correlated features


# In[164]:


train_label.drop(labels = ['AgeBin', 'FamilySize', 'FareBin'], axis=1, inplace=True)


# In[165]:


train_label.head()


# In[166]:


test_label.head()


# In[167]:


test_label.drop(labels=['AgeBin', 'FamilySize', 'FareBin'], axis=1, inplace=True)


# In[168]:


# using Univariate roc-auc or mse for feature selection
roc_values = []
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
X_train, X_test,y_train,y_test = train_test_split(train_label.drop(labels=['Survived'], axis=1), train_label['Survived'], test_size=0.25, random_state=1)


# In[169]:


for var in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train.loc[:,var].to_frame(), y_train)
    y_score = clf.predict(X_test.loc[:,var].to_frame())
    roc_values.append(roc_auc_score(y_test,y_score))


# In[170]:


roc_values


# In[171]:


roc_values = pd.Series(roc_values)


# In[172]:


roc_values.index = X_train.columns


# In[173]:


roc_values


# In[174]:


roc_values[roc_values <= 0.5]


# In[175]:


train_label.drop(labels = ['Ticket_Cat'], axis=1, inplace=True)


# In[176]:


train_label.head()


# In[177]:


test_label.drop(labels=['Ticket_Cat'], axis=1, inplace=True)


# In[178]:


from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score


# In[179]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# In[180]:


X_train,X_test,y_train,y_test = train_test_split(train_label.drop(labels=['Survived'], axis=1), train_label['Survived'], test_size=0.3, random_state=1)


# In[181]:


from sklearn.ensemble import RandomForestClassifier


# In[182]:


sfs = SFS(estimator=RandomForestClassifier(n_jobs=-1), k_features=12,forward=True, verbose=2, scoring='roc_auc', cv=10)


# In[183]:


sfs = sfs.fit(np.array(X_train),y_train)


# In[184]:


sfs_2 = SFS(estimator=RandomForestClassifier(n_jobs=-1, n_estimators=30), k_features=10,forward=True, verbose=2, scoring='roc_auc', cv=10)


# In[185]:


sfs_2 = sfs_2.fit(np.array(X_train), y_train)


# In[186]:


sfs_2.subsets_


# In[187]:


selected_features = X_train.columns[list(sfs_2.k_feature_idx_)]


# In[188]:


selected_features


# In[189]:


X_train.columns


# In[190]:


sfs_3 = SFS(estimator=RandomForestClassifier(n_jobs=-1, n_estimators=30), k_features=10, forward=False, verbose=2, floating=False, scoring='roc_auc', cv=10)


# In[191]:


sfs_3.fit(np.array(X_train), y_train)


# In[192]:


selected_features_backward = X_train.columns[list(sfs_3.k_feature_idx_)]


# In[193]:


selected_features_backward


# * # according to these forward and backward selection techniques, Fare is not an important feature
# * # but we have seen in visualizations that Fare is important and hence, we include it for further analysis

# In[194]:


# let's make models on these and evaluate their performance


# In[195]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


# In[196]:


logreg= LogisticRegression()
logreg.fit(X_train, y_train)


# In[197]:


y_pred = logreg.predict(X_test)


# In[198]:


logreg.score(X_train,y_train)


# In[199]:


from sklearn.metrics import confusion_matrix, roc_auc_score
roc_auc_score(y_test,y_pred)


# In[200]:


train_label.head()


# In[201]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# In[202]:


logreg = LogisticRegression()


# In[203]:


logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import roc_auc_score
logreg_score = logreg.score(X_train,y_train)


# In[204]:


logreg_score


# In[205]:


roc_auc_score(y_test,y_pred)


# In[206]:


# let's find out the most important features using regression coefficients


# In[207]:


logreg.coef_


# In[232]:


score_dict = {}
roc_auc_score_dict = {}


# In[233]:


logreg = LogisticRegression()
svc = SVC()
linear_svc = LinearSVC()
gaussian_nb = GaussianNB()
decision_tree = DecisionTreeClassifier()
random_forest = RandomForestClassifier(n_estimators=100)
xg_boost = XGBClassifier()


# In[234]:


alg = [LogisticRegression(),
SVC(),
LinearSVC(),
GaussianNB(),
DecisionTreeClassifier(),
RandomForestClassifier(n_estimators=100),
XGBClassifier()]


# In[235]:



for est in alg:
    est_name = est.__class__.__name__
    print(est_name)
    model = est
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score_dict[est_name] = model.score(X_train,y_train)
    roc_auc_score_dict[est_name] = roc_auc_score(y_test,y_pred)

    


# In[236]:


score_dict


# In[237]:


roc_auc_score_dict


# In[255]:


score_df = pd.DataFrame.from_dict(data=score_dict, orient='index' )


# In[258]:


score_df.columns = ['Model Score']


# In[259]:


score_df


# In[260]:


roc_auc_score_df = pd.DataFrame.from_dict(data=roc_auc_score_dict, orient='index')


# In[261]:


roc_auc_score_df.columns = ['ROC-AUC Score']


# In[262]:


roc_auc_score_df


# In[266]:


combined_score_df = pd.concat([score_df, roc_auc_score_df], axis=1)


# In[267]:


combined_score_df


# In[272]:


test_label.head()


# In[268]:


# this states that XBG classifier performs best, though scores for Decision Tree and Random Forest are better, they are clearly overfitting


# In[273]:


from sklearn.preprocessing import StandardScaler
sc_test = StandardScaler()
scaled_test = sc.fit_transform(test_label)


# In[274]:


xgb = XGBClassifier()


# In[275]:


xgb.fit(X_train,y_train)


# In[276]:


y_pred = xgb.predict(scaled_test)


# In[278]:


submission = pd.DataFrame({"PassengerId": submission["PassengerId"],"Survived": y_pred})


# In[280]:


submission.head()


# In[284]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




