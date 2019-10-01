#!/usr/bin/env python
# coding: utf-8

# # Import library

# In[1]:


#Import library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
seed =45
get_ipython().magic(u'matplotlib inline')
plt.style.use('fivethirtyeight')


# # Import data set

# In[2]:


path = '../input/'
#path = ''
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
print('Number rows and columns:',train.shape)
print('Number rows and columns:',test.shape)


# # Explore data set

# In[3]:


train.head(5)


# # Dependent varaiable

# In[4]:


plt.figure(figsize=(12,6))
sns.countplot(train['Survived'],palette='Blues')
plt.title('Dependent variable distribution plot')
plt.xlabel('Survived')

train['Survived'].value_counts()


# The 'target' variable is balanced 

# # CORELATION PLOT

# In[5]:


cor = train.corr()
plt.figure(figsize=(12,6))
sns.heatmap(cor,cmap='Set1',annot=True)


# # Missing value is data set

# In[6]:


k= pd.DataFrame()
k['train']= train.isnull().sum()
k['test'] = test.isnull().sum()
k.T


# Missing value in test train data set are in same propotion and same column

# In[7]:


def missing_value(df):
    col = df.columns
    for i in col:
        if df[i].isnull().sum()>0:
            df[i].fillna(df[i].mode()[0],inplace=True)


# In[8]:


missing_value(train)
missing_value(test)


# In[9]:


def uniq(df):
    col = df.columns
    k = pd.DataFrame(index=col)
    for i in col:
        k['No of Unique'] = df[i].nunique()
        k['first Unique values'] = df[i].unique()[0]
        k['sencond Unique values'] = df[i].unique()[1]
        return k.T
uniq(train)


# In[10]:


def category_type(df):
    col = df.columns
    for i in col:
        if df[i].nunique()<=7:
            df[i] = df[i].astype('category')
category_type(train)
category_type(test)


# # Univariate analysis

# In[11]:


train.select_dtypes(include=['category']).head()


# In[12]:


fig ,ax = plt.subplots(2,2,figsize=(16,16))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.countplot(data=train,x='Pclass',hue='Survived',palette='gist_rainbow',ax=ax1)
sns.countplot(data=train,x='Sex',hue='Survived',palette='viridis',ax=ax2)
sns.countplot(data=train,x='SibSp',hue='Survived',palette='viridis',ax=ax3)
sns.countplot(data=train,x='Parch',hue='Survived',palette='gist_rainbow',ax=ax4)


# In[13]:


plt.figure(figsize=(16,4))
sns.countplot(data=train,x='Embarked',hue='Survived',palette='gist_rainbow')


# In[14]:


train[['Age','Fare']].describe()


# In[15]:


fig,ax = plt.subplots(2,2,figsize=(16,16))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.distplot(train['Age'],bins=20,color='r',ax=ax1)
sns.boxplot(y='Age',x='Survived',data=train,ax=ax2)
sns.pointplot(y='Age',x='Survived',data=train,ax=ax3)
sns.violinplot(y='Age',x='Survived',data=train,ax=ax4)


# In[16]:


fig,ax = plt.subplots(2,2,figsize = (16,16))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.distplot(train['Fare'],bins=50,color='r',ax=ax1)
sns.boxplot(y='Fare',x='Survived',data=train,ax=ax2)
sns.pointplot(y='Fare',x='Survived',data=train,ax=ax3)
sns.violinplot(y='Fare',x='Survived',data=train,ax=ax4)


# # Extract features from Name

# In[17]:


corpus = [w.split() for w in train['Name']]
corpus[0:20]


# In[18]:


def Name_extract(df):
    k = []
    corpus = [w.split() for w in df['Name']]
    
    for i in corpus:
        if 'Mr.' in i:
            k.append('Mr.')
        elif 'Mrs.' in i:
            k.append('Mrs')
        elif 'Miss.' in i:
            k.append('Miss.')
        elif 'Master.' in i:
            k.append('Master.')
        elif 'Dr.' in i:
            k.append('Dr.')
        elif 'Capt.' in i:
            k.append('Capt.')
        elif 'Don.' in i:
            k.append('Don.')
        elif 'Col.' in i:
            k.append('Col.')
        elif 'Major.' in i:
            k.append('Major.')
        else:
            k.append('other')
    
    no_word = [len(l.split()) for l in df['Name']]
    no_char = [len(m) for m in df['Name']]
    df['name_category'],df['no_word'],df['no_char'] = k, no_word,no_char
    df['name_category'] = df['name_category'].astype('category')
    df['no_word'] = df['no_word'].astype('category')
    return df


# In[19]:


train = Name_extract(train)
test = Name_extract(test)


# In[20]:


train['name_category'].value_counts()


# Look thier is 1 captain, 1 Don , 2 Col , 2 Major ,7 Doctor in data set.
# We have categories for all the name.

# In[21]:


fig,ax = plt.subplots(2,1,figsize=(16,8))
ax1,ax2 = ax.flatten()
sns.countplot(data=train,x='name_category',hue='Survived',ax=ax1,palette='gist_rainbow')
sns.countplot(data=train,x='no_word',hue='Survived',palette='viridis',ax=ax2)


# The captain of the ship did not survived

# In[22]:


fig,ax = plt.subplots(2,2,figsize=(16,16))
ax1,ax2,ax3,ax4 = ax.flatten()
sns.distplot(train['no_char'],bins=50,color='r',ax=ax1)
sns.boxplot(data=train,y='no_char',x='Survived',ax=ax2)
sns.pointplot(data=train,y='no_char',x='Survived',ax=ax3)
sns.violinplot(data=train,y='no_char',x='Survived',ax=ax4)


# # Extract feature in Cabin variable

# In[23]:


def extract_cabin(df):  
    no_cabin = [len(w.split()) for w in df['Cabin']]
    df['no_cabin'] = no_cabin
    df['no_cabin'] = df['no_cabin'].astype('category')

extract_cabin(train)
extract_cabin(test)


# In[24]:


plt.figure(figsize=(16,8))
sns.countplot(data=train,x='no_cabin',hue='Survived',palette='gist_rainbow')


# # Co relation plot

# In[25]:


cor = train.corr()
plt.figure(figsize=(10,4))
sns.heatmap(cor,annot=True)
plt.tight_layout()


# # Determine outliers in dataset

# In[26]:


def outlier(df,columns):
    for i in columns:
        quartile_1,quartile_3 = np.percentile(df[i],[25,75])
        quartile_f,quartile_l = np.percentile(df[i],[1,99])
        IQR = quartile_3-quartile_1
        lower_bound = quartile_1 - (1.5*IQR)
        upper_bound = quartile_3 + (1.5*IQR)
        print(i,lower_bound,upper_bound,quartile_f,quartile_l)
                
        df[i].loc[df[i] < lower_bound] = quartile_f
        df[i].loc[df[i] > upper_bound] = quartile_l
num_col = ['Fare','Age','no_char']       
outlier(train,num_col)
outlier(test,num_col)


# # One Hot Encoding

# In[27]:


def OHE(df1,df2):
    #cat_col = column

    len_df1 = df1.shape[0]
    
    df = pd.concat([df1,df2],ignore_index=True)
    cat_col = df1.select_dtypes(include =['category']).columns
    c2,c3 = [],{}
    
    print('Categorical feature',len(cat_col))
    for c in cat_col:
        if df[c].nunique()>2 :
            c2.append(c)
            c3[c] = 'ohe_'+c
    
    df = pd.get_dummies(df, prefix=c3, columns=c2,drop_first=True)

    df1 = df.loc[:len_df1-1]
    df2 = df.loc[len_df1:]
    print('Train',df1.shape)
    print('Test',df2.shape)
    return df1,df2


# In[28]:


train1,test1 = OHE(train,test)


# # Encoder 

# In[29]:


le = LabelEncoder()
#col =['Sex']
train1['Sex'] = le.fit_transform(train1['Sex'])
test1['Sex'] = le.fit_transform(test1['Sex'])


# In[30]:


train1.head().T


# # Split data set

# In[31]:


train1.columns


# In[32]:


unwanted = ['PassengerId','Survived','Name','Cabin','Ticket']
X = train1.drop(unwanted,axis=1)
y = train1['Survived'].astype('category')
x_test = test1.drop(unwanted,axis=1)
#del train1,test1


# # Hyperparameter tuning

# In[33]:


#Grid Search
logreg = LogisticRegression(class_weight='balanced')
param = {'C':[0.001,0.003,0.005,0.01,0.03,0.05,0.1,0.3,0.5,1,2,3,3,4,5,10,20]}
clf = GridSearchCV(logreg,param,scoring='roc_auc',refit=True,cv=10)
clf.fit(X,y)
print('Best roc_auc: {:.4}, with best C: {}'.format(clf.best_score_, clf.best_params_))


# # Logistic Regression model
# Logistic regression is used for modelling. The data set is split using Stratified Kfold. In each split model is created and predicted using that model. The final predicted value is average of all model. 

# In[34]:


kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
pred_test_full =0
cv_score =[]
i=1
for train_index,test_index in kf.split(X,y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = X.loc[train_index],X.loc[test_index]
    ytr,yvl = y.loc[train_index],y.loc[test_index]
    
    #model
    lr = LogisticRegression(C=2)
    lr.fit(xtr,ytr)
    score = roc_auc_score(yvl,lr.predict(xvl))
    print('ROC AUC score:',score)
    cv_score.append(score)    
    pred_test = lr.predict_proba(x_test)[:,1]
    pred_test_full +=pred_test
    i+=1


# In[35]:


print('Confusion matrix\n',confusion_matrix(yvl,lr.predict(xvl)))
print('Cv',cv_score,'\nMean cv Score',np.mean(cv_score))


# In[36]:


lr.coef_


# In[37]:


lr.score(xvl,yvl)


# # Reciever Operating Characteristics

# In[38]:


proba = lr.predict_proba(xvl)[:,1]
frp,trp, threshold = roc_curve(yvl,proba)
roc_auc_ = auc(frp,trp)

plt.figure(figsize=(14,8))
plt.title('Reciever Operating Characteristics')
plt.plot(frp,trp,'r',label = 'AUC = %0.2f' % roc_auc_)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')


# # Predict for unsen data set

# In[39]:


y_pred = pred_test_full/5
submit = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':y_pred})
submit['Survived'] = submit['Survived'].apply(lambda x: 1 if x>0.5 else 0)
#submit.to_csv('lr_titanic.csv.gz',index=False,compression='gzip') 
submit.to_csv('lr_titanic.csv',index=False) 


# In[40]:


submit.head()


# # Thank you for visiting
