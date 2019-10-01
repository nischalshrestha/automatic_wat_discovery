#!/usr/bin/env python
# coding: utf-8

# # About me and this notebook
# _About me_ : I am a person who has no degree in math or computer science. I work in data management area and learn Python by myself in last 2 years, but very small chance to use it in my work. I preferred to use Excel + VBA to complete all data related taks. I hope that keep practicing would make me become a data scientist someday.
# 
# _About notebook_ : This notebook is to conclude what I've learned from various source, I am so new to this area please feel free to comment & suggest. I will learn from you guys.
# 
# Thanks,<br>
# **Nopp**
# 
# -----
# **Credit/Reference**:
# - https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# - https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy
# - https://www.kaggle.com/ash316/eda-to-prediction-dietanic
# - https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling
# - https://www.kaggle.com/sshadylov/titanic-solution-using-random-forest-classifier

# # 0. Import some required dependencies

# In[ ]:


import sys
print('Python version: {}'.format(sys.version))

import pandas as pd
print('pandas version: {}'.format(pd.__version__))

import numpy as np

print('numpy version: {}'.format(np.__version__))

import matplotlib as mlp
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
print('matplotlib version: {}'.format(mlp.__version__))

import seaborn as sns
print('seaborn version: {}'.format(sns.__version__))

import os
print('\nFile list:',os.listdir('../input'))

import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore')


# # 1. What is this dataset all about
# The Titanic was the largest ship afloat at the time of her maiden voyage and carried 2,224 people on that maiden voyage from Southampton on 10th April 1912. Her destination was New York City, America but her last port of call was at Cobh (Queenstown), Ireland on 11th April 1912.
# 
# This dataset contain all passengers record with their surviability.

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# ### Data dictionary
# - **survival** : Survival	: 0 = No, 1 = Yes
# - **pclass**	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# - **sex**: gender of the passenger
# - **Age**: age in year unit
# - **sibsp**: number of siblings / spouses aboard the Titanic	
# - **parch**: number of parents / children aboard the Titanic	
# - **ticket**: Ticket number	
# - **fare**: Passenger fare	
# - **cabin**: Cabin number	
# - **embarked**:	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
# -----
# ### Variable note
# **pclass**: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# 
# **age**: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# **sibsp**: The dataset defines family relations in this way...
# 
# **Sibling** = brother, sister, stepbrother, stepsister
# 
# **Spouse** = husband, wife (mistresses and fiancés were ignored)
# 
# **parch**: The dataset defines family relations in this way...
# 
# **Parent** = mother, father
# 
# **Child** = daughter, son, stepdaughter, stepson, some children travelled only with a nanny, therefore parch=0 for them.

# # 2. Ask some intesting questions/Create hypothesis 
# - Male or Female has higher survival rate ?
# - Passenger who in higher class would have higher rate of survive since the staff will help them first.
# - Age 20-30 should have highest survive rate since they are in the healthiest condition.

# # 3. Clean up process

# ## 3.1 Overview
# Before make any change to the input data, we should quickly see through input data and gain some basic understanding first

# In[ ]:


df.info()


# In[ ]:


df.columns.to_series().groupby(df.dtypes).groups


# ### From `.info()` we know that 
# - Numerical feature are ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch','Age', 'Fare'] 
# - Categorical feature are ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

# In[ ]:


#show few first rows
df.head()


# In[ ]:


#Overview
df.describe()


# ### Now we know that
# - Age, Cabin and Embarked have some missing data.
# - Suvived rate is 0.3(30%)
# - Age has quite high variance compare to other features, 14.52 and 49.69
# - Oldest passenger is 80 years old
# - Highest fare is 512.32 USD
# - Sib, Parch is maximum at 8 and 6 accordingly.

# ## 3.2 Take care missing data

# In[ ]:


df.isna().sum()


# In[ ]:


sns.heatmap(data=df.isna(),yticklabels=False,cmap='coolwarm',cbar=False)


# **Let's fill them up one by one**
# - Age: I think passengers in different classes may have different income rate and age, older may be the one who earn more salary and can spend more. So I will group by `Pclass` then fill missing value with mean.
# - Cabin: There are too many missing data, so I will remove this feature
# - Embarked: Since only 2 of them are missing, and they are not numerical, So I can not replace missing value with mean value. I will fill it by most appear category

# In[ ]:


#Group them by age and find the mean of each Pclass
plt.figure(figsize=(12, 5))
ax = sns.boxplot(data=df,x=df['Pclass'],y=df['Age'],palette='coolwarm') # create plot object.
medians = df.groupby(['Pclass'])['Age'].median().values #get median values
median_labels = [str(np.round(s, 2)) for s in medians] #create label from median values
pos = range(len(medians)) # get range of median values
#Loop to put value label
for tick,label in zip(pos,ax.get_xticklabels()):
    ax.text(pos[tick], medians[tick] + 0.5, median_labels[tick], 
            horizontalalignment='center', size=13, color='r', weight='semibold')


# In[ ]:


#create function to fill age
def fill_age_na(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[ ]:


df['Age'] = df[['Age','Pclass']].apply(fill_age_na,axis=1)


# In[ ]:


df['Age'].isna().sum() # no more missing value


# **Cabin** since there is too many missing value, I will remove it from dataframe
# However, I saw some kernel use this data as representative of passenger location before disaster

# In[ ]:


col_to_drop = ['Cabin']
df.drop(columns=col_to_drop,axis=1,inplace=True)


# In[ ]:


df.columns # 'Cabin' is now removed.


# **Embarked** I will fill missing value by most frequent appear value

# In[ ]:


sns.countplot(x=df['Embarked'])


# In[ ]:


#or use mode
df['Embarked'].mode()[0]


# In[ ]:


#Let's fill it
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace = True)


# In[ ]:


#check if all are filled
df['Embarked'].isna().sum()


# ## 3.3 Correct data
# I will skip this part since I don't see any value to correct

# ## 3.4 Feature engineering/Create new usful feature

# In[ ]:


# create name length feature, since I think longer name may harder to call by staff and lead to death
# you may improve this by removing those initial first(remove Mr. Mrs, Ms, Dr. etc)
df['NameLength'] = df['Name'].apply(len)


# In[ ]:


df['NameLength'].hist(bins=30) #most of passenger has name length around 20-30 character


# In[ ]:


# create family size since bigger family may help each other and all survive
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # plus 1 for passenger itself


# In[ ]:


df['FamilySize'].hist(bins=20) #most of passenger travel alone


# In[ ]:


# create feature IsAlone to see if the passenger travel alone
def IsTravelAlone(col):
    if col == 1:
        return 1
    else:
        return 0


# In[ ]:


df['IsAlone'] = df['FamilySize'].apply(IsTravelAlone)


# In[ ]:


sns.countplot(data=df,x=df['IsAlone']) # most of passenger travel alone


# ## 3.5 Cut unwanted feature
# Base on my understanding to the data, I want to remove following feature before analysis
# - PassengerID: It's just id of the passenger
# - Name: I think name length can represent this feature in more usable way, name itself not usful.
# - Ticket: I don't think ticket number will relate to surviability at all, however if its telling where the passenger sit, may be we can keep it with new formatting.

# In[ ]:


cols_drop = ['PassengerId','Name','Ticket']
df.drop(cols_drop, axis=1, inplace = True)


# # 4. Exploratory Data Analysis
# Personally, I love this part the most. I want to see what data is telling me.

# In[ ]:


#let's see how each feature interact to each other
sns.pairplot(data=df,hue='Sex',size=1.2)


# ## Let's answer the questions

# ### 1. Male or Female has higher survive rate ?

# In[ ]:


print(df.groupby(['Sex'])['Survived'].mean())
sns.countplot(x=df['Sex'],hue=df['Survived']) # total number of survived female is higher and survived mean is also higher than male


# It's also interesting to see what is age range of those male/female who survived.
# You may see that most of female who survived has lower age than female as well.

# In[ ]:


fig = plt.figure(figsize=(10,8))
sns.violinplot(x='Sex',y='Age',hue='Survived',data=df,split=True)


# ### 2. Passenger who in higher class will has higher since staff may help them first.
# plot below show that half or more than half of Pclass=1 are survived, and most of surviver are female that has fare at

# In[ ]:


print(df.groupby(['Pclass'])['Survived'].mean()) # highest class has 62% survaival rate while lowest class has only 24% survival rate
sns.catplot(x='Sex',y='Fare',hue='Survived',data=df,col='Pclass',kind='swarm')


# ### 3. Age 20-30 should have highest survive rate since they are in the healthiest age range
# 20-40 is age range that has highest survived rate, my hypothesis is partially true

# In[ ]:


grid = sns.FacetGrid(data=df,col='Survived',size=8)
grid.map(plt.hist,'Age',bins=50)


# ### Additional exploration

# In[ ]:


#check family size
sns.countplot(x=df['FamilySize'])


# In[ ]:


# is there any relationship between age, fare and class
sns.jointplot(x='Age',y='Fare',data=df)


# In[ ]:


#set overall size
fig = plt.figure(figsize=(15,10))
#set total number of rows and columns
row = 5
col = 2
#set title
fig.suptitle('Various plot',fontsize=20)

#box 1
fig.add_subplot()
ax = fig.add_subplot(2,2,1)
sns.countplot(x='Sex',data=df,hue='IsAlone')
#box 2
ax = fig.add_subplot(2,2,2)
df.groupby('Pclass')['Age'].plot(kind='hist',alpha=0.5,legend=True,title='Pclass vs Age')
#box 3
ax = fig.add_subplot(2,2,3)
df.groupby('Pclass')['Fare'].plot(kind='hist',alpha=0.5,legend=True,title='Pclass vs Fare')
#box 4
ax = fig.add_subplot(2,2,4)
sns.violinplot(x='Sex',y='Age',data=df,hue='Survived',split=True)

#some more setting
plt.tight_layout(pad=4,w_pad=1,h_pad=1.5)
plt.show()


# **At the end of EDA, we may talk to friend or some domain expert to see if our insight are align with the history/domain knowledge or not**

# # 5. Preprocessing
# Since machine learning accept only numerical value, we have to convert all text to number.
# 
# ## 5.1 Convert all categorical feature to nummerical

# In[ ]:


df.head() #which feature are still categorical


# In[ ]:


categorical_feature = []
#loop each column
for i in range(df.shape[1]):
    #if column datatype is object/categorical
    if df[df.columns[i]].dtype == 'object':
        categorical_feature.append(df.columns[i])
        
#show
categorical_feature


# In[ ]:


#convert categorical feature to numerical
#drop_first=True, will help avoid variable dummy trap
df = pd.get_dummies(data=df,columns=categorical_feature,drop_first=True) 
df.head()


# ## 5.2 See how each feature are correlated

# In[ ]:


fig = plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='coolwarm',linewidths=0.2)
plt.show()


# ## 5.3 Split train and test dataset
# Since I have no label for test dataset, So I can not evaluate the model performance. So I will split the train dataset into train and test dataset
# 

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


dfX = df.drop('Survived',axis=1)
dfY = df['Survived']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(dfX, dfY, test_size=0.20, 
                                                    random_state=0)
#I saw some kernel split into train, test and validation. Should I do that to improve the model ?


# In[ ]:


#check size of data
X_train.shape,y_train.shape,X_test.shape,y_test.shape


# ## 5.4 Normalization/Standarization
# There are few popular method to perform this task, it's depend on character of your data.
# For this study I will use StandardScaler as my tool

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()


# In[ ]:


X_train = sc.fit_transform(X_train) #fit scaler with training data
X_test = sc.transform(X_test) #apply scaler to test data


# In[ ]:


X_train[0,:]


# In[ ]:


X_test[0,:]


# Now all feature are scaled and ready to use as model input.

# In[ ]:


df.corr().loc['Survived']


# # 6. Model building
# Since I saw that some feature are correlated(has linear relationship) with `Survived`.
# So I wanted to use `LinearRegression`, but it would not be possible since the output is not limited to 0 and 1. It's can be higher than 1 or lower than 0.

# In[ ]:


# import library to evaluate model
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score


# **Create function to reduce coding**

# ## 6.1 Create baseline model

# In[ ]:


# this part just to show why should not use LinearRegression for binary outcome
from sklearn.linear_model import LinearRegression,LogisticRegression
model_lm = LinearRegression()
model_lm.fit(X_train,y_train)
pred_lm = model_lm.predict(X_test)

# find bad output
bad_output = []
for i in pred_lm:
    if i < 0 or i > 1:
        bad_output.append(i)

bad_output # so let's use LogisticRegression


# **So my baseline model will be LogisticRegression**

# In[ ]:


model_lg = LogisticRegression(solver='lbfgs')


# In[ ]:


model_lg.fit(X_train,y_train)


# In[ ]:


pred_lg = model_lg.predict(X_test)


# In[ ]:


pred_lg


# **Evaluate the model**

# In[ ]:


print(classification_report(y_test,pred_lg)) #classification report

#confusion matrix
cm = confusion_matrix(y_test, pred_lg)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True,cmap='RdYlGn')
plt.title('Model: LogisticRegression \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, pred_lg)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# # 7. Model improvement
# Since the baseline model may not best fit to this problem, we should try different model, hyper-parameter or data shuffle to improve the model performance.
# 
# ** To improve the model, I plan to perform following task **
# 1. Try various machine learning algorithm. Since each algorithm was developed to solve certain problem, so it other model may fit to this problem better than logistic regression. Other machine learning algorithm that I want to try are below.
#     - K-Nearest-Neigbors
#     - Support Vector Machine
#     - Naive bayes
#     - Decision Tree
#     - Random forrest
#     - GradientBoosting
#     - XGBoost
#     - Artificial Neural Network
# 2. Apply cross validation for better generalization
# 3. Find better hyper-parameters for each machine learning algorithm
# 4. Feature selection
# 5. Use differrerent normalization method
# 

# ## 7.1 Try various machine learning algorithm
# Since other machine learning algorithm may perform better on this problem, I will try following model to see if they can give better performance.
# 
# To perform this part I've created a list to store accuracy score of each model.

# In[ ]:


acc_score = [] # create list to store accuracy score


# **Since the steps of each model would be similar to each other, I will use function to wrap those processes into 1 line**

# In[ ]:


def build_train_predict(clf,X_train,y_train,X_test,strAlg,acc_score):
    '''
    1. Create model
    2. Train model
    3. Prediction
    4. Evaluate
    5. Keep score
    '''
    model = clf
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    plot_score(y_test,pred,strAlg,acc_score)
    return clf,pred


# In[ ]:


# create function to plot score for later use
def plot_score(y_test,y_pred,strAlg,lstScore):
    '''
    1. Compare prediction versus real result and plot confusion matrix
    2. Store model accuracy score to list
    '''
    lstScore.append([strAlg,accuracy_score(y_test, y_pred)])
    #print(classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True,cmap='RdYlGn')
    plt.title('Model: {0} \nAccuracy:{1:.3f}'.format(strAlg,accuracy_score(y_test, y_pred)))
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()


# ### 7.1.1 Logistic Regression

# In[ ]:


model_lg,pred_lg = build_train_predict(LogisticRegression(),
                                       X_train,y_train,X_test,
                                       'LogisticRegression',acc_score)


# ### 7.1.2 K-Nearest-Neigbors
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


model_knn,pred_knn = build_train_predict(KNeighborsClassifier(),
                                       X_train,y_train,X_test,
                                       'KNN',acc_score)


# ### 7.1.3 Suport Vector Machine

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


model_svm,pred_svm = build_train_predict(SVC(),
                                       X_train,y_train,X_test,
                                       'SVM',acc_score)


# ### 7.1.4 Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
model_gnb,pred_gnb = build_train_predict(GaussianNB(),
                                       X_train,y_train,X_test,
                                       'GaussianNB',acc_score)


# In[ ]:


from sklearn.naive_bayes import BernoulliNB
model_bnb,pred_bnb = build_train_predict(BernoulliNB(),
                                       X_train,y_train,X_test,
                                       'BernoulliNB',acc_score)


# ### 7.1.5 Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


model_dt,pred_dt = build_train_predict(DecisionTreeClassifier(),
                                       X_train,y_train,X_test,
                                       'DecisionTreeClassifier',acc_score)


# ### 7.1.6 Random forrest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


model_rfc,pred_rfc = build_train_predict(RandomForestClassifier(),
                                       X_train,y_train,X_test,
                                       'RandomForestClassifier',acc_score)


# ### 7.1.7 Gradient Boost

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


model_gbc,pred_gbc = build_train_predict(GradientBoostingClassifier(),
                                       X_train,y_train,X_test,
                                       'GradientBoostingClassifier',acc_score)


# ### 7.1.8 Extra Trees

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier


# In[ ]:


model_et,pred_et = build_train_predict(ExtraTreesClassifier(),
                                       X_train,y_train,X_test,
                                       'ExtraTreesClassifier',acc_score)


# ### 7.1.9 Adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


model_adb,pred_adb = build_train_predict(AdaBoostClassifier(),
                                       X_train,y_train,X_test,
                                       'AdaBoostClassifier',acc_score)


# ### 7.1.10 XGBoost

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


model_xgb,pred_xgb = build_train_predict(XGBClassifier(),
                                       X_train,y_train,X_test,
                                       'XGBClassifier',acc_score)


# ### 7.1.11 Artificial Neural Network

# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense

#get number of input node and number of neuron in hidden layer
dims = X_train.shape[1]
h_dims = int((dims+1)/2)
dims,h_dims

#create model
model_ann = Sequential() #initialize
#input
model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu',input_dim=dims))
#hidden
model_ann.add(Dense(units=h_dims,kernel_initializer='uniform',activation='relu'))
#output
model_ann.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#compile
model_ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#train
model_ann.fit(X_train,y_train,batch_size=32,epochs=100,verbose=0)

#evaluate
pred_ann = model_ann.predict(X_test)
pred_ann = pred_ann > 0.5
plot_score(y_test,pred_ann,'ANN',acc_score)


# In[ ]:


# See the summary, which model is leading
df_acc = pd.DataFrame(acc_score,columns=['Name','TestScore']).sort_values(by=['TestScore','Name'],ascending=False)
df_acc


# ## 7.2 Cross validation
# Accuracy obtain from train dataset can be bias since some extreme observation may not exist in train dataset and lead to bad prediction when facing test dataset.
# cross-validation with Kfold would help on this issue. The score from cross validated model would be more reliable 
# 
# **This part will not be added to accuracy summmary table**

# In[ ]:


from sklearn.model_selection import cross_val_score

#create function to store
def cross_val_MinMaxMean(clf,X_train,y_train,fold):
    scores = cross_val_score(clf,X_train,y_train,cv=fold)
    print('Min: {} \nMax: {} \nMean: {}'.format(scores.min(),scores.max(),scores.mean()))


# ### 7.2.1 Logistic Regression

# In[ ]:


cross_val_MinMaxMean(LogisticRegression(),X_train,y_train,10)


# ### 7.2.2 K-Nearest-Neigbors

# In[ ]:


cross_val_MinMaxMean(KNeighborsClassifier(),X_train,y_train,10)


# ### 7.2.3 Suport Vector Machine

# In[ ]:


cross_val_MinMaxMean(SVC(),X_train,y_train,10)


# ### 7.2.4 Naive Bayes

# In[ ]:


cross_val_MinMaxMean(GaussianNB(),X_train,y_train,10)


# ### 7.2.5 Decision Tree

# In[ ]:


cross_val_MinMaxMean(DecisionTreeClassifier(),X_train,y_train,10)


# ### 7.2.6 Random forrest

# In[ ]:


cross_val_MinMaxMean(RandomForestClassifier(),X_train,y_train,10)


# ### 7.2.7 Gradient Boost

# In[ ]:


cross_val_MinMaxMean(GradientBoostingClassifier(),X_train,y_train,10)


# ### 7.2.8 Extra Trees

# In[ ]:


cross_val_MinMaxMean(ExtraTreesClassifier(),X_train,y_train,10)


# ### 7.2.9 Adaboost

# In[ ]:


cross_val_MinMaxMean(AdaBoostClassifier(),X_train,y_train,10)


# ### 7.2.10 XGBoost

# In[ ]:


cross_val_MinMaxMean(XGBClassifier(),X_train,y_train,10)


# ### 7.2.11 Artificial Neural Network

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold

def create_model():
    model = Sequential()
    model.add(Dense(h_dims,input_dim=dims,activation='relu'))
    model.add(Dense(h_dims,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model,epochs=100,batch_size=10,verbose=0)
kfold = StratifiedKFold(n_splits=10,shuffle=True)
cross_val_MinMaxMean(model,X_train,y_train,kfold)


# ## 7.3 Find better hyper parameters
# 
# This part will try to find better hyper-parameters for each model using `GridSearchCV`

# In[ ]:


#import library for model improvement
from sklearn.model_selection import GridSearchCV

# function to reduce coding
def wrap_gridsearchCV(clf,X_train,y_train,X_test,param_grid,strAlg,acc_score):
    '''
    1. Create GridSearch model
    2. Train model
    3. Predict
    4. Evaluate
    5. Keep score
    '''
    model = GridSearchCV(estimator=clf,param_grid=param_grid,cv=10,
                         refit=True,verbose=0,n_jobs=-1)
    model.fit(X_train,y_train)
    print('\nBest hyper-parameter: {} \n'.format(model.best_params_))
    pred = model.predict(X_test)
    plot_score(y_test,pred,strAlg,acc_score)
    return model,pred


# ### 7.3.1 Logistic Regression

# In[ ]:


param_grid = {
    'C': [0.1,1, 10, 100, 1000],
    'solver': ['newton-cg','lbfgs','liblinear','sag','saga'],
}
model_grid_lg,pred_grid_lg = wrap_gridsearchCV(LogisticRegression(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'LogisticRegression GCV',acc_score)


# ### 7.3.2 K-Nearest-Neigbors

# In[ ]:


param_grid = {
    'n_neighbors': [i for i in range(1,51)]
}
model_grid_knn,pred_grid_knn = wrap_gridsearchCV(KNeighborsClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'KNN GCV',acc_score)


# ### 7.3.3 Suport Vector Machine

# In[ ]:


param_grid = {
    'C': [0.1,1, 10, 100, 1000],
    'gamma': [1,0.1,0.01,0.001,0.0001]
}
model_grid_svm,pred_grid_svm = wrap_gridsearchCV(SVC(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'SVM GCV',acc_score)


# ### 7.3.4 Naive Bayes

# In[ ]:


# there is no hyper-parameter to play with


# ### 7.3.5 Decision Tree

# In[ ]:


param_grid = {
    'max_depth': [None,1,2,3,4,5,7,8,9,10],
    'criterion': ['gini', 'entropy']
}
model_grid_dt,pred_grid_dt = wrap_gridsearchCV(DecisionTreeClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'DecisionTreeClassifier GCV',acc_score)


# ### 7.3.6 Random forrest

# In[ ]:


param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'max_depth': [i for i in range(5,10)],
    'min_samples_leaf': [2,3,4,5]
}
model_grid_rfc,pred_grid_rfc = wrap_gridsearchCV(RandomForestClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'RandomForestClassifier GCV',acc_score)


# ### 7.3.7 Gradient Boost

# In[ ]:


param_grid = {
    'loss': ['deviance', 'exponential'],
    'n_estimators': [i for i in range(100,1000,100)],
    'min_samples_leaf': [1,2,3,4,5]
}
model_grid_gbc,pred_grid_gbc = wrap_gridsearchCV(GradientBoostingClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'GradientBoostingClassifier GCV',acc_score)


# ### 7.3.8 Extra Trees

# In[ ]:


param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'max_depth': [i for i in range(5,10)],
    'min_samples_leaf':[2,3,4,5]
}
model_grid_et,pred_grid_et = wrap_gridsearchCV(ExtraTreesClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'ExtraTreesClassifier GCV',acc_score)


# ### 7.3.9 Adaboost

# In[ ]:


param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'learning_rate' : [0.25, 0.75, 1.00]
}
model_grid_et,pred_grid_et = wrap_gridsearchCV(AdaBoostClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'AdaBoostClassifier GCV',acc_score)


# ### 7.3.10 XGBoost

# In[ ]:


param_grid = {
    'n_estimators': [i for i in range(100,1000,100)],
    'max_depth': [i for i in range(5,10)]
}
model_grid_et,pred_grid_et = wrap_gridsearchCV(XGBClassifier(),
                                               X_train,y_train,X_test,
                                               param_grid,
                                               'XGBClassifier GCV',acc_score)


# ### 7.3.11 Artificial Neural Network

# In[ ]:


## later


# ## 7.4 Feature selection
# All input features can be `Signal` or `Noise`, we may try to keep only feature that seems to give better signal of our model and remove some noise feature.
# This would give better model accuracy, we will test this below.
# 
# Selecting feature is crucial, I may select a list of feature that well correlated with target feature(`survived`) using correlation matrix but it's may lead to overfitting as well.
# I would be better if model can tell us which feature is more important than other.
# 
# Tree based classifier has this attribute after the model is trained.

# In[ ]:


#Take feature importance to select feature for next training
dt_fi = model_dt.feature_importances_
rfc_fi = model_rfc.feature_importances_
gbc_fi = model_gbc.feature_importances_
et_fi = model_et.feature_importances_
ada_fi = model_adb.feature_importances_
xgb_fi = model_xgb.feature_importances_

fi = [dt_fi,rfc_fi,gbc_fi,et_fi,ada_fi,xgb_fi]
model_name = ['DecisionTree','RandomForrest','GradientBoost',
        'ExtraTree','AdaBoost','XGBoost']
model_name = pd.Series(model_name)
df_fi = pd.DataFrame(fi,columns=dfX.columns)
df_fi.index = model_name
df_fi


# In[ ]:


#set overall size
fig = plt.figure(figsize=(20,10))
#set total number of rows and columns
row = 2
col = 3
#set title
fig.suptitle('Feature importance',fontsize=20)

# boxes
for index,i in enumerate(df_fi.index):
    fig.add_subplot()
    ax = fig.add_subplot(2,3,index+1)
    sns.barplot(df_fi.loc[i],df_fi.columns)
    

#some more setting
plt.tight_layout(pad=4,w_pad=1,h_pad=1.5)
plt.show()


# In[ ]:


# Final score table


# In[ ]:


pd.DataFrame(acc_score,columns=['model','score']).sort_values(by=['score','model'],
                                                              ascending=False)


# ### Which model should I use ? Please feel free to suggest
# 
# There are some more tasks to do on this kernel, I will continue soon.
# Thanks to anyone who's reading this ;)

# -------------------------------
# 
# # Misc

# In[ ]:


#display tree graph
import graphviz
from sklearn import tree
tree_dot = tree.export_graphviz(model_dt,out_file=None, 
                                feature_names = dfX.columns, class_names = True,
                                filled = True, rounded = True)
tree_img = graphviz.Source(tree_dot) 
tree_img


# In[ ]:


print("--- %s seconds ---" % (time.time() - start_time))

