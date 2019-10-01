#!/usr/bin/env python
# coding: utf-8

# In[89]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# for handling data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualisation
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns

# for machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# importing data
df_train=pd.read_csv('../input/train.csv',sep=',')
df_test=pd.read_csv('../input/test.csv',sep=',')
df_data = df_train.append(df_test) # The entire data: train + test.


# **Missing Data**

# In[90]:


print(pd.isnull(df_data).sum())
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())


# **Description of Training and Testing Data**

# In[91]:


df_train.describe()


# In[92]:


df_test.describe()


# In[93]:


df_data.columns


# **Extracting Name Titles.**
#         We can use the help of name titles to get to know more about passenger categories like Gender, Class, etc.

# In[94]:


df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #Creating new column name Title


# In[95]:


df_data.Title.head() #Lets see the result.


# In[96]:


df_data.Title.tail()


# In[97]:


df_data.Title


# > **Titles like Dona, Mrs, Miss can be classified as Single Title for eg. Miss  
# Titles like Lady Countess Sir can be classified as Single Title for eg. Honor**

# In[98]:


#classify common titles. 
df_data["Title"] = df_data["Title"].replace('Master', 'Master')
df_data["Title"] = df_data["Title"].replace('Mlle', 'Miss')
df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
df_data["Title"] = df_data["Title"].replace(['Don','Jonkheer'],'Mr')
df_data["Title"] = df_data["Title"].replace(['Capt','Rev','Major', 'Col','Dr'], 'Millitary')
df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')


# In[99]:


# Assign in df_train and df_test:
df_train["Title"] = df_data['Title'][:891]
df_test["Title"] = df_data['Title'][891:]

# convert Title categories to Columns
titledummies=pd.get_dummies(df_train[['Title']], prefix_sep='_') #Title
df_train = pd.concat([df_train, titledummies], axis=1) 
ttitledummies=pd.get_dummies(df_test[['Title']], prefix_sep='_') #Title
df_test = pd.concat([df_test, ttitledummies], axis=1) 


# In[100]:


#Fill the na values in Fare
df_data["Embarked"]=df_data["Embarked"].fillna('S') #NAN Values set to S class
df_train["Embarked"] = df_data['Embarked'][:891] # Assign Columns to Train Data
df_test["Embarked"] = df_data['Embarked'][891:] #Assign Columns to Test Data
print('Missing Data Fixed') # Print confirmation (looks good xD)


# In[101]:


# convert Embarked categories to Columns
dummies=pd.get_dummies(df_train[["Embarked"]], prefix_sep='_') #Embarked
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[["Embarked"]], prefix_sep='_') #Embarked
df_test = pd.concat([df_test, dummies], axis=1)
print("Embarked created")


# **Fixing Missing Fare Value **

# In[102]:


# Fill the na values in Fare based on average fare
import warnings
warnings.filterwarnings('ignore')
df_data["Fare"]=df_data["Fare"].fillna(np.median(df_data["Fare"]))
df_train["Fare"] = df_data["Fare"][:891]
df_test["Fare"] = df_data["Fare"][891:]


# **Fixing missing Age Data.**

# In[103]:


titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Millitary','Honor']
for title in titles:
    age_to_impute = df_data.groupby('Title')['Age'].median()[title]
    df_data.loc[(df_data['Age'].isnull()) & (df_data['Title'] == title), 'Age'] = age_to_impute
# Age in df_train and df_test:
df_train["Age"] = df_data['Age'][:891]
df_test["Age"] = df_data['Age'][891:]
print('Missing Ages Estimated')


# **Name Feature**
# 
# We can drop the name feature now that we've extracted the titles.

# In[104]:


#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
df_train['Sex'] = df_train['Sex'].map(sex_mapping)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)

df_train.head()


# **Non numeric features**
# 
# We are going to use a decision tree model. The model requires only numeric values, but one of our features is categorical: "female" or "male". this can easily be fixed by encoding this feature: "male" = 1, "female" = 0. 
# 
# Also we will drop **Cabin** because lots of data is missing and it can create trouble.
# 
# 

# In[105]:


df_train = df_train.drop(['Cabin'], axis = 1)
df_test = df_test.drop(['Cabin'], axis = 1)
df_train = df_train.drop(['Name'], axis = 1)
df_test = df_test.drop(['Name'], axis = 1)


# ## Some Final Encoding
# 
# The last part of the preprocessing phase is to normalize labels. The LabelEncoder in Scikit-learn will convert each unique string value into a number, making out data more flexible for various algorithms. 
# 
# The result is a table of numbers that looks scary to humans, but beautiful to machines. 

# In[106]:


from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Title','Embarked','Ticket']
    
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
df_train, df_test = encode_features(df_train, df_test)
df_train.head()


# **Working on Prediction now**

# **Predict**
# 
# We have our training data, and we have our test data. but in order to evaluate our model we need to split the training dataset into a train dataset and an evaluation dataset (validation). The validation data would be used to evaluate the model, while the training data would be used to train the data.
# 
# To do that, we can use the function "train_test_split" from the sklearn module. the sklean module is probably the most commonly used library in most simple machine learning tasks (this does not include deep learning where other libraries can be more popular)
# 

# In[107]:


from sklearn.model_selection import train_test_split

predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
target = df_train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# 
# **Training the model**
# 
# Now we are finally ready, and we can train the model.
# 
# First, we need to import our model - A decision tree classifier (again, using the sklearn library).
# 
# Then we would feed the model both with the data (X_train) and the answers for that data (y_train)
# 

# In[108]:


#importing Logistic Regression classifier

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train,y_train)


# In[109]:


#printing the training score
print('The training score for logistic regression is:',(model1.score(X_train,y_train)*100),'%')
print('Validation accuracy', accuracy_score(y_val, model1.predict(X_val)))


# In[110]:


#importing random forest classifier

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=6)
model2.fit(X_train,y_train)


# In[111]:


#printing the training score
print('The training score for logistic regression is:',(model2.score(X_train,y_train)*100),'%')
print('Validation accuracy', accuracy_score(y_val, model2.predict(X_val)))


# In[112]:


#importing Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier(n_estimators=7,learning_rate=1.1)
model3.fit(X_train,y_train)


# In[113]:


#printing the training score
print('The training score for logistic regression is:',(model3.score(X_train,y_train)*100),'%')
print('Validation accuracy', accuracy_score(y_val, model3.predict(X_val)))


# In[114]:


from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn import svm #support vector Machine
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
df_test = df_test.dropna()
classifiers=['Logistic Regression','Random Forest','GradientBoosting']
models=[LogisticRegression(),RandomForestClassifier(n_estimators=100),GradientBoostingClassifier(n_estimators=7,learning_rate=1.1)]
for i in models:
    model = i
    cv_result = cross_val_score(model,predictors,target, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2


# In[115]:


plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()


# In[116]:


new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()


# In[123]:


prediction = model2.predict(df_test)
passenger_id = df_data[892:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': prediction } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )


# In[124]:




submission = pd.read_csv('titanic_pred.csv')
print(submission.head())
print(submission.tail())



# In[ ]:





# In[ ]:




