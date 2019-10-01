#!/usr/bin/env python
# coding: utf-8

# # TITANIC SURVIVOR PREDICTION.

# ## [please star/upvote if u like it.]

# In[ ]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().magic(u'matplotlib inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import the necessary modelling algos.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


# In[ ]:


train=pd.read_csv(r'../input/train.csv')
#train.head()
test=pd.read_csv(r'../input/test.csv')
#test.head()
df=train.copy()
#df.head()
test_df=test.copy()
#test_df.head()


# In[ ]:


# now data exploration begins.

df.head()
df.index
df.columns
df.shape


# The training dataset has 891 rows or training examples and 12 columns or features. Out of these the 'Survived' is our target variable.

# In[ ]:


df.describe()  # displays different descriptive measures of the numerical features.


#  
# ######  Some Observations --
# 
# age has less than 891 implies that it has some missing(Nan) values.
# 
# the mean of survived indicates that only 38% people survived and rest died.    
# 
# also the age varies from 0.42 to 80. Age less than 1 yr is represented as decimal.
# 
# 50% denotes the median value of features.

# In[ ]:


df.info() # age and cabin both have missing values. also emabarked has some nan values.
# can also use .isnull().sum() to get the count of missing values


# In[ ]:


df.head()


# ###### Some key points about features
# 
# survived is the target variable that we have to predict. 0 means die and 1 means survival.
# 
# Now some of the more relevant features that I will focus on include --
# 
# Pclass:  
# 
# Sex:
# 
# Age:
# 
# Fare:
# 
# Embarked:

# In[ ]:


df.groupby('Survived').Survived.count() # of the given examples 549 people died while only 342 survived.


# In[ ]:


sns.factorplot(x='Survived',data=df,kind='count',palette=['#ff4125','#006400'],size=5,aspect=1)


# In[ ]:


# consider 'Sex' feature.
df[df.Survived==1].groupby('Sex').Survived.count()
pd.crosstab(index=[df.Sex],columns=[df.Survived],margins=True) # set normalize=True to view %.


# ###### 233 female survived while only 109 males. This clearly shows that more females survived than males did. the following graph clearly shows the picture.

# In[ ]:


sns.factorplot(x='Survived',data=df,hue='Sex',palette=['#0000ff','#FFB6C1'],kind='count',size=5,aspect=1)


# In[ ]:


pd.crosstab(index=[df.Sex],columns=[df.Survived],margins=True,normalize='index')  


# In[ ]:


sns.factorplot(x='Sex',y='Survived',kind='point',data=df,palette=['#ff4125'],size=5,aspect=1)


# In[ ]:


sns.factorplot(x='Sex',y='Survived',data=df,kind='bar',palette=['#0000ff','#FFB6C1'],size=5,aspect=1)


# ######  # around 18% of all males survived whereas around 75% of females survived.  This again shows that females survived in greater number.

# In[ ]:


# consider 'Pclass' feature.
df[df.Survived==1].groupby('Pclass').Survived.count()
pd.crosstab(index=[df.Pclass],columns=[df.Survived],margins=True) # set normalize=index to view rowwise %.


# In[ ]:


sns.factorplot(x='Survived',y=None,hue='Pclass',kind='count',data=df,size=5,aspect=1,palette=['#ff0000','#006400','#0000ff'])


# In[ ]:


# consider 'Pclass' feature.
df[df.Survived==1].groupby('Pclass').Survived.count()
pd.crosstab(index=[df.Pclass],columns=[df.Survived],margins=True,normalize=True) # set normalize=index to view rowwise %.


# ###### this again shows that 38% of people survived that accident. also this highlights that only 9% of total passengers who traveled in Pclass 2 survived and rest died and similarly 15% of passengers in Pclass 1 survived and rest died.

# In[ ]:


pd.crosstab(index=[df.Pclass],columns=[df.Survived],margins=True,normalize='index') 


# In[ ]:


sns.factorplot(x='Pclass',y='Survived',kind='point',data=df,size=5,aspect=1,palette=['#ff0000'])


# In[ ]:


sns.factorplot(x='Pclass',y='Survived',data=df,kind='bar',palette=['#ff0000','#006400','#0000ff'],size=5,aspect=1)


# ###### now this shows an even better picture. 75 % of people died in class 3 and only 24% survived. similarly for class 2.for class 1 only 37% died and rest survived probably bcoz of better facilities.in a nutshell most of the people in pclass 1 survived and most of the people in plass 3 died.

# In[ ]:


# grouping by both male or female and the respective Pclasses
pd.crosstab(index=[df.Sex,df.Pclass],columns=[df.Survived],margins=True) 
# the result clearly shows that most of the male in class 2 and 3 died and most of the females in class 1 and 2 survived.
# see the tabulation below.


# In[ ]:


sns.factorplot(x='Sex',hue='Pclass',data=df,col='Survived',kind='count',palette=['#ff0000','#006400','#0000ff'],size=5,aspect=1)


# ###### the graph highlights the picture very clearly. majority of the males in class 2 and 3 died. this is bcoz they were males and wre traveling in a lower class and hence this makes sense. on the other hand most of the females in class 1 and 2 survived which again makes sense as females were given prioroty and also they were traveling in higher classes

# In[ ]:


#now let us see how survival varies with 'Embarked'
df.groupby('Embarked').Survived.count() # most of the people were embarked with 'S'

pd.crosstab(index=[df.Embarked],columns=[df.Survived],margins=True,normalize='index')



# In[ ]:


sns.factorplot(x='Survived',data=df,hue='Embarked',kind='count',palette=['#ff0000','#006400','#0000ff'],size=5,aspect=1)


# In[ ]:


pd.crosstab(index=[df.Sex,df.Embarked],columns=[df.Survived],margins=True)


# In[ ]:


sns.factorplot(x='Sex',data=df,kind='count',hue='Embarked',col='Survived',palette=['#ff0000','#006400','#0000ff'],size=5,aspect=1)


# ###### the graph clearly shows that majority of the males embarked with S died . also very few females died who were embarked C or Q.

# In[ ]:


pd.crosstab(index=[df.Pclass,df.Embarked],columns=[df.Survived],margins=True)


# In[ ]:


sns.factorplot(x='Survived',col='Embarked',data=df,hue='Pclass',kind='count',palette=['#ff0000','#006400','#0000ff'],size=5,aspect=1)


# In[ ]:


# now we need to convert categorical variables into numerical for modelling.
# can use labels or sep col using get_dummies()

#sex
for frame in [train,test,df,test_df]:
   frame.loc[frame.Sex=='male','Sex']=0
   frame.loc[frame.Sex=='female','Sex']=1
   
#embarked    
for frame in [train,test,df,test_df]:
   frame.loc[frame.Embarked=='C','Embarked']=0
   frame.loc[frame.Embarked=='S','Embarked']=1
   frame.loc[frame.Embarked=='Q','Embarked']=2
#df.head(10)
       


# In[ ]:


#now age and fare are continuous variables.
#we can convert them to discrete intervals.

#age
df.Age.describe()   # age varies from 0.42 to 80.00
for frame in [train,test,df,test_df]:
    frame['bin_age']=np.nan
    frame['bin_age']=np.floor(frame['Age'])//10
    frame['bin_fare']=np.nan
    frame['bin_fare']=np.floor(frame['Fare'])//50
    
df.head(10)[['Fare','bin_fare','Age','bin_age']] 
# df.bin_age.unique()
# df.bin_fare.unique()
 


# In[ ]:


#can drop Age and Fare columns
for frame in [train,df,test_df,test]:
    frame.drop(['Age','Fare'],axis=1,inplace=True)
# df.head()
test.head()


# In[ ]:


#now we can see how survival varies with bin_age and bin_fare.
df.groupby('bin_age').Survived.count()
pd.crosstab(index=[df.bin_age],columns=[df.Survived],margins=True)
pd.crosstab(index=[df.Sex,df.Survived],columns=[df.bin_age,df.Pclass],margins=True)


# ###### all the males and females in Pclass 2 and first bin_age survived the accident which hints that children were given preference. also note that in Pclass 1 and bin_age 2,3,4 almost all the people survived which again shows that people in Pclass 1 had better facilities.
# 
# 

# In[ ]:


sns.factorplot(x='bin_age',hue='Survived',kind='count',data=df,palette=['#ff4125','#006400'],size=5,aspect=1)


# In[ ]:


#similarly for bin_fare
df.groupby('bin_fare').Survived.count()
pd.crosstab(index=[df.bin_fare],columns=[df.Survived],margins=True)
pd.crosstab(index=[df.Sex,df.Survived],columns=[df.bin_fare,df.Pclass],margins=True)


# In[ ]:


sns.factorplot(x='bin_fare',hue='Survived',kind='count',data=df,palette=['#ff4125','#006400'],size=5,aspect=1)


# In[ ]:


df.info()
# embarked and bin_age still have null values.
df.describe(include=[np.number])


# ######  now both of these columns have some missing values and so we need to fill the missing values. for now we impute with the median.
# 

# In[ ]:


for frame in [test,train,df,test_df]:
    frame.bin_age.fillna(frame.bin_age.median(),inplace=True)
    frame.Embarked.fillna(frame.Embarked.median(),inplace=True)
    frame.bin_fare.fillna(frame.bin_fare.median(),inplace=True)
# just to check 
df.info()
    


# ######  data exploration, cleaning and preprocessing ends. Now can move onto modelling algorithms.

# In[ ]:


for frame in [train,df,test]:
    frame.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
for frame in [test_df]:
    frame.drop(['Name','SibSp','Parch','Ticket','Cabin'],axis=1,inplace=True)
#df.head()

     


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(df.drop('Survived',axis=1),df.Survived,test_size=0.25,random_state=42)


# In[ ]:


models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),
        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]
model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',
             'GradientBoostingClassifier','GaussianNB']

acc=[]
d={}

for model in range(len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    acc.append(accuracy_score(pred,y_test))
     
d={'Modelling Algo':model_names,'Accuracy':acc}
d


# In[ ]:


acc_frame=pd.DataFrame(d)
acc_frame


# In[ ]:


sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)


# In[ ]:


sns.factorplot(x='Modelling Algo',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)


# ######  most of the algorithms have accuracy b/w 78 to around 81%. this is when we have test size of 0.2 with the given preprocessed data .
# 

# ###### we can now tune the models to increase the accuracies.   

# ######  ALSO NOTE THAT ON THIS MODEL KNN & SVM WITH rbf KERNEL GIVES THE HIGHEST ACCURACY.

# In[ ]:


# first lets try to tune logistic regression.
# here we tune the parameters 'C' and 'penalty'
params_dict={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2']}
clf_lr=GridSearchCV(estimator=LogisticRegression(),param_grid=params_dict,scoring='accuracy')
clf_lr.fit(x_train,y_train)
pred=clf_lr.predict(x_test)
print(accuracy_score(pred,y_test))


# In[ ]:


#now lets try KNN.
#lets try to tune n_neighbors. the default value is 5. so let us vary from say 1 to 50.
no_of_test=[i+1 for i in range(50)]
#no_of_test
params_dict={'n_neighbors':no_of_test}
clf_knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_dict,scoring='accuracy')
clf_knn.fit(x_train,y_train)
pred=clf_knn.predict(x_test)
print(accuracy_score(pred,y_test))


# In[ ]:


#lets RandomForest also. the default value of estimators is 10. so lets vary from 1 to 100.
no_of_test=[]
for i in range(0,101,10):
     if(i!=0):
        no_of_test.append(i)
no_of_test
params_dict={'n_estimators':no_of_test}
clf_rf=GridSearchCV(estimator=RandomForestClassifier(),param_grid=params_dict,scoring='accuracy')
clf_rf.fit(x_train,y_train)
pred=clf_rf.predict(x_test)
print(accuracy_score(pred,y_test))


# In[ ]:


#lets GradientBoosting also. the default value of estimators is 100. so lets vary from 1 to 1000.
no_of_test=[]
for i in range(0,1001,50):
     if(i!=0):
        no_of_test.append(i)
no_of_test
params_dict={'n_estimators':no_of_test}
clf_gb=GridSearchCV(estimator=GradientBoostingClassifier(),param_grid=params_dict,scoring='accuracy',cv=5)
clf_gb.fit(x_train,y_train)
pred=clf_gb.predict(x_test)
print(accuracy_score(pred,y_test))  # results same as before.


# ######  after tuning gradient boosting gives the best accuracy on model.

# In[ ]:


pred=clf_gb.predict(test)
#pred
dict={'PassengerId':test_df['PassengerId'],'Survived':pred}
ans=pd.DataFrame(dict)
ans.to_csv('answer.csv',index=False) # saving to a csv file for predictions on kaggle.


# 

# # THE END.
# 

# ## [please upvote / star if you liked the kernel ]
