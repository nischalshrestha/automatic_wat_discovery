#!/usr/bin/env python
# coding: utf-8

# **This is my first Kaggle for the Titanic competition.**
# 
# The notebooks explores the basic use of Pandas and scikit-learn for this Classifcation problem.  
# For more advanced approaches like using Seaborn plots, Feature Engineering, GridSearch CV  
# and ML models based on stacking and voting have a look at [my second Titanic kernel](https://www.kaggle.com/dejavu23/titanic-survival-my-2nd-titanic-kernel)
# 
# 
# 
# **My goals for this notebook:**
# 
# * **[Part 1: Exploratory Data Analysis](#Part-1:-Exploratory-Data-Analysis)**  
# understand the data by EDA and derive simple models with Pandas as baseline
# 
# 
# * **[Part 2: Data wrangling](#Part-2:-Data-wrangling)**  
# fill nan, convert categorical to numerical,  
# create train and test data for ML algorithms
# 
# * **[Part 3: Scikit-learn basic ML algorithms](#Part-3:-Scikit-learn-basic-ML-algorithms-and-comparison-of-model-results)**  
# implement different Classifiers from the sklearn library:  
# Logistic regression, Gaussian naive Bayes, KNN, Decision tree, Random forest, SVM
# 
# 
# * **[Part 3: Comparison of Model  results](#Part-3:-Scikit-learn-basic-ML-algorithms-and-comparison-of-model-results)**  
# use metrics like confusion_matrix, classification_report, accuracy_score  
# and implement k fold cross validation for comparison of test score
#  
# 
# **References**  
# **This notebook has some own approaches but is mainly based on these tutorials, notebooks and courses:**
# * **[Datacamp: Kaggle Tutorial: EDA & Machine Learning](https://www.datacamp.com/community/tutorials/kaggle-machine-learning-eda)**
# * **[Udemy: Python for Data Science and Machine Learning Bootcamp](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/)**
# * **[Data School: Machine learning in Python with scikit-learn](https://www.youtube.com/playlist?list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A)**
# 
#     
# 

# ## Some Background Information
# 
# 
# **The sinking of the RMS Titanic in the early morning of 15 April 1912, four days into the ship's maiden voyage from Southampton to New York City, was one of the deadliest peacetime maritime disasters in history, killing more than 1,500 people. The largest passenger liner in service at the time, Titanic had an estimated 2,224 people on board when she struck an iceberg in the North Atlantic. The ship had received six warnings of sea ice but was travelling at near maximum speed when the lookouts sighted the iceberg. Unable to turn quickly enough, the ship suffered a glancing blow that buckled the starboard (right) side and opened five of sixteen compartments to the sea. The disaster caused widespread outrage over the lack of lifeboats, lax regulations, and the unequal treatment of the three passenger classes during the evacuation. Inquiries recommended sweeping changes to maritime regulations, leading to the International Convention for the Safety of Life at Sea (1914), which continues to govern maritime safety.**  
# *from Wikipedia*

# **Imports**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
sns.set()

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore")

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


sns.__version__


# In[ ]:


df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")


# ## Part 1: Exploratory Data Analysis

# In[ ]:


df_train.head()


# df_train has 891 entries, some values for Cabin and Age are missing

# In[ ]:


df_train.info()


# In[ ]:


df_test.head()


# In[ ]:


df_test.info()


# Also in df_test some values for Age and many values for Cabin are missing

# In[ ]:


df_train.describe()


# Comparing distribution of features in df_train and df_test, Pclass and Age seem very similar, distributions for SibSo, Parch and Fare only slightly different

# In[ ]:


df_test.describe()


# **Of all passengers in df_train, how many survived, how many died ?** 

# In[ ]:


sns.countplot(x='Survived', data=df_train);


# In[ ]:


print(df_train.Survived.sum()/df_train.Survived.count())


# more people died than survived (38% survived)
# 
# -> base model : no survivors
# 
# submission : 0.627 accuracy

# In[ ]:


#df_test['Survived'] = 0
#df_test[['PassengerId', 'Survived']].to_csv('no_survivors.csv', index=False)


# **Sex: Female more likely to survive than male**

# In[ ]:


df_train.groupby(['Survived','Sex'])['Survived'].count()


# In[ ]:


sns.catplot(x='Sex', col='Survived', kind='count', data=df_train);


# In[ ]:


print("% of women survived: " , df_train[df_train.Sex == 'female'].Survived.sum()/df_train[df_train.Sex == 'female'].Survived.count())
print("% of men survived:   " , df_train[df_train.Sex == 'male'].Survived.sum()/df_train[df_train.Sex == 'male'].Survived.count())


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(16,7))
df_train['Survived'][df_train['Sex']=='male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
df_train['Survived'][df_train['Sex']=='female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Survived (male)')
ax[1].set_title('Survived (female)')

plt.show()


# Women were more likely to survive than men 
# 
# 74 % of women survived
# but only 19% of men
# (in training set)
# 
# second model :
# all women survived and all men died
# 
# submission : 0.766 accuracy
# 

# In[ ]:


#df_test['Survived'] = df_test.Sex == 'female'
#df_test['Survived'] = df_test.Survived.apply(lambda x: int(x))
#df_test[['PassengerId', 'Survived']].to_csv('women_survive.csv', index=False)


# **Passenger Class : Survival rate decreases with Pclass**

# In[ ]:


pd.crosstab(df_train.Pclass, df_train.Survived, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


print("% of survivals in") 
print("Pclass=1 : ", df_train.Survived[df_train.Pclass == 1].sum()/df_train[df_train.Pclass == 1].Survived.count())
print("Pclass=2 : ", df_train.Survived[df_train.Pclass == 2].sum()/df_train[df_train.Pclass == 2].Survived.count())
print("Pclass=3 : ", df_train.Survived[df_train.Pclass == 3].sum()/df_train[df_train.Pclass == 3].Survived.count())


# In[ ]:


sns.catplot('Pclass','Survived', kind='point', data=df_train);


# **Passenger Class and Sex :**
# 
# **Almost all women in Pclass 1 and 2 survived and nearly all men in Pclass 2 and 3 died**

# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], df_train.Pclass, margins=True).style.background_gradient(cmap='autumn_r')


# In[ ]:


sns.catplot('Pclass','Survived',hue='Sex', kind='point', data=df_train);


# **Embarked : Survival rate lowest for S and highest for C**

# In[ ]:


sns.factorplot(x='Survived', col='Embarked', kind='count', data=df_train);


# In[ ]:


sns.factorplot('Embarked','Survived', kind='point', data=df_train)
plt.show()


# **Embarked and Sex**

# In[ ]:


sns.factorplot('Embarked','Survived', hue= 'Sex', kind='point', data=df_train)
plt.show()


# **Embarked, Pclass and Sex :**
# 
# ** Practically all women of Pclass 2 that embarked in C and Q survived, also nearly all women of Pclass 1 survived. **
# 
# ** All men of Pclass 1 and 2 embarked in Q died, survival rate for men in Pclass 2 and 3 is always below 0.2 **
# 
# ** For the remaining men in Pclass 1 that embarked in S and Q, survival rate is approx. 0.4 **

# In[ ]:


sns.factorplot('Embarked','Survived', col='Pclass', hue= 'Sex', kind='point', data=df_train)
plt.show()


# In[ ]:


pd.crosstab([df_train.Survived], [df_train.Sex, df_train.Pclass, df_train.Embarked], margins=True)


# third model :
# based on PClass, Sex and Embarked ,
# submission : 0.779 accuracy

# In[ ]:


# model 3
df_test['Survived'] = 0
# all women survived
df_test.loc[ (df_test.Sex == 'female'), 'Survived'] = 1
# except for those in Pclass 3 and embarked in S
df_test.loc[ (df_test.Sex == 'female') & (df_test.Pclass == 3) & (df_test.Embarked == 'S') , 'Survived'] = 0
#df_test[['PassengerId', 'Survived']].to_csv('embarked_pclass_sex.csv', index=False)


# **Age:  continuous numerical  to  8 bins **

# In[ ]:


for df in [df_train, df_test]:
    df['Age_bin']=np.nan
    for i in range(8,0,-1):
        df.loc[ df['Age'] <= i*10, 'Age_bin'] = i


# In[ ]:


print(df_train[['Age' , 'Age_bin']].head(10))


# In[ ]:


sns.factorplot('Age_bin','Survived',hue='Sex',kind='point',data=df_train)
plt.show()


# In[ ]:


sns.factorplot('Age_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
plt.show()


# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Age_bin, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# For passengers in Age_bin = 1 (younger than 10) : All male in Pclass 1 and 2 survived
# 
# All female in Pclass 3 and Age_bin = 5 died. 
# 
# (Survival rate for female in Pclass 3 and Age_bin = 4 is below 50%)
# 
# (Survival rate for male in Pclass 1 and Age_bin = 4 is above 50%)
# 
# 

# In[ ]:


# in Pclass 1 and 2 all men in Age_bin = 1 survived
df_test.loc[ (df_test.Sex == 'male') & (df_test.Pclass == 1) & (df_test.Age_bin == 1), 'Survived'] = 1
df_test.loc[ (df_test.Sex == 'male') & (df_test.Pclass == 2) & (df_test.Age_bin == 1), 'Survived'] = 1


# **SibSp and Parch**

# In[ ]:


sns.factorplot('SibSp','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
plt.show()


# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.SibSp, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# For males, no survival rate above 0.5 for any values of SibSp.
# For females, passengers with SibSp = 3 and Pclass = 3 died, also all females with SibSp > 4 died.
# For females with SibSp = 1 and Pclass = 3 survival rate is below 0.5

# In[ ]:


# all females with SibSp > 7 died
df_test.loc[ (df_test.Sex == 'female') & (df_test.SibSp > 7) , 'Survived'] = 0


# In[ ]:


sns.factorplot('Parch','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
plt.show()


# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Parch, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# Very similar to SibSp - 1 , but different values  ?
# For females with Parch = 2 and Pclass = 3 survival rate is below 0.5  
# All females with Parch = 4 and Pclass = 3 died.
# All females with Parch > 4 died.
# (For females with Parch = 1 and Pclass = 3 survival rate is below 0.5)
# For males,all survival rates below 0.5 for any values of Parch, except for Parch = 2 and Pclass = 1.

# In[ ]:


# survival rate is below 0.5 for females with Parch = 2 and Pclass = 3 
#df_test.loc[ (df_test.Sex == 'female') & (df_test.Pclass == 3) & (df_test.Parch == 2), 'Survived'] = 0

# All females with Parch = 4 and Pclass = 3 died
##df_test.loc[ (df_test.Sex == 'female') & (df_test.Pclass == 3) & (df_test.Parch == 4), 'Survived'] = 0

# all females with Parch > 4 died
#df_test.loc[ (df_test.Sex == 'female') & (df_test.Parch > 4) , 'Survived'] = 0

# For males with Parch = 2 and Pclass = 1 survival rate is above 0.5
##df_test.loc[ (df_test.Sex == 'male') & (df_test.Pclass == 1) & (df_test.Parch == 1) , 'Survived'] = 1

#df_test.head(20)


# **Fare:  continuous numerical  to  12 bins **

# In[ ]:


sns.distplot(df_train['Fare'])
plt.show()


# In[ ]:


for df in [df_train, df_test]:
    df['Fare_bin']=np.nan
    for i in range(12,0,-1):
        df.loc[ df['Fare'] <= i*50, 'Fare_bin'] = i


# In[ ]:


sns.factorplot('Fare_bin','Survived', col='Pclass' , row = 'Sex', kind='point', data=df_train)
plt.show()


# In[ ]:


pd.crosstab([df_train.Sex, df_train.Survived], [df_train.Fare_bin, df_train.Pclass], margins=True).style.background_gradient(cmap='autumn_r')


# 

# In[ ]:


# males in Fare_bin = 11 survived
df_test.loc[ (df_test.Sex == 'male') & (df_test.Fare_bin == 11), 'Survived'] = 1


# fourth model : model 3 + Age_bin, SibSp, Parch and Fare_bin  
# submission : 0.789 accuracy

# In[ ]:


# model 4
# df_test[['PassengerId', 'Survived']].to_csv('model_4.csv', index=False)


# In[ ]:


df_test.drop(['Survived'],axis=1,inplace=True)


# 
# ## Part 2: Data wrangling
# 
# building two new dataframes df_train_ml and df_test_ml  
# these will have only ordinal features and no missing values so they can be used for ML algorithms  
# converting categorical to numerical by pd.get_dummies  
# dropping all features that seem to be not useful for prediction  
# Then use the Standard scaler and apply train/test split

# In[ ]:


df_train_ml = df_train.copy()
df_test_ml = df_test.copy()


# In[ ]:


df_train_ml = pd.get_dummies(df_train_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_train_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)
df_train_ml.dropna(inplace=True)


# In[ ]:


passenger_id = df_test_ml['PassengerId']
df_test_ml = pd.get_dummies(df_test_ml, columns=['Sex', 'Embarked', 'Pclass'], drop_first=True)
df_test_ml.drop(['PassengerId','Name','Ticket', 'Cabin', 'Age_bin', 'Fare_bin'],axis=1,inplace=True)


# In[ ]:


df_train_ml.head(10)


# In[ ]:


df_train_ml.info()


# In[ ]:


df_test_ml.info()


# In[ ]:


#df_test_ml.drop(['Survived'],axis=1,inplace=True)
df_test_ml.head()


# In[ ]:


df_test_ml.info()


# **Correlation Matrix**

# In[ ]:


corr = df_train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# Survived and Fare positively correlated, Survived and Sex_male negatively correlated.  
# Also, Survived and Pclass_3 negatively correlated. SibSp and Parch correlated

# **Standard Scaler**

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for df_train_ml
scaler.fit(df_train_ml.drop('Survived',axis=1))
scaled_features = scaler.transform(df_train_ml.drop('Survived',axis=1))
df_train_ml_sc = pd.DataFrame(scaled_features, columns=df_train_ml.columns[:-1])

# for df_test_ml
df_test_ml.fillna(df_test_ml.mean(), inplace=True)
# scaler.fit(df_test_ml)
scaled_features = scaler.transform(df_test_ml)
df_test_ml_sc = pd.DataFrame(scaled_features, columns=df_test_ml.columns)


# 

# **train_test_split**

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train_ml.drop('Survived',axis=1), df_train_ml['Survived'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(df_train_ml_sc, df_train_ml['Survived'], test_size=0.30, random_state=101)


# **all data for submission**

# In[ ]:


# unscaled
X_train_all = df_train_ml.drop('Survived',axis=1)
y_train_all = df_train_ml['Survived']
X_test_all = df_test_ml

# scaled
X_train_all_sc = df_train_ml_sc
y_train_all_sc = df_train_ml['Survived']
X_test_all_sc = df_test_ml_sc


# **fillna: fill nan with mean values for that column** 

# In[ ]:


X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")


# In[ ]:





# ## Part 3: Scikit-learn basic ML algorithms and comparison of model results
# 
# **Test simple sklearn models and compare by metrics**
# 
# 
# **We test the following classifiers from scikit-learn:**  
# Logistic regression  
# Gaussian Naive Bayes  
# K nearest neighbors KNN  
# Decision tree classifier  
# Random forest classifier  
# SVM classifier
# 
# **For comparison of the results we use these metrics:**  
# accuracy_score, classification_report, confusion_matrix
# 
# 
# **First we apply the data from test/train split to get a first overview of the model performance.  
# Later we use the k fold cross validation which gives a better estimate for out of sample data.** 

# In[ ]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# Train again for all data and submit

# In[ ]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[ ]:


sub_logreg = pd.DataFrame()
sub_logreg['PassengerId'] = df_test['PassengerId']
sub_logreg['Survived'] = pred_all_logreg
#sub_logmodel.to_csv('logmodel.csv',index=False)


# **Gaussian Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[ ]:





# **KNN - KNeighborsClassifier**

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[ ]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[ ]:


knn.fit(X_train_all, y_train_all)
pred_all_knn = knn.predict(X_test_all)


# In[ ]:


sub_knn = pd.DataFrame()
sub_knn['PassengerId'] = df_test['PassengerId']
sub_knn['Survived'] = pred_all_knn
#sub_knn.to_csv('knn.csv',index=False)


# **Decision Tree Classifier**

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[ ]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# another decision tree with different parameters for max_features, max_depth and min_sample_split

# In[ ]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# all data

# In[ ]:


dtree_2.fit(X_train_all, y_train_all)
pred_all_dtree2 = dtree_2.predict(X_test_all)


# In[ ]:





# In[ ]:





# **Random Forest Classifier**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6, max_features=7)
rfc.fit(X_train, y_train)


# In[ ]:


pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test, pred_rfc))


# Train again for all data and submit

# In[ ]:


rfc.fit(X_train_all, y_train_all)
pred_all_rfc = rfc.predict(X_test_all)


# In[ ]:


sub_rfc = pd.DataFrame()
sub_rfc['PassengerId'] = df_test['PassengerId']
sub_rfc['Survived'] = pred_all_rfc
#sub_rfc.to_csv('randforest.csv',index=False)


# In[ ]:





# **SVM Classifier**

# In[ ]:


from sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100)#, probability=True)
svc.fit(X_train_sc, y_train_sc)


# In[ ]:


pred_svc = svc.predict(X_test_sc)
print(confusion_matrix(y_test_sc, pred_svc))
print(classification_report(y_test_sc, pred_svc))
print(accuracy_score(y_test_sc, pred_svc))


# In[ ]:





# In[ ]:


svc.fit(X_train_all_sc, y_train_all_sc)
pred_all_svc = svc.predict(X_test_all_sc)

sub_svc = pd.DataFrame()
sub_svc['PassengerId'] = df_test['PassengerId']
sub_svc['Survived'] = pred_all_svc
sub_svc.to_csv('svc.csv',index=False)


# In[ ]:





# **k fold cross_validation**
# 
# This algorithm splits the data into k sets and then makes k fits using every set k-1 times as training and one time as test data  
# It leads to a better estimate for out of sample data  than simple train test split

# In[ ]:


from sklearn.model_selection import cross_val_score


# for SVM classifier

# In[ ]:


scores_svc = cross_val_score(svc, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_svc)
print(scores_svc.mean())


# for Random Forest classifier

# In[ ]:


scores_rfc = cross_val_score(rfc, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_rfc)
print(scores_rfc.mean())


# for DecisionTreeClassifier

# In[ ]:


scores_dtree_2 = cross_val_score(dtree_2, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_dtree_2)
print(scores_dtree_2.mean())


# **The Classifiers with best performance are Decision Tree, Random Forest and SVC**
# 

# In[ ]:


print("dtree_2 : " , scores_dtree_2.mean())
print("rfc     : " , scores_rfc.mean())
print("svc     : " , scores_svc.mean())


# **Note on scores**
# 
# Some kernels for the Titanic competition calculate scores based on the training set. This is not a good indicator for the model performance, because we want to know how well the model generalizes for data that was not used for fitting the model. Therefore, scores in this and in my other kernels are always for out of sample test or validation data.

# **Conclusion**  
# With this notebook I learned the basics of EDA with Pandas and Matplotlib as well as the foundations  
# for applying the classification models of the scikit learn library.  
# By EDA we found a strong impact of features like Sex, Age, Embarked on the target.  
# We then built a simple baseline model with Pandas, using only these features.  
# Again using Pandas, we also created a dataset that can be used by the sklearn Classifiers for prediction.  
# Deciding by k fold cross validation score, the best ML models for this task and set of features were:  
# Decision Tree, Random Forest and SVC  
# Submitting their predictions gives a score of 0,78 and a place in the top 30% of the Leaderboard.  
# 
# In [my second Titanic kernel](https://www.kaggle.com/dejavu23/titanic-survival-my-2nd-kernel) I study how to improve this score by 
# * using more or other features built from the existing ones (Feature Engineering)
# * tuning the model parameters with GridSearchCV
# * and applying techniques like boosting, stacking and voting

# 

# 
