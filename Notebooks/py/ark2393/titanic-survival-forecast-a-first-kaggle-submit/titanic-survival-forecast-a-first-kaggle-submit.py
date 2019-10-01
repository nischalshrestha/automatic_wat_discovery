#!/usr/bin/env python
# coding: utf-8

# This is a Python Notebook for  the Titanic Survival Challenge were the objective will not be only to develop a model, will also be analyze the features and compare a variety of models that can help to solve this challenge.
# 
# This notebook will be divided on the next sections:
# 
#  1. Intro 
#  2. Data cleaning and Transformation
#  3. Data Exploration 
#  4. Data Modeling with
#      4.1 Decision Tree Model
#      4.2 Logistic Regression
#      4.3 Support Vector Machine.
#  5. Comparation
#  6. Conclusion
# 
# let's take on count that the predictive models might need different preprocessing steps, so I will use different dataframes if needed

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing libary
from sklearn import preprocessing, svm, tree, linear_model, metrics #methods 
#for machine learning tools like preprocessing, models and metrics
from sklearn.model_selection import train_test_split #replace cross_validation method
import matplotlib.pyplot as plt #plotting library
get_ipython().magic(u'matplotlib inline')


# ## Intro 
# A first impression

# In[ ]:


#first let's read the data and take a look
titanic_df = pd.read_csv("../input/train.csv") 
titanic_df.head()


# There 12 features including our target feature (Survived), so now check how is builded our dataset
# 
# Take this as a note for embarkation: C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


titanic_df.info()


# First, there are three columns with missing data, "Age", "Cabin" and "Embarked". This problem is handled on the next step.
# There are five non-numeric columns, we should take a look and then decide how to transform them. Non-numeric data is troublesome so we will handle it on the next step too.

# ### Cleaning and Transforming Data

# Here the first thing is to handle missing data. There are several ways to deal with them, some of them are removing the record, nullify the value (set to null) or impute values. So we must check first the column and how it can affect the target

# In[ ]:


print("Cabin value count: %d " % titanic_df["Cabin"].count())
print("Age value count: %d" % titanic_df["Age"].count())
print("Embarked value count: %d" % titanic_df["Embarked"].count())


# Cabin column must be dropped, is amount of missing data is to big that it will affect negatively the model.
# For embarked, the value of missing data is very slow, so we can try to imput some value.
# Age contains a lot of missing value, but is not so big as Cabin. We could impute the mean age in the missing ages value, but I will use the simplest way: remove the rows (not column) after filling Embarked missing values

# In[ ]:


titanic_df.drop("Cabin",axis=1,inplace=True)
titanic_df["Embarked"].value_counts()


# There is a majority class, which is "S", so we can input "S" for those missing values and proceed to remove the rows with missing values on Age

# In[ ]:


titanic_df["Embarked"].fillna("S",inplace = True)
titanic_df.dropna(inplace=True)
titanic_df.info()


# Now that all the columns have the same number of records, let's transform the non-numeric data and drop columns to improve our analysis. We can make some suppositions based on the columns info. By example, "Name" column have the name of the passengers, and it should be very unique, so it can be 714 different names and that doesn't help to generalize a model. The same goes to "Ticket" column.
# "PassengerId" is a unique numerical value, but for our goal (predict survival rate) this information doesn't help. Let's check them.

# In[ ]:


print("Name value count: %d " % titanic_df["Name"].value_counts().size)
print("Ticket value count: %d " % titanic_df["Ticket"].value_counts().size)
print("PassengerId value count: %d " % titanic_df["PassengerId"].value_counts().size)
print("Sex value count: %d " % titanic_df["Sex"].value_counts().size)
print("Embarked value count: %d " % titanic_df["Embarked"].value_counts().size)


# As previously said, "Name", "Ticket" and "PassengerID" have very unique values, so we should drop them. For "Sex" and "Embarked", let's transform them into numeric values to improve our analysis. Let's save the true label for later too.

# In[ ]:


titanic_df.drop(["Name","Ticket","PassengerId"],axis=1, inplace=True)
sex_labels= titanic_df["Sex"].unique()
embarked_labels = titanic_df["Embarked"].unique()


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(titanic_df.Sex.values)
titanic_df["Sex"] = le.transform(titanic_df.Sex.values)
sex_labels = titanic_df["Sex"].unique()
sex_labelsE = le.inverse_transform(sex_labels)
le.fit(titanic_df.Embarked.values)
titanic_df["Embarked"] = le.transform(titanic_df.Embarked.values)
embarked_labels = titanic_df["Embarked"].unique()
embarked_labelsE = le.inverse_transform(embarked_labels)


# In[ ]:


titanic_df.head()


# Now let's start the Data Exploration

# ### Data Exploration

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
titanic_df.groupby('Pclass').sum()['Survived'].plot.pie(
    figsize = (8,8), autopct = '%1.1f%%', startangle = 90, fontsize = 15, explode=(0.05,0,0) )
ax.set_ylabel('')
ax.set_title('Survival rate', fontsize = 16)
ax.legend(labels = titanic_df['Pclass'].unique().sort(), loc = "best", title='Class', fontsize=14)


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Survival rate")
titanic_df.groupby("Pclass").mean()["Survived"].plot.bar()
ax.set_xticklabels(labels = ax.get_xticklabels(),rotation=0)


# We can see that Passenger class is a important feature for forecast survival rate. In average more passenger of the 1° class survived and of the survivors distribution they were a majority class.

# In[ ]:


fig = plt.figure()
sorted_labes = [x for (y,x) in sorted(zip(sex_labels,sex_labelsE))]
ax = fig.add_subplot(111)
ax.set_ylabel("Survival rate")
titanic_df.groupby("Sex").mean()["Survived"].plot.bar()
ax.set_xticklabels(labels = sorted_labes,rotation=20)


# Female passengers had a greater chance to survive than male passengers (75% vs 20%), let's see plots with the relation between PClass and Sex for male and female for all the classes

# In[ ]:


index_name=titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"].index.names
index_level=titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"].index.levels
index_ = zip(index_name,index_level)


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3)
titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"][1].plot.bar(ax=axes[0] )
titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"][2].plot.bar(ax=axes[1] )
titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"][3].plot.bar(ax=axes[2] )
axes[0].set_title('Class 1')
axes[0].set_xticklabels(labels = sorted_labes,rotation=20)
axes[0].set_yticks(np.arange(0.0,1.1,0.1))
axes[1].set_title('Class 2')
axes[1].set_xticklabels(labels = sorted_labes,rotation=20)
axes[1].set_yticks(np.arange(0.0,1.1,0.1))
axes[2].set_title('Class 3')
axes[2].set_xticklabels(labels = sorted_labes,rotation=20)
axes[2].set_yticks(np.arange(0.0,1.1,0.1))
fig.tight_layout()


# Now, let's see about ranges of ages grouped by 10 units

# In[ ]:


years_range = np.arange(0,90,10)


# In[ ]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,12))
titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][1,0].plot.bar(ax=axes[0,0], title = ("Women Class 1") )
titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][1,1].plot.bar(ax=axes[0,1], title = ("Men Class 1") )
titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][2,0].plot.bar(ax=axes[1,0], title = ("Women Class 2") )
titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][2,1].plot.bar(ax=axes[1,1], title = ("Men Class 2") )
titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][3,0].plot.bar(ax=axes[2,0], title = ("Women Class 3") )
titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][3,1].plot.bar(ax=axes[2,1], title = ("Men Class 3") )
axes[0,0].set_yticks(np.arange(0.0,1.1,0.1))
axes[0,1].set_yticks(np.arange(0.0,1.1,0.1))
axes[1,0].set_yticks(np.arange(0.0,1.1,0.1))
axes[1,1].set_yticks(np.arange(0.0,1.1,0.1))
axes[2,0].set_yticks(np.arange(0.0,1.1,0.1))
axes[2,1].set_yticks(np.arange(0.0,1.1,0.1))
fig.tight_layout()


# We can see some paterns like  a female passenger had bigger survival rate in all the clases than male passengers. Also we see that children and older people had the same pattern.
# Now let's check the others features

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=3)
sorted_labes = [x for (y,x) in sorted(zip(embarked_labels,embarked_labelsE))]
titanic_df.groupby(["Pclass","Embarked"]).mean()["Survived"][1].plot.bar(ax=axes[0] )
titanic_df.groupby(["Pclass","Embarked"]).mean()["Survived"][2].plot.bar(ax=axes[1] )
titanic_df.groupby(["Pclass","Embarked"]).mean()["Survived"][3].plot.bar(ax=axes[2] )
axes[0].set_title('Class 1')
axes[0].set_yticks(np.arange(0.0,1.1,0.1))
axes[0].set_xticklabels(labels = sorted_labes,rotation=20)
axes[1].set_title('Class 2')
axes[1].set_yticks(np.arange(0.0,1.1,0.1))
axes[1].set_xticklabels(labels = sorted_labes,rotation=20)
axes[2].set_title('Class 3')
axes[2].set_yticks(np.arange(0.0,1.1,0.1))
axes[2].set_xticklabels(labels = sorted_labes,rotation=20)
fig.tight_layout()


# There is a relation between the embarked port and the survival rate, but is not as bigger like Age and Sex

# In[ ]:


titanic_df.groupby("SibSp").mean()["Survived"].plot.bar()


# In[ ]:


titanic_df.groupby("Parch").mean()["Survived"].plot.bar()


# In[ ]:


fare_ranges = np.arange(0,max(titanic_df.Fare)+1,max(titanic_df.Fare)/10)
titanic_df.groupby(pd.cut(titanic_df["Fare"],fare_ranges)).mean()["Survived"].plot.bar()


# Between SibSp, Parch, Fare there is no important correlation.

# Just to be sure, I will run a Random Forest Assesing Feature algorithm (A Feature Selection Algorithm) on the transformed features to asses if my chosed features 

# In[ ]:


titanic_features = titanic_df.drop("Survived", axis=1)
feat_labels = titanic_df.columns[1:]


# In[ ]:


from sklearn import ensemble
forest = ensemble.RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(titanic_features,titanic_df["Survived"])
importances = forest.feature_importances_
indices= np.argsort(importances)[::-1]
for f in range(titanic_features.shape[1]):
    print("%2d) %-*s %f" % (f+1, 30, feat_labels[f], importances[indices[f]]))


# As the data exploration show, the most influential features are Pclass, Sex and Age; the other ones have a little correlation. Let's check also an option with only the major features

# In[ ]:


titanic_3features = titanic_features[titanic_features.columns[:3]]
titanic_3features.head()


# ## Starting with Machine Learning Models

# We have to split the Data into two subsets: training and test data.
# This subsets will allow us to measure the accuracy and precision of the models to chose the best one to use

# In[ ]:


from sklearn import model_selection
from sklearn import preprocessing, metrics


# In[ ]:


#let's standarize the feature value to improve the prediction
sc = preprocessing.StandardScaler()
#------ for all features
sc.fit(titanic_features)
titanic_features_std = sc.transform(titanic_features)
#------ only 3 features
sc.fit(titanic_3features)
titanic_3features_std = sc.transform(titanic_3features)


# In[ ]:


#let's split the data into training and test subsets
#-------- for all features
x_train, x_test, y_train, y_test =  model_selection.train_test_split(
    titanic_features_std, titanic_df.Survived, test_size = 0.3, random_state = 0)
#-------- only 3 features
x_3f_train, x_3f_test, y_3f_train, y_3f_test = model_selection.train_test_split(
    titanic_3features_std, titanic_df.Survived, test_size = 0.3, random_state = 0)


# ### Decision Trees

# Decision Trees are of the most intuitive algorithms, so I will start with it.
# Remember, the first thing is to train the model and test it with the training data.

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
cm_tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)


# #### All features

# In[ ]:


cm_tree.fit(x_train,y_train)


# In[ ]:


y_predict = cm_tree.predict(x_test)
print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))
print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))


# #### 3 major features

# In[ ]:


cm_tree.fit(x_3f_train,y_3f_train)


# In[ ]:


y_3f_predict = cm_tree.predict(x_3f_test)
print("The accuracy is: %2f" % metrics.accuracy_score(y_3f_test,y_3f_predict))
print("The precision is: %2f" % metrics.precision_score(y_3f_test,y_3f_predict))


# ### Logistic Regression

# Of the most classic ML algoritmhs for Classification Task

# In[ ]:


from sklearn.linear_model import LogisticRegression
cm_lr = LogisticRegression(C=1000.0, random_state = 0)


# #### all Features

# In[ ]:


cm_lr.fit(x_train,y_train)


# In[ ]:


y_predict = cm_lr.predict(x_test)
print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))
print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))


# #### Only 3 features

# In[ ]:


cm_lr.fit(x_3f_train,y_3f_train)


# In[ ]:


y_predict = cm_lr.predict(x_3f_test)
print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))
print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))


# ### SVM - Support Vector Classifier

# This Algorithm is very popular in the Classification Task due the performance and the heuristic approach (the algorithm doesn't depend of random values)

# In[ ]:


from sklearn.svm import SVC
svm = SVC(kernel = 'linear', C = 10.0, random_state = 0)


# #### All Features

# In[ ]:


svm.fit(x_train, y_train)


# In[ ]:


y_predict = svm.predict(x_test)
print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))
print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))


# #### Only 3 Features

# In[ ]:


svm.fit(x_3f_train, y_3f_train)


# In[ ]:


y_predict = svm.predict(x_3f_test)
print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))
print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))


# ## Conclusion

# The Data Exploration and Cleaning phases wheren't to deep, but it gave us the enough information to know which columns had to be dropped before creating the ML model and of the remainig features which where the Most Influential ones. 
# On the ML phase the three models are from the simplest one to learn and had a good performance. I can't say that one model is better than other because each one had good points and bad point in different dataset types, and for this one they had very low differneces. Altough, I will use the Logistic Regression to make my submit for this dataset
