#!/usr/bin/env python
# coding: utf-8

# # Explaining the steps performed in Data Science in detail, by taking an example of Titanic Survivor Data
# 
# ## NOTE:- This notebook will be helpful to those who are getting their hands dirty in the field of Data Science & Machine Learning, and wants to get a rough idea about the different components involved in the process. Though, every problem has a different solution and approach, the basic components remains same in general.
# 
# ### Special NOTE:- The first and foremost thing you should always remember is you can't find the data first and then solve problems. You should always find the problems first, formulate it, and ask, is it the problem that can be solved by Data Science and Machine Learning/ Deep Learning techniques. As, ML is not a solution for every problem on earth. Anyways, let's dive into the process, I just want to tell people about this small, but important thing!
# 
# ## Let's describe the components invloved in the Data Science process
# _____________________________________________________________________________________________________________
# 
# * **Data acquiring/gathering/collecting** - Data often comes from various sources such as database, web requests, surveys, etc. It can be in the structured, semi-structured or unstructured form. Python/R has good packages/libraries that can take care of data ingestion and manipulation.
# 
# * **Data Preparation** - This is a very crucial step and often this is the step where most data scientists spends time on. This step helps in understanding your data rigorously, also you get to know about the statistics of the data and its distribution. A basic exploratory analysis can be done by visualizing the data and see if there is an anomaly in your data or is it disributed in a strange manner. Another important thing done on this step is taking care of the NULL/NaN values (In real world this is a very common scenario) using various data impuation techniques or simply removing the samples (The data impuations/removal is highly dependent on how well your domain knowledge is for the specific problem).
# 
# * **Analyze Data** - Once your data is pre-processed, the data is ready in the form where it can be feed into the analytics system. There are many ways of analyzing your data, the most common is through stastical and machine learning techniques. After you choose your techniques, you build a model in this step (Building model simply means, training or learning your data with help of learning algorithms and then use this model for further predictions (on unseen data) based on the seen/learned data).
# 
# * **Report your findings** - Once you are statisfied with the results, often you have to communicate this result/findings to the higher authority or a non-technical person. This step is the most important one, because if you can't communicate your findings, the whole point of the process becomes useless. These days, Data Science team in most companies is responsible for market growths, analyzing trends, profits, and studying users behaviour (list goes on and on, just describing few of them!). Hence, your findings should be in such a manner that a non-technical person can understand your reports/graphs/charts and can take data-driven actions.
# 
# ## These are the basic components invloved in data science process
# 
# ### Enough of the reading, I know you must be bored by now! Let's dive into the actual code and go through with each step with the titanic data
# 
#    

# #### Importing the libraries

# In[ ]:


import numpy as np      
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# #### Importing the data 

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')


# #### Let's look at our data

# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# #### From the data, we can see that pre-processing is required. We will take care of NaN values and convert categorical feature like Sex and Embarked into numerical values (Often we have to convert categorical values into numerical values, so we can feed it into Machine Learning models. We will use popular technique called One-Hot Encoding, there are many other methods, but this is often used)

# ### Before we move forward to the pre-processing stage, we will do some exploratory analysis on data to better understand it. In expolratory analysis, we often check for the consistency in values, asks question on the data and see basic trends from it. Also, we find the correlation of the features and remove some features with high correaltion (as they may not contribute much in predictions). Let's do some analysis

# In[ ]:


# This function will describe the basic statistics about data
train_data.describe()


# #### From the above description, it is clear that the data is consitent and have no outliers (for e.g sometimes Age column would have values more than 100 or 150, this is a sign of an outlier)

# In[ ]:


# This will return the correlation values of each feature with each other
train_data.corr()


# #### We can see that most features are playing independent roles in the contribution, except Parch & SibSp. Though the value is not that high, so we will consider all the features in model building

# ### Now let's ask some questions and get to know basic trends from data

# In[ ]:


#Passengers who survived
People_Survived = train_data[train_data['Survived']==1]


# In[ ]:


People_Survived.head()


# In[ ]:


# Let's know how many male and female have been survived
Gender_Survived = People_Survived[['Sex','Survived']].groupby('Sex').count()


# In[ ]:


Gender_Survived['Gender'] = Gender_Survived.index


# In[ ]:


plt.bar(Gender_Survived.iloc[:,1].values, Gender_Survived.iloc[:,0], color='red')
plt.xlabel("Gender")
plt.ylabel("No. of Male and Female Survived")
plt.show()


# #### It's clear from the above analysis that, female has more survival rate than male. (Its' already given to us in the competition description, but this is how we usally approach the preliminary analysis)

# In[ ]:


#Let's know is there any affect on survival due to Pclass
Pclass_survived = People_Survived[['Pclass','Survived']].groupby('Pclass').count()


# In[ ]:


Pclass_survived['pclass'] = Pclass_survived.index


# In[ ]:


plt.bar(Pclass_survived.iloc[:,1].values, Pclass_survived.iloc[:,0], color='blue')
plt.xlabel('Number of Classes')
plt.ylabel('Number of people survived in each class')
plt.xticks(np.arange(1,4))
plt.show()


# #### It's clear from the above figure that, ...

# In[ ]:


#Let's check the number of passengers who has been survived is from which port ?
Embarked_No = People_Survived[['Embarked','PassengerId']].groupby('Embarked').count()


# In[ ]:


Embarked_No['embarked'] = Embarked_No.index


# In[ ]:


plt.bar(Embarked_No.iloc[:,1].values, Embarked_No.iloc[:,0], color='yellow')
plt.xlabel('Ports from where passengers embarked')
plt.ylabel('Total passenger survived from each port')
plt.show()


# In[ ]:


#Let's see the distribution of the age (range) for the passengers who have been survived and not survived
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# #### From above figure we can see that, large number of people who survived are between the age of 15-35 years

# ### In this way we can do some preliminary analysis on the data and get to understand our data well. You can go much further into analysis, these are few of the examples. Now, let's dive into our pre-processing part and clean our data, so that we can feed it into a model

# #### First we will remove the unwanted features from our data

# In[ ]:


train_data = train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)


# In[ ]:


train_data.head()


# In[ ]:


train_data.isnull().sum()


# #### As we can see, we have total 177 NaN values. Instead of removing the entries, we will use data impuation techniques using median to fill the values

# In[ ]:


#Scikit Learn provides a built in imputation class under preprocessing module
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data.iloc[:,3].values.reshape(-1,1))


# In[ ]:


train_data.head(10) # You can see the median value has been updated to Null values. 


# In[ ]:


train_data.isnull().sum()


# #### Now we got 2 NaN values left, so we will simply remove the two entries 

# In[ ]:


train_data = train_data.dropna()


# In[ ]:


train_data.isnull().sum()


# #### Now that our training data is cleaned, we will do the same thing for the test data

# In[ ]:


test_data = test_data.drop(['Name','Ticket','Cabin'],axis=1)


# In[ ]:


test_data.head()


# In[ ]:


test_data.isnull().sum()


# In[ ]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
test_data['Age'] = imputer.fit_transform(test_data.iloc[:,3].values.reshape(-1,1))


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data['Fare'] = imputer.transform(test_data.iloc[:,6].values.reshape(-1,1))


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_df = test_data.drop('PassengerId',axis=1)


# ### Now that our training and testing data are cleaned, we will convert categorical feature to numerical feature, as we can't feed text into machine learning models. One popular method that is widely used is One Hot Encoding. In this method, vectors are created based on the number of categories, let's see it in action to get a good intuition

# In[ ]:


#Pandas has built in method named: get_dummies(), which will take care of One Hot Encoding
train_data = pd.get_dummies(train_data)
test_df = pd.get_dummies(test_df)


# In[ ]:


train_data.head()


# In[ ]:


test_df.head()


# ### This method has created vectors according to the number of categories.
# ### Special NOTE:- You must be thinking that why can't we simply map some numbers to the categories, for e.g. Embarked {C: 1,Q: 2,S: 3}. You asked the right question. Let's discuss why this mapping is bad for the machine learning model. If we map these numbers with the categories, what will happen is that machine learning model will learn and interpret those mappings according to the values of those categories. It will simply learn S being the highest value, Q being the med value and C being the lower value. But, in reality these value has nothing to do with the precedence. These C,Q, & S is just simply the port from where the passengers embarked. So by mapping these values, it will create bias for lower/higher values depending on the problem. This is the reason behind using vectors (One Hot Encoding) instead of simply mapping the values.
# 
# ### Sometimes we have to convert the label for our training data, then in that case we can simply map the numbers. For e.g. the lablel/Outcome for some problem is Spam OR Not Spam : Then we can simply map it like {Spam: 0 , Not Spam: 1} OR {Spam: 1 , Not Spam: 2}

# **---**

# ### Now let do the final step of data pre-processing: normalization. Scaling helps in making our data to be in the same range and help prevents domination by single feature with high values.

# In[ ]:


column_to_normalize = ['Pclass','Age','Fare']


# In[ ]:


train_data[column_to_normalize] = train_data[column_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))  #Min-Max Normalization


# In[ ]:


train_data.head()


# In[ ]:


test_df[column_to_normalize] = test_df[column_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))  #Min-Max Normalization


# In[ ]:


test_df.head()


# ### Now that we are ready with the pre-processed data, we will start feeding the training data into machine learning models and will make predictions on the test data

# In[ ]:


#Let's first seprate our features and labels
X_train = train_data.drop(['Survived'],axis=1)
y = train_data['Survived']


# ## We will use all the popular machine learning algorithms and see which one works best for this problem

# In[ ]:


# These are the list of algorithms that we are going to use to train our model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.metrics import accuracy_score


# In[ ]:


#Logistic Regression
clf = LogisticRegression()
clf.fit(X_train,y)
y_pred_LR = clf.predict(test_df)
log_reg_acc = clf.score(X_train, y)


# In[ ]:


#KNN K-Nearest Neighbors
clf = KNeighborsClassifier()
clf.fit(X_train,y)
y_pred_KNN = clf.predict(test_df)
knn_acc = clf.score(X_train, y)


# In[ ]:


#Support Vector Machine (SVM)
clf = SVC()
clf.fit(X_train,y)
y_pred_svm = clf.predict(test_df)
svm_acc = clf.score(X_train, y)


# In[ ]:


#Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train,y)
y_pred_DT = clf.predict(test_df)
decision_tree_acc = clf.score(X_train, y)


# In[ ]:


#Random Forest
clf = RandomForestClassifier()
clf.fit(X_train,y)
y_pred_RF = clf.predict(test_df)
random_forest_acc = clf.score(X_train, y)


# In[ ]:


#Gradient Boosting
clf = GradientBoostingClassifier()
clf.fit(X_train,y)
y_pred_GB = clf.predict(test_df)
gradient_boosting_acc = clf.score(X_train, y)


# In[ ]:


#Ada Boost 
clf = AdaBoostClassifier()
clf.fit(X_train,y)
y_pred_ada = clf.predict(test_df)
ada_boost_acc = clf.score(X_train, y)


# In[ ]:


Accuracy_df = pd.DataFrame({"Models": ['Logistic Regression','KNN','SVM','Decision Tree','Random Forest','Gradient Boosting', 'Ada Boost'], 
                            "Accuracy":[log_reg_acc,knn_acc,svm_acc,decision_tree_acc,random_forest_acc, gradient_boosting_acc, ada_boost_acc ]})


# In[ ]:


Accuracy_df.sort_values(by='Accuracy',ascending=False)


# ### Decision Tree and Random Forest has the best accuracy among other models. We can further improve the accuracy of all the models by tunning the hyper-parameters. 
# 
# #### Special NOTE:- Often when I had deal with this kind of relational/numerical data, Tree based and Ensemble based methods have proven to be very effective in prediction

# # Here we come to an end of the Data Science process. If you like the notebook, please upvote. All the best and enjoy Data Science!
