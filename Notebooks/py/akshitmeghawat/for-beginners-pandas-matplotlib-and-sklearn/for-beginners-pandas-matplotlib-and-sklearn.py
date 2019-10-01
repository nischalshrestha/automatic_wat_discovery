#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas


# In[ ]:


train_df = pandas.read_csv('../input/train.csv')


# In[ ]:


# Lets look at some sample rows and general information of the data
train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


'''
As you can see from above, there are 891 entries with 12 columns.
Most of the columns have have all 891 'non-null' values but some don't.
The column named 'Cabin' may not be very useful since it has only 204 'non-null' values.
Lets drop the column 'Cabin'
'''

train_df = train_df.drop(labels = ['Cabin'], axis = 1)


# In[ ]:


'''
We still have two columns left which some missing values: 'Age' and 'Embarked'.
The reason we haven't dropped those is they might be useful features in predicting whether a passenger survived or not.
So instead of dropping the columns, we delete all rows which have null values.
If we had done this before, we would have dropped nearly 700 rows because of the large amount of null values in 'Cabin'
'''

train_df = train_df.dropna()

'''
Note: we have lost a lot of valuable data in this process.
You could also replace the null with dummy values, such as mean, or fill with linear regression etc. 
'''


# In[ ]:


# Now let's visualize our data
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[ ]:


# Lets draw some bar graphs and scatter plots to get a 'feel' of the data 
train_df.Survived.value_counts().plot(kind='bar')
plt.title("Distribution of Survival, (1 = survived, 0 = did not survive)")


# In[ ]:


train_df.Pclass.value_counts().plot(kind="bar")
plt.title("Number of passengers per class")


# In[ ]:


train_df.Embarked.value_counts().plot(kind='bar')
plt.title("Passengers per embarked location")


# In[ ]:


train_df.SibSp.value_counts().plot(kind='bar')
plt.title("Passengers with siblings or spouse")


# In[ ]:


# Now lets compare some columns with survived class

plt.scatter(train_df.Survived, train_df.Age, alpha = 0.1)
plt.title("Age distribution v/s survived")


# In[ ]:


# Looks we do not get much information gain by analysing age

# Lets look at the sex column and compare it with survived class
train_df.Survived[train_df.Sex == 'male'].value_counts().plot(kind = 'bar')
plt.title("Analyzing male passengers: survived and not survived")


# In[ ]:


# As you can see a significantly larger fraction of men did not survive
train_df.Survived[train_df.Sex == 'female'].value_counts().sort_index().plot(kind = 'bar', color='pink')
plt.title("Analyzing female passengers: survived and not survived")


# In[ ]:


# As you can see a significantly larger fraction of women survived
# So sex is a feature with reasonable predicting power

# Lets look at the Pclass and compare it with survived class
train_df.Survived[train_df.Pclass != 3].value_counts().sort_index().plot(kind = 'bar', color = 'green')
plt.title("Analyzing high class passengers: not survived and survived")


# In[ ]:


train_df.Survived[train_df.Pclass == 3].value_counts().sort_index().plot(kind = 'bar', color = 'green')
plt.title("Analyzing low class passengers: not survived and survived")


# In[ ]:


# Majority of the low class passengers did not survive

# Now's let's combine class with sex column
fig = plt.figure(figsize=(18,4), dpi=1600)
ax1=fig.add_subplot(141)
highclass_female = train_df.Survived[train_df.Sex == 'female'][train_df.Pclass != 3].value_counts().sort_index()
highclass_female.plot(kind='bar', label='female, highclass', color='pink')
plt.title("Sex v/s Class")
plt.legend(loc='best')

ax2=fig.add_subplot(142, sharey=ax1)
lowclass_female = train_df.Survived[train_df.Sex == 'female'][train_df.Pclass == 3].value_counts().sort_index()
lowclass_female.plot(kind='bar', label='female, low class', alpha = 0.4, color='pink')
plt.legend(loc='best')

ax3=fig.add_subplot(143, sharey=ax1)
highclass_male = train_df.Survived[train_df.Sex == 'male'][train_df.Pclass != 3].value_counts().sort_index()
highclass_male.plot(kind='bar', label='male, low class',color='blue')
plt.legend(loc='best')

ax4=fig.add_subplot(144, sharey=ax1)
lowclass_male = train_df.Survived[train_df.Sex == 'male'][train_df.Pclass == 3].value_counts().sort_index()
lowclass_male.plot(kind='bar', label='male, highclass', alpha = 0.4, color='blue')
plt.legend(loc='best')


# In[ ]:


# make a copy of the data to make changes
train = train_df.copy()


# In[ ]:


# We'll modify the columns so we can use for logistic regression.
# Convert the 'sex' columns into numbers.

train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1


# In[ ]:


# Converting embarked column to numbers.

train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2


# In[ ]:


# Initial list of predictors
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked"]


# In[ ]:


# Lets try logistic regression on the data.

from sklearn.linear_model import LogisticRegression


# In[ ]:


logit = LogisticRegression()
logit.fit(train[predictors], train.Survived)


# In[ ]:


train_predictions = logit.predict(train[predictors])


# In[ ]:


print(sum(train_predictions == train.Survived) * 1.0 / train.shape[0])


# In[ ]:


# Let's try with the features we predicted
predictors_new = ["Pclass", "Sex", "Age", "Embarked"]
logit_new = LogisticRegression()
logit_new.fit(train[predictors_new], train.Survived)
train_predictions_new = logit_new.predict(train[predictors_new])
print (sum(train_predictions_new == train.Survived) * 1.0 / train.shape[0])


# In[ ]:


# Logistic Regression does not seem to be improving with adding or subtracting features

# Lets try SVM

from sklearn.svm import SVC


# In[ ]:


classifier_linear = SVC(kernel = 'linear', gamma = 3)
classifier_linear.fit(train[predictors], train.Survived)
classifier_linear_predictions = classifier_linear.predict(train[predictors])
print (sum(classifier_linear_predictions == train.Survived) * 1.0 / train.shape[0])


# In[ ]:


# RBF kernel
classifier_rbf = SVC(kernel = 'rbf', gamma = 3)
classifier_rbf.fit(train[predictors], train.Survived)
classifier_rbf_predictions = classifier_rbf.predict(train[predictors])
print (sum(classifier_rbf_predictions == train.Survived) * 1.0 / train.shape[0])


# In[ ]:


# So far SVM with rbf kernel does well.
# Let's test it with cross validation

eighty_precentile = int(.8 * train.shape[0])

train_set = train[:eighty_precentile]
cv_set = train[eighty_precentile:]

# RBF kernel
classifier_rbf_cv = SVC(kernel = 'rbf', gamma = 3)
classifier_rbf_cv.fit(train_set[predictors], train_set.Survived)
cv_predictions = classifier_rbf_cv.predict(cv_set[predictors])
print (sum(cv_predictions == cv_set.Survived) * 1.0 / cv_set.shape[0])


# In[ ]:


# trying a Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc = RandomForestClassifier()
rfc.fit(train[predictors], train.Survived)
print (rfc.score(train[predictors], train.Survived))


# In[ ]:


rfc_with_cv = RandomForestClassifier()
rfc_with_cv.fit(train_set[predictors], train_set.Survived)
print (rfc_with_cv.score(cv_set[predictors], cv_set.Survived))


# In[ ]:


titanic_test = pandas.read_csv("../input/test.csv")


# In[ ]:


# Processing the test set similarly to our training set
titanic_test = titanic_test.drop(labels = ['Cabin', 'Fare'], axis = 1)
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2


# In[ ]:


# Make predictions using the test set.
test_predictions = rfc_with_cv.predict(titanic_test[predictors])


# In[ ]:


# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pandas.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": test_predictions
    })

submission.to_csv("kaggle.csv", index=False)


# In[ ]:




