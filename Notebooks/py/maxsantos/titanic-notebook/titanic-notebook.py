#!/usr/bin/env python
# coding: utf-8

# **Predicting Survival on the Titanic**
# 
# This notebook covers each of the steps taken during my first Kaggle submission, in which I predict whether or not passengers survived on the Titanic. I have used the process to practice what I have learnt so far in Data Science, and to try and demonstrate the different components of a Data Science/Machine Learning project from start to finish. 
# 
# **1. Importing Data**
# 
# Firstly, I will import the test and train data sets provided by Kaggle as DataFrames, along with a few useful packages. Throughout this project, I will be using the 'train' data set for my analysis and modelling. The 'test' data set will only be used at the very last stage, to make predictions on for submission to Kaggle. 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
from IPython.display import display

test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")


# **2. Exploratory Data Analysis and Data Visualisation**
# 
# It is important to now inspect the data, to see it's structure and what different features we are dealing with. The table below shows each of the columns (or 'features') that are available, along with their type. It is also clear that both Age and Cabin have a significant number of missing values, whilst Embarked is missing two values. This will be dealt with later on in the project, but is useful to know from the start. 
# 
# The obvious first statistic to calculate is the percentage of passengers that survived, which is shown to be 38.4%. In order to predict survival, it is now a good idea to inspect what effect each feature has on this survival rate.

# In[ ]:


display(train.head())
print(train.info())

PercSurvived = train["Survived"].value_counts(normalize=True)*100
print("\n","Percentage Survived","\n",PercSurvived)


# At first glance, the features that are likely to have affected survival are Sex, Embarked (showing where passengers boarded) and Pclass (showing the passengers class). The first plot below demonstrates the large difference between the Male and Female survival rates, due to Women and Children being given priority on the lifeboats. 

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

sex_plot = sns.factorplot(x="Sex", y="Survived", data=train,
                   size=5, kind="bar", palette="muted")
sex_plot.set_ylabels("Probability of Survival")
plt.show()


# There also appears to be a significant difference in survival depending on where passengers Embarked, showing that this may also be a useful feature to consider when fitting an algorithm. The second chart shows a similar (although not identical) trend between Males and Females, and again demonstrates the large affect that gender had on survival. 

# In[ ]:


embarked_plot1 = sns.factorplot(x="Embarked", y="Survived", data=train,
                   size=6, kind="bar", palette="muted")
embarked_plot1.set_ylabels("Probability of Survival")

plt.show()


# In[ ]:


embarked_plot2 = sns.factorplot(x="Embarked", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
embarked_plot2.set_ylabels("Probability of Survival")

plt.show()


# Lastly, the plot below demonstrates the effect of a passengers class on survival. There is a significant drop in survival with increasing Pclass, especially in females going from 2 to 3. I would be useful to create a similar plot for the 'Age' feature, however, this will be done at a later stage, as this column has many missing values.

# In[ ]:


class_plot = sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=train, palette="muted")

plt.show()


# **3. Cleaning and Preparing the Data**
# 
# Before doing any further analysis or fitting of algorithms, I am going to impute the missing values for the Age column (as this clearly has the potential to be an important feature). I have done this using the median of the Age column, as opposed to the mean, which could be affected by extreme values. 
# 
# In order to help with the analysis of the Age column, I will create a new feature called 'Child', which will signify whether a passenger was above or below the age of 13.
# 
# I have also created a 'Family Size' feature, denoting the number of people in a given passengers' family (including siblings, parents/children and themselves), as I feel that this would have been an important factor in a passengers' chances of survival. 
# 
# Lastly, I have simplified the Cabin column, to help with determining whether this is an important feature. 

# In[ ]:


# Impute missing values for age in training set
train["Age"] = train["Age"].fillna(train["Age"].median())

# Create Child column in training set ('Feature Engineering')
train["Child"] = float("NaN")
train.loc[train["Age"] < 13, "Child"] = 1
train.loc[train["Age"] >= 13, "Child"] = 0

# Create Family Size column for training set
train["Family_Size"] = train["SibSp"] + train["Parch"] + 1

# Simplify Cabin column, by slicing off numbers
# NaN Cabin values labelled as 'N'
train["Cabin"] = train["Cabin"].fillna("N")
train["Cabin"] = train["Cabin"].apply(lambda x: x[0])


# 

# Next I am going to visualise the new features that have been created. The 'Child plot' shows the large difference in survival between male children and adults. Interestingly, the survival rate for females is slightly higher for adults, however, this may be due to the random nature of the data, and may not be representative of the whole data set (especially as there is a large confidence interval on the statistic). 
# 
# Family Size appears to have an affect on the survival of males, however the survival rate of females is fairly consistent and appears independant of Family Size. For families of size 5 and above, there is no clear trend, which may be due to a lack of data and small numbers of these large families (which is again demonstrated by the large confidence intervals on these values). 
# 
# Similarly, the Cabin plot shows large confidence intervals, which is due to the fact that there are a very large number of missing values in that column. Due to the majority of the Cabin column being NaN values, I am going to ignore this feature for the rest of the project, although I do believe it may have had an affect on survival and would be worth exploring in future analyses. 

# In[ ]:


# Visualise new features

# Child plot
child_plot = sns.factorplot(x="Child", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
child_plot.set_ylabels("Probability of Survival")

plt.show()

# Famliy_Size plot
family_plot = sns.factorplot(x="Family_Size", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
family_plot.set_ylabels("Probability of Survival")

plt.show()

# Cabin plot
cabin_plot = sns.factorplot(x="Cabin", y="Survived", hue="Sex", data=train,
                   size=6, kind="bar", palette="muted")
cabin_plot.set_ylabels("Probability of Survival")

plt.show()


# Next, it is necessary to convert categorical columns to integer values (such as Sex and Embarked), and fill in any last missing values,  in order to fit an algorithm to the data. 

# In[ ]:


# Convert sex to integer values in training set
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1

# Convert embarked to integer values, and impute missing values
train.loc[train["Embarked"] == "S", "Embarked"] = 0
train.loc[train["Embarked"] == "C", "Embarked"] = 1
train.loc[train["Embarked"] == "Q", "Embarked"] = 2
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].median())

display(train.head())


# **4. Modelling the Data and Making Predictions**
# 
# **Setting Up Training and Test Data**
# 
# I am going to split the 'train' data set in to two sub-sets, one for training different algorithms, and one for fitting them (the 'test' sub-set). I will use a test size of 20%. 
# 
# To do this, I will first define the features I want to use, which are Pclass, Sex, Age, Embarked, Child and Family_Size (for reasons discussed above). My target variable (which I will be predicting), is Survived. 

# In[ ]:


from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = train.drop(['Survived', 'PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)
y = train['Survived']

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=11)


# **First Prediction: Creating and Fitting Random Forest Algorithm**
# 
# For my first prediction I have decided to use the Random Forest algorithm, which makes use of Decision Trees. Decision Trees are a popular algorithm in Data Science, and involve splitting the data in to buckets based on a yes/no condition of different variables. The aim of this splitting is to group the data as precisely as possible according to the target variable ('Survived'). In a perfect tree, this would result in all of the buckets only containing passengers of one type (either Survived=0, or Survived=1). 
# 
# However, Decision Trees can lead to overfitting of data, and the Random Forest algorithm aims to solve this by creating a large number of trees (or a 'forest'!), and then using them to 'vote' on the target variable for each observation. 
# 
# I will also be using a grid search to help choose which parameters to use in the Random Forest classifier. A grid search consists of specifying a 'grid' of different parameter combinations to try, fitting each of them separately, noting how well they perform, and choosing the best performing combination for the final model. It is also essential to use cross-validation whilst doing a grid search, to avoid overfitting. Lastly, I will calculate the importance of each feature in predicting the target variable, giving me an indication of how much each feature is contributing to the model. 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Create random forest classifier
forest = RandomForestClassifier(random_state=11)

# Choose some parameter combinations to try (these values were borrowed from another user)
parameters = {'n_estimators': [4, 6, 9, 100], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
accuracy = make_scorer(accuracy_score)

# Run the grid search with 10-fold cross-validation
grid = GridSearchCV(forest, parameters,scoring=accuracy,cv=10)
grid = grid.fit(X_train, y_train)
print("Tuned Random Forest Parameters: {}".format(grid.best_params_),'\n')
print("Best score is {}".format(grid.best_score_),'\n')

# Set the classifier to the best combination of parameters
forest = grid.best_estimator_

# Fit the best algorithm to the data, and print feature importances & prediction score
forest.fit(X_train, y_train)
print('Feature Importances','\n','Pclass, Sex, Age, Embarked, Child, Family_Size')
print(forest.feature_importances_)

predictions = forest.predict(X_test)
print(accuracy_score(y_test, predictions))



# 

# **Predicting on the Kaggle Test Data**
# 
# Before predicting on the Kaggle 'test' data set, I will do some data cleaning and preparation (as was done before to the 'train' data set). In total, I am going to try three different algorithms for making predictions, and will discuss the accuracy of each at the end of the notebook. 

# In[ ]:


#impute missing values in test
test["Age"] = test["Age"].fillna(test["Age"].median())
#convert to integer values in test
test.loc[test["Sex"] == 'male', 'Sex'] = 0
test.loc[test["Sex"] == 'female', 'Sex'] = 1
test.loc[test["Embarked"] == 'S', 'Embarked'] = 0
test.loc[test["Embarked"] == 'C', 'Embarked'] = 1
test.loc[test["Embarked"] == 'Q', 'Embarked'] = 2
#Create Family Size Column for test
test["Family_Size"] = test["SibSp"] + test["Parch"] + 1
#Create Child column in test set 
test["Child"] = float("NaN")
test.loc[train["Age"] < 13, "Child"] = 1
test.loc[train["Age"] >= 13, "Child"] = 0

display(test.head())


# In[ ]:


#fit tree to test data
test_features = test.drop(['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin'], axis=1)
prediction = forest.predict(test_features)

#create submission file
forestsubmission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
   })

forestsubmission.to_csv("forestsubmission.csv",index=False)

display(forestsubmission.head())


# **Second Prediction: Creating and Fitting Logistic Regression Model**
# 
# Logistic Regression is another common method used, and is worth exploring here. It measures the relationship between each of the features and the categorical target variable (which must have only two possible outcomes), using a logistic function. 
# 
# I have also displayed below the Confusion Matrix for this model, which shows the actual number of correct/incorrect predictions for each value of the target variable. The accuracy of the model can calculated from the matrix, however, accuracy is not always the most suitable measure of a model's success. Therefore, I have also display the Classification Report, which shows the Precision and Recall values, which can be very useful statistics. In this case, a high Precision indicates a low rate of incorrect survival predictions. A high Recall, indicates that a large number of survivals were predicted correctly. 

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions2 = logreg.predict(X_test)
print('Training Set Score: ',logreg.score(X_train, y_train))

# Print confusion matrix, showing actual numbers of correct and incorrect predictions
print('\n','Confusion Matrix:','\n',confusion_matrix(y_test, predictions2))

# Accuracy on test set (diagonal divided by total in confusion matrix)
print('\n','Test Set Score: ',logreg.score(X_test, y_test))

# Print classification report, showing precision and recall calculated from confusion matrix
# high precision = a low rate of incorrect survival predictions
# high recall = predicted a large number of survivals correctly
print('\n','Classification Report: ','\n',classification_report(y_test, predictions2))


# In[ ]:


# Fit logreg to Kaggle test data
prediction2 = logreg.predict(test_features)

# Create submission file
logregsubmission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction2
   })

logregsubmission.to_csv("logregsubmission.csv",index=False)

display(logregsubmission.head())


# **Third Prediction: Creating and Fitting k-Nearest Neighbors Algorithm**
# 
# The last algorithm that I am going to use is k-Nearest Neighbors (or k-NN). This method involves looking at the *k* nearest neighbours to an observation, and using them to 'vote' on which value of the target variable should be assigned. Therefore, for a particular passenger, whatever value of Survived is most common amongst it's neighbors, is what will be used to classify whether that passenger survived or not. 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
predictions3 = knn.predict(X_test)
print('Training Set Score: ',knn.score(X_train, y_train))
print('Test Set Score: ',knn.score(X_test, y_test))



# In[ ]:


# Fit knn to kaggle test data
prediction3 = knn.predict(test_features)

# Create submission file
knnsubmission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction3
   })

knnsubmission.to_csv("knnsubmission.csv",index=False)

display(knnsubmission.head())


# **5. Conclusion**
# 
# Each of the models got the following scores in the Kaggle competition:
# 
# Random Forest:  0.75598
# 
# Logistic Regression: 0.52631
# 
# k-Nearest Neighbors: 0.59808
# 
# As expected, the Random Forest model performed best out of the three, with a score of 76%. This is a reasonable value, however, there are clearly a lot of improvements that could be made. For example, the Cabin and Fare features could be taken in to account, and some more fine tuning of the algorithms could help performance. 
# 
