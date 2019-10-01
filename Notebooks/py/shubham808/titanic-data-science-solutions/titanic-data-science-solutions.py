#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Science 
# 
# Objective of this Notebook is to Simplify the Titanic Survival Prediction Problem by using a Multiple Layer Perceptron
# 
# The Various Steps are:
# 1. Question or problem definition.
# 2. Acquire training and testing data.
# 3. Wrangle, prepare, cleanse the data.
# 4. Analyze, identify patterns, and explore the data.
# 5. Model, predict and solve the problem.
# 6. Visualize, report, and present the problem solving steps and final solution.
# 7. Supply or submit the results.
# 
# ## Question and problem definition
# 
# > Knowing from a training set of samples listing passengers who survived or did not survive the Titanic disaster, can our model determine based on a given test dataset not containing the survival information, if these passengers in the test dataset survived or not.
# 
#  Here are the highlights to note:
# - On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. Translated 32% survival rate.
# - One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew.
# - Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# Below is all you need to import for this to work so make sure you can import everything.

# In[ ]:



import pandas as pd
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

#sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
#Deep Learning

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.metrics import categorical_accuracy as accuracy
from keras import regularizers
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


# ## Acquire data
# 
# The Python Pandas packages helps us work with our datasets. We start by acquiring the training and testing datasets into Pandas DataFrames. We also combine these datasets to run certain operations on both datasets together.
# We will seperate the labels from the train features

# In[ ]:


features_train = pd.read_csv('../input/train.csv')
features_test = pd.read_csv('../input/test.csv')

labels_train = features_train['Survived']
features_train.drop('Survived', axis=1, inplace=True)


# ## Analyze by describing data
# 
# Pandas also helps describe the datasets answering following questions early in our project.
# 
# **Which features are available in the dataset?**
# 
# Noting the feature names for directly manipulating or analyzing these. These feature names are described on the [Kaggle data page here](https://www.kaggle.com/c/titanic/data).

# In[ ]:


print(features_train.columns.values)


# **Which features are categorical?**
# 
# These values classify the samples into sets of similar samples. Within categorical features are the values nominal, ordinal, ratio, or interval based? Among other things this helps us select the appropriate plots for visualization.
# 
# - Categorical: Survived, Sex, and Embarked. Ordinal: Pclass.
# 
# **Which features are numerical?**
# 
# Which features are numerical? These values change from sample to sample. Within numerical features are the values discrete, continuous, or timeseries based? Among other things this helps us select the appropriate plots for visualization.
# 
# - Continous: Age, Fare. Discrete: SibSp, Parch.

# In[ ]:



features_train.head()


# **Which features are mixed data types?**
# 
# Numerical, alphanumeric data within same feature. These are candidates for correcting goal.
# 
# - Ticket is a mix of numeric and alphanumeric data types. Cabin is alphanumeric.
# 
# **Which features may contain errors or typos?**
# 
# This is harder to review for a large dataset, however reviewing a few samples from a smaller dataset may just tell us outright, which features may require correcting.
# 
# - Name feature may contain errors or typos as there are several ways used to describe a name including titles, round brackets, and quotes used for alternative or short names.

# In[ ]:


features_train.tail()


# **Which features contain blank, null or empty values?**
# 
# These will require correcting.
# 
# - Cabin > Age > Embarked features contain a number of null values in that order for the training dataset.
# - Cabin > Age are incomplete in case of test dataset.
# 
# **What are the data types for various features?**
# 
# Helping us during converting goal.
# 
# - Seven features are integer or floats. Six in case of test dataset.
# - Five features are strings (object).

# ### Assumtions based on data analysis
# 
# We arrive at following assumptions based on data analysis done so far. We may validate these assumptions further before taking appropriate actions.
# 
# 1. We may want to complete Age feature as it is definitely correlated to survival.
# 2. We may want to complete the Embarked feature as it may also correlate with survival or another important feature.
# 
# Dropping Features
# 
# 3. Ticket feature may be dropped from our analysis as it contains high ratio of duplicates.
# 4. Cabin feature may be dropped as it is highly incomplete or contains many null values both in training and test dataset.
# 5. PassengerId may be dropped from training dataset as it does not contribute to survival.
# 6. Name feature may not contribute directly to survival, so it maybe dropped.
# 7. Pclass may also be dropped.
# 
# 
# Creating New Features
# 
# 1. We may want to create a new feature called Family based on Parch and SibSp to get total count of family members on board.
# 2. We may want to engineer the Name feature to extract Title as a new feature.

# ###Scaling Features
# 
# As few of our most informative features are numerical in nature.
# Therefore, the range of all features should be normalized so that each feature contributes approximately proportionately to the output and not generate distortions.
# We will focus on the features Age and Fare while Scaling them.
# We calculate the max value then subtract the mean from each.
# This will be done after we pre-process the features for now we just look at the logic.

# In[ ]:




#Combining Features for uniformity in test and train
combined_features = features_train.append(features_test)
combined_features.reset_index(inplace=True)
combined_features.drop('index', axis=1, inplace=True)


# ###Dropping Features
# 
# This is a good starting goal to execute. By dropping features we are dealing with fewer data points. Speeds up our notebook and eases the analysis.
# Based on our assumptions and decisions we want to drop the Cabin, Pclass and Ticket features.
# 
# Note that where applicable we perform operations on both training and testing datasets together to stay consistent.
# To do this we will first combine them both.

# In[ ]:


combined_features = features_train.append(features_test)
combined_features.reset_index(inplace=True)
combined_features.drop('index', axis=1, inplace=True)


# ### Creating new feature extracting from existing
# 
# We want to analyze if Name feature can be engineered to extract titles and test correlation between titles and survival, before dropping Name and PassengerId features.
# 
# In the following code we extract Title feature using regular expressions. The RegEx pattern `(\w+\.)` matches the first word which ends with a dot character within Name feature. The `expand=False` flag returns a DataFrame.
# 
# When we plot Title, Age, and Survived, we note the following observations.
# 
# - Most titles band Age groups accurately. For example: Master title has Age mean of 5 years.
# - Survival among Title Age bands varies slightly.
# - Certain titles mostly survived (Mme, Lady, Sir) or did not (Don, Rev, Jonkheer).
# 
# - We decide to retain the new Title feature for model training.

# In[ ]:


#processing the importance- titles
combined_features['Title']= combined_features['Name'].map(
    lambda name: name.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
combined_features['Title']=combined_features.Title.map(Title_Dictionary)


# We can convert the categorical titles to ordinal.

# Now we can safely drop the Name feature from training and testing datasets. We also do not need the PassengerId feature in the training dataset.

# ### Converting a categorical feature
# 
# Now we can convert features which contain strings to numerical values. This is required by most model algorithms. Doing so will also help us in achieving the feature completing goal.
# 
# Let us start by converting Sex feature to a new feature called Gender where female=1 and male=0.

# In[ ]:



# Processing sex
combined_features['Sex'] = combined_features['Sex'].map({'male':1,'female':0})


# ### Completing a numerical continuous feature
# 
# Now we should start estimating and completing features with missing or null values. We will first do this for the Age feature.
# 
# We can consider three methods to complete a numerical continuous feature.
# 
# 1. A simple way is to generate random numbers between mean and [standard deviation](https://en.wikipedia.org/wiki/Standard_deviation).
# 
# 2. More accurate way of guessing missing values is to use other correlated features. In our case we note correlation among Age, Gender, and Pclass. Guess Age values using [median](https://en.wikipedia.org/wiki/Median) values for Age across sets of Pclass and Gender feature combinations. So, median Age for Pclass=1 and Gender=0, Pclass=1 and Gender=1, and so on...
# 
# 3. Another meathod would be to use the Title feature we created above and use the average value of the age for each title to fill the gap. This provides a more realistic assumption. 

# In[ ]:


# Processing the ages
grouped_train = combined_features.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()

grouped_test = combined_features.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()
def fillAges(row, grouped_median):
    if row['Sex'] == 'female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 1, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 1, 'Mrs']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['female', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['female', 1, 'Royalty']['Age']

    elif row['Sex'] == 'female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 2, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 2, 'Mrs']['Age']

    elif row['Sex'] == 'female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 3, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 3, 'Mrs']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 1, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 1, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['male', 1, 'Royalty']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 2, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 2, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 2, 'Officer']['Age']

    elif row['Sex'] == 'male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 3, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 3, 'Mr']['Age']

combined_features.head(891).Age = combined_features.head(891).apply(lambda r: fillAges(r, grouped_median_train) if np.isnan(r['Age'])
    else r['Age'], axis=1)

combined_features.iloc[891:].Age = combined_features.iloc[891:].apply(lambda r: fillAges(r, grouped_median_test) if np.isnan(r['Age'])
    else r['Age'], axis=1)


# ### Create new feature combining existing features
# 
# We can create a new feature for FamilySize which combines Parch and SibSp. This will enable us to drop Parch and SibSp from our datasets.

# In[ ]:


# Processing family
combined_features['FamilySize'] = combined_features['Parch'] + combined_features['SibSp'] + 1
combined_features['Singleton'] = combined_features['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined_features['SmallFamily'] = combined_features['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined_features['LargeFamily'] = combined_features['FamilySize'].map(lambda s: 1 if 5 <= s else 0)


# ### Completing a categorical feature
# 
# Embarked feature takes S, Q, C values based on port of embarkation. Our training dataset has two missing values. We simply fill these with the most common occurance.

# In[ ]:



#Processing Cabin:
combined_features.Cabin.fillna('U', inplace=True)


# ### Quick completing and converting a numeric feature
# 
# We can now complete the Fare feature for single missing value in test dataset using mode to get the value that occurs most frequently for this feature. We do this in a single line of code.
# 
# Note that we are not creating an intermediate new feature or doing any further analysis for correlation to guess missing feature as we are replacing only a single value. The completion goal achieves desired requirement for model algorithm to operate on non-null values.
# 
# We may also want round off the fare to two decimals as it represents currency.

# In[ ]:



# Processing Fares: fill empty spaces with mean
combined_features.head(891).Fare.fillna(combined_features.head(891).Fare.mean(), inplace=True)
combined_features.iloc[891:].Fare.fillna(combined_features.iloc[891:].Fare.mean(), inplace=True)


# ###Dropping Features that are not informative
# We find that 
# 
# 
# 1.  Name can be Dropped
# 2. Also PassengerId does not contribute to survival
# 3. Embarked feature can be dropped
# 4. Cabin Feature can be dropped
# 5. As information from SibSp and Parch was extracted into Family Size they will be dropped
# 6. Pclass will be dropped
# 7. Ticket will be dropped

# In[ ]:


# Processing names: drop names and encode the titles using dummy encoding from pandas
combined_features.drop('Name', axis=1, inplace=True)
dummy_titles = pd.get_dummies(combined_features['Title'], prefix='Title')
combined_features = pd.concat([combined_features, dummy_titles], axis=1)
combined_features.drop('Title', axis=1, inplace=True)

# Processing Fares: fill empty spaces with mean
combined_features.head(891).Fare.fillna(combined_features.head(891).Fare.mean(), inplace=True)
combined_features.iloc[891:].Fare.fillna(combined_features.iloc[891:].Fare.mean(), inplace=True)


#Processing Embarked: drop it
combined_features.drop('Embarked', axis=1, inplace=True)

#Processing Ticket: drop it
combined_features.drop('Ticket', axis=1, inplace=True)

#Processing Cabin:
combined_features.Cabin.fillna('U', inplace=True)

# mapping each Cabin value with the cabin letter
combined_features['Cabin'] = combined_features['Cabin'].map(lambda c: c[0])

cabin_dummies = pd.get_dummies(combined_features['Cabin'], prefix='Cabin')
combined_features = pd.concat([combined_features, cabin_dummies], axis=1)
combined_features.drop('Cabin', axis=1, inplace=True)


# Processing sex
combined_features['Sex'] = combined_features['Sex'].map({'male':1,'female':0})


# Processing pclass
pclass_dummies = pd.get_dummies(combined_features['Pclass'], prefix="Pclass")
combined_features = pd.concat([combined_features, pclass_dummies], axis=1)
combined_features.drop('Pclass', axis=1, inplace=True)

# Processing PassengerId
combined_features.drop('PassengerId', axis=1, inplace=True)
# Processing family
combined_features['FamilySize'] = combined_features['Parch'] + combined_features['SibSp'] + 1
combined_features['Singleton'] = combined_features['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combined_features['SmallFamily'] = combined_features['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combined_features['LargeFamily'] = combined_features['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
combined_features.drop('Parch', axis=1, inplace=True)
combined_features.drop('SibSp', axis=1, inplace=True)

#Scaling
scale=max(combined_features['Age'])
combined_features['Age']/=scale
mean=np.mean(combined_features['Age'])
combined_features['Age']-=mean

scale=max(combined_features['Fare'])
combined_features['Fare']/=scale
mean=np.mean(combined_features['Fare'])
combined_features['Fare']-=mean


features_train=combined_features[:891]
features_test=combined_features[891:]


# ## Model, predict and solve
# 
# Now we are ready to train a model and predict the required solution.  We must understand the type of problem and solution requirement to narrow down to a select few models which we can evaluate. Our problem is a classification and regression problem. We want to identify relationship between output (Survived or not) with other variables or features (Gender, Age, Port...).  With these two criteria - Supervised Learning plus Classification and Regression, we can narrow down our choice of models to a few. These include:
# 
# - Logistic Regression
# - Support Vector Machines
# - Decision Tree
# - Random Forrest
# - Artificial neural network

# Logistic Regression is a useful model to run early in the workflow. Logistic regression measures the relationship between the categorical dependent variable (feature) and one or more independent variables (features) by estimating probabilities using a logistic function, which is the cumulative logistic distribution. Reference [Wikipedia](https://en.wikipedia.org/wiki/Logistic_regression).
# 
# Note the confidence score generated by the model based on our training dataset.

# In[ ]:


# Logistic Regression
#


# We can use Logistic Regression to validate our assumptions and decisions for feature creating and completing goals. This can be done by calculating the coefficient of the features in the decision function.
# 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
# 
# - Sex is highest positivie coefficient, implying as the Sex value increases (male: 0 to female: 1), the probability of Survived=1 increases the most.
# - This way Age*Class is a good artificial feature to model as it has second highest negative correlation with Survived.
# - So is Title as second highest positive correlation.

# Next we model using Support Vector Machines which are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training samples, each marked as belonging to one or the other of **two categories**, an SVM training algorithm builds a model that assigns new test samples to one category or the other, making it a non-probabilistic binary linear classifier. Reference [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).
# 
# Note that the model generates a confidence score which is higher than Logistics Regression model.

# In[ ]:


# Support Vector Machines

svc = SVC()
svc.fit(features_train, labels_train)
pred = svc.predict(features_test)
acc_svc = round(svc.score(features_train, labels_train) * 100, 2)
acc_svc


# In pattern recognition, the k-Nearest Neighbors algorithm (or k-NN for short) is a non-parametric method used for classification and regression. A sample is classified by a majority vote of its neighbors, with the sample being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor. Reference [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
# 
# KNN confidence score is better than Logistics Regression but worse than SVM.

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# This model uses a decision tree as a predictive model which maps features (tree branches) to conclusions about the target value (tree leaves). Tree models where the target variable can take a finite set of values are called classification trees; in these tree structures, leaves represent class labels and branches represent conjunctions of features that lead to those class labels. Decision trees where the target variable can take continuous values (typically real numbers) are called regression trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).
# 
# The model confidence score is the highest among models evaluated so far.

# In[ ]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(features_train, labels_train)
Y_pred = decision_tree.predict(features_test)
acc_decision_tree = round(decision_tree.score(features_train, labels_train) * 100, 2)
acc_decision_tree


# The next model Random Forests is one of the most popular. Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Reference [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).
# 
# The model confidence score is the highest among models evaluated so far. We decide to use this model's output (Y_pred) for creating our competition submission of results.

# In[ ]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(features_train, labels_train)
Y_pred = random_forest.predict(features_test)
random_forest.score(features_train, labels_train)
acc_random_forest = round(random_forest.score(features_train, labels_train) * 100, 2)
acc_random_forest


# ###Multiple Layer Perceptron
# 
# Now we will try to use an arificial Neural Network to this problem.
# We see that the number of input features is 25 we will make a sequential  model containing 3 Dense Layers with input dimensions as 25.
# Also we will add regularization. This is necessary due to deficiency of training data which could and in this case does lead to Overfitting.
# Regularization is added in the form of Dropout Layers mingled between the Dense Layers
# You may try to experiment with the ordering and number of layers to get the best output
# So, Try playing Lego a bit with them and try to find something interesting.
# Let me know if you do ;) 

# In[ ]:


model = Sequential()
model.add(Dense(512, input_dim=25))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


# Before Fitting the Model we convert the labels to catagorical features using np_utils so that we may get the softmax probability output which we can later use to predict survival based on probability being greater than 0.5

# In[ ]:


labels_train = np_utils.to_categorical(labels_train)


# we'll use categorical cross entropy for the loss, and adam as the optimizer
# You may want to play around with these too
# its always good to experiment

# In[ ]:



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
clf=model


# Next we fit the model and split the validation at 0.1 or 10 percent this allows us to keep track of overfitting.

# In[ ]:


clf.fit(np.array(features_train),labels_train,validation_split=0.1,epochs=88)
out = clf.predict(np.array(features_test))


# 

# In[ ]:


pred=[]
for x in out:
    if(x[1]>0.5):
        pred.append(1)
    else:
        pred.append(0)

out1=pred
idk = open('test_titanic.csv','r')
idk = csv.DictReader(idk)
pred = []
for row in idk:
    pred.append(row["PassengerId"])
est = csv.writer(open('result.csv', 'w'), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
est.writerows([["PassengerID","Survived"]])
k=0
for x in pred:
    est.writerows([[str(x), str(out1[k])]])
    k+=1


# Our submission to the competition site Kaggle results in scoring 2211 of 6953 competition entries. This result is indicative while the competition is running. This result only accounts for part of the submission dataset. Not bad for our first attempt. Any suggestions to improve our score are most welcome.
