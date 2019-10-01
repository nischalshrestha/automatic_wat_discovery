#!/usr/bin/env python
# coding: utf-8

# ### Kernel experience
# - save the output
#     - problems saving multiple files from the same cell (save only the last file)
#     - seems to be random behavion -> dont try to same more output files when push "publish" button
# - submit to kaggle
# - delete a kernel

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# Every programing language has a hello world program version. <br>
# I think Titanic is the hello world for Machine Learning <br>
# 
# Majority, if not all, ML projects have many cycles of development until the final version, if there is one. <br>
# I like to devide each cycle in three phases (similar with lean startup concept):
#     1. get ideas
#     2. implement
#     3. evaluate
# 
# In this notebook, I would like to go through many cycles to see if we can improve the score on competition leaderboard!
# Let's go to work !
# 
# 
# 
# 

# # Cycle one
# The main goals of this cycle is to get used with the dataset, get insights from first visualizations and try the simplest model possible.

# In[ ]:


train_kaggle = pd.read_csv("../input/train.csv")
test_kaggle = pd.read_csv("../input/test.csv")


# ### Few stats about the datasets
# #### training set

# In[ ]:


print("(# of rows, # of columns) " + str(train_kaggle.shape))
train_kaggle.describe(include="all")


# Training set contains in total 891 exemples. <br>
# From the above tables we can see that we deal with measing data, like age and cabin. We will deal with missing data bellow in the notebook.

# #### test set

# In[ ]:


print("(# of rows, # of columns) " + str(test_kaggle.shape))
test_kaggle.describe(include="all")


# Test set contains in total 418 examples. <br>
# Here, we are also dealing with missing data in columns like age, fare and cabin.

# ### First visualization !
# Let's see how balanced are the classes from our target variable : Survived

# In[ ]:


survived_classes = train_kaggle["Survived"].groupby(train_kaggle["Survived"]).count()
survived_classes.plot.bar()


# From above histogram we can see that the number of people who survived are unfortunetely lower that those who died. <br>
# 0 = No, 1 = Yes
# <br>
# Main reasons for death were : (info from http://www.eszlinger.com/titanic/titanfacts.html): <br>
#         * 2,208 lifeboat seats were needed and only 1,178 lifeboat seats were carried aboard.
#         * One of the first lifeboats to leave the Titanic carried only 28 people; it could have held 64 people.
#         * Very few people actually went down with the ship. Most died and drifted away in their life-jackets.
#         
#         

# ### Feature correlations

# In[ ]:


import seaborn as sns
train_corr = train_kaggle.corr(method="spearman")
plt.figure(figsize=(10,7))
sns.heatmap(train_corr, annot=True)


# As we can see from the above heatmap, there is no strong correlation between feature variables.
# It is a good news, it means that all features will have an individual importance to predict the target variable. 
# 
# It is also a good because we have a training dataset of small size. When you have a small training set with many correlated features, it means that in the end you have fewer features to reflat the reality and also it is prone to overfiting.

# ### Give it a try with Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Let's start the easiest way possible. <br>
# Start with the most relevant set of features which also don't contain missing values. <br>
# For me, the most relevant ones would be Pclass, Sex and Fare
# 

# In[ ]:


print(train_kaggle.shape) 
train_kaggle[["Pclass", "Sex", "Fare"]].describe(include="all")


# #### Feature visualization

# #### Sex feature

# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
train_kaggle.groupby("Sex")["Sex"].count().plot.bar(x="Sex", title="Sex feature distribution")

# for the next charts I want to represent the features related to the target variable, survived. 
# Because of this I will create two new columns, alive, not_alive
train_kaggle["Alive"] = train_kaggle["Survived"].apply(lambda s : s )
train_kaggle["Not_alive"] = train_kaggle["Survived"].apply(lambda s : abs(1 - s))
plt.subplot(1, 2, 2)
train_kaggle.groupby(["Sex"])["Alive", "Not_alive"].sum().plot.bar(title="Sex feature distribution related to target variable")



# From the first chart we can see that on titanic was almost double males than females. From the second chart we can see that the chance to survive for a men is more lower than for a female. <br>
# Based on these charts, the sentance [Women and children first](https://en.wikipedia.org/wiki/Women_and_children_first) seems to be true.

# #### Pclass feature

# Based on kaggle description, Pclass represent the ticket class or in other works the socio-economic status

# In[ ]:


train_kaggle.groupby(["Pclass"])["Pclass"].count().plot.bar(title="Distribution of Pclass feature")
train_kaggle.groupby(["Pclass"])["Alive", "Not_alive"].sum().plot.bar(title = "Pclass feature distribution related to target variable")


# It seems that on titanic was way more 'poor' people than 'rich' ones. <br>
# The second chart shows that rich people (Pclass=1) had a bigger chance to survive. Somehow it reflec the reality, because in general righ people are more influential.

# ### Fare features

# Fare feature represent the ticket price each passanger paid.

# In[ ]:


bins = list(range(0, 110, 10))
bins.append(600)
train_kaggle["Fare_category"] = pd.cut(train_kaggle.Fare, bins=bins).apply(lambda x : x.right)
train_kaggle.groupby(["Fare_category"])["Alive", "Not_alive"].sum().plot.bar(figsize=(15,5))


# From the above chart, it seems if you would buy an expensive ticket you would have more change to survive. Somehow this chart reflect the results from Pclass charts.

# In[ ]:


train_kaggle[["Fare"]].plot.box(vert=False, figsize=(15,5))
print ("Mean, median fare " + str(train_kaggle["Fare"].mean()) + ", " + str(train_kaggle["Fare"].median()))


# Based on above boxplot, we have a some outliers in our training set. This can be due to data errors or maybe some passangers paid a lot more than the majority.  <br>
# A good practice is to remove the outliers from the training set, but due to the very small size of titanic training set, it is not such an easy decision. We have to see the results with/without the outliers.
# 

# ### Implement first verion of Decision Tree

# Sex feature is categorical and we need to tranform it into numerical for DecisionTree model.

# In[ ]:


sexLabelEncoder = LabelEncoder()
sexLabelEncoder.fit(train_kaggle["Sex"])
train_kaggle["Sex_encoded"] = sexLabelEncoder.transform(train_kaggle["Sex"])
test_kaggle["Sex_encoded"] = sexLabelEncoder.transform(test_kaggle["Sex"])
train_kaggle.head(5)


# Because the training set is so small I would like to use it all to train the model. <br>
# I will check the performance of the model directly on kaggle test set. Lucky us that we have 10 submits/day

# In[ ]:


dt_col = ["Pclass", "Sex_encoded", "Fare"]
dt_model = DecisionTreeClassifier(random_state=1987)
dt_model.fit(train_kaggle[dt_col], train_kaggle["Survived"])


# Visualize feature importance

# In[ ]:


featureImportance = pd.Series(dt_model.feature_importances_, index=dt_col).sort_values(ascending=True)
featureImportance.plot(kind="barh")


# ### Our first submit to kaggle

# In[ ]:


print("Our initial features are " + str(dt_col))
test_kaggle.describe()


# The kaggle test set contains in total 418 examples. From the above table we can see that the "Fare" features has one missing value.  <br>
# All sklearn models are wainting as features to be all numeric and not missing. Because of the fare missing value, the decision tree predict method will fail. <br>
# 0ne way to handle missing values in sklearn is using Imputer which imput the missing values using either mean, median or most frequent value of the column.
# 

# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


dt_imputer = Imputer(missing_values='NaN', strategy="median")
test_kaggle_imputer = dt_imputer.fit_transform(test_kaggle[dt_col])
test_kaggle_predictions = dt_model.predict(test_kaggle_imputer)


# ## Cycle 1 : Evaluate

# In[ ]:


kaggle_test_dt_submition = pd.DataFrame({"PassengerId":test_kaggle["PassengerId"], "Survived":test_kaggle_predictions})
# kaggle_test_dt_submition.to_csv("dt_submission_0.csv", index=False)


# We obtained the 0.77033 score on kaggle public leaderboard. 

# Let's see where we get wrong predictions using the training set. <br>
# For now we are gonna make these investigations based on Pclass and Sex

# In[ ]:


train_kaggle["dt_prediction"] = dt_model.predict(train_kaggle[dt_col])

wrong_predictions = train_kaggle[train_kaggle["Survived"] != train_kaggle["dt_prediction"]]     .groupby(["Pclass", "Sex"])['Survived']     .agg(['count'])   
survived_counts = train_kaggle.     groupby(["Pclass", "Sex"])["Survived"].     agg(["count"])  
results = pd.     merge(wrong_predictions, survived_counts, left_index=True, right_index=True).     rename(columns={"count_x" : "wrong_label_count", "count_y" : "label_count"} )
    
results["wrong_prediction_percetage"] = (100 * results["wrong_label_count"]) / results["label_count"]
results


# Our model makes the best predictions(percentage) for female from pclass 1 and the worst predictions for male from pclass 1. <br>
# I cannot see a clear pattern where our model makes wrong predictions based on the above table. Maybe you can see one ;)

# All the predictions were made based on a single decision tree. Let's see how it looks !

# In[ ]:


import graphviz 
from sklearn.tree import export_graphviz
def plot_decision_tree(decision_tree, features_names) :
    dot_data = export_graphviz(decision_tree=decision_tree, out_file=None, feature_names=features_names)
    return graphviz.Source(dot_data)


# In[ ]:


plot_decision_tree(dt_model, dt_col)


# That was it for the first cycle. <br>
# It's a pretty good score using the default decision tree hyperparameters and a small subset of features. <br>
# Let's see if we can improve the accuracy tunning some of the hyperparameters!

# In[ ]:


train_kaggle["dt_prediction"] = dt_model.predict(train_kaggle[dt_col])

wrong_predictions = train_kaggle[train_kaggle["Survived"] != train_kaggle["dt_prediction"]]     .groupby(["Fare_category", "Pclass", "Sex"])['Survived']     .agg(['count'])  
    
survived_counts = train_kaggle.     groupby(["Fare_category", "Pclass", "Sex"])["Survived"].     agg(["count"])  
results = pd.     merge(wrong_predictions, survived_counts, left_index=True, right_index=True).     rename(columns={"count_x" : "wrong_label_count", "count_y" : "label_count"} )
    
results["wrong_prediction_percetage"] = (100 * results["wrong_label_count"]) / results["label_count"]
results


# ## Cycle 2
# I suppose the score can be improved using the same set of features and only tuning decision tree hyperparameters. <br>
# In the first cycle we initialized decision tree with its default parameters. This can cause over-complex trees which don't generalize well. <br>
# Sklearn implementation of decision tree offers us multiple options to avoid such problems :
#     * setting max depth
# 

# In[ ]:


max_depth = 6
dt_model_depth = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, random_state=1987)
dt_model_depth.fit(train_kaggle[dt_col], train_kaggle["Survived"])


# max_depth = 4  0.78468 <br>
# max_depth = 5  0.78468 <br>
# max_depth = 6  0.79425 <br>
# max_depth = 7  0.77033 <br>

# In[ ]:


kaggle_test_predictions6 = dt_model_depth.predict(test_kaggle_imputer)
kaggle_test_dt_submition6 = pd.DataFrame({"PassengerId":test_kaggle["PassengerId"], "Survived":kaggle_test_predictions6})
# kaggle_test_dt_submition6.to_csv("dt_depth_"+str(max_depth)+"_submission_0.csv", index=False)


# In[ ]:


plot_decision_tree(dt_model_depth, dt_col)

