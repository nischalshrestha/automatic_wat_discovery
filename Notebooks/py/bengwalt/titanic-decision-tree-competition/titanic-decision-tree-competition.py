#!/usr/bin/env python
# coding: utf-8

# This is my first notebook on Kaggle! I follow the general framework laid out in [this notebook by Diego Milla](https://www.kaggle.com/dmilla/introduction-to-decision-trees-titanic-dataset) with some additional exploratory data analysis and some slight changes in the way the data was organized. His model was slightly more accurate because of the changes I made to the <b>Age</b> feature.

# In[ ]:


# some of the more general imports needed for the script
import numpy as np
import pandas as pd
import re
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

# showing overview of the train dataset
train.head(10)


# In[ ]:


original_train = train.copy() 
full_data = [train, test]


# In[ ]:


# Feature that tells whether a passenger had a cabin on the Titanic
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)


# In[ ]:


# let's start by exploring age

train[['Age']].describe()


# In[ ]:


train[['Age']].isnull().sum()


# In[ ]:


# Here we remove all NULLS in the 'Age' column and replace with pseudorandom ages derived from the mean
# and standard deviation of the data belonging to the 'Age' column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(float)
    
train[['Age']].describe()


# In[ ]:


# let's split this into quantiles and see what we get

train[['Age']].quantile([0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.98])


# In[ ]:


# taking a look at the distribution

sns.distplot(train['Age'], color='Red')


#  ### Age Normality and EDA
# Some tests for normality and miscellaneous EDA. We need to decide if we are going to use a Gaussian approach or not and exploring some of the features will help us to do this.

# In[ ]:


import pylab
from scipy.stats import probplot


measurements = train['Age']   
probplot(measurements, dist="norm", plot=pylab)
pylab.show()


# In[ ]:


from scipy.stats import shapiro

# some tests for normality, we need to decide if we are going to use a gaussian approach

# normality test
stat, p = shapiro(train['Age'])
print('Shapiro-Wilk Statistic=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[ ]:


for dataset in full_data:
    dataset['age_quantile'] = pd.qcut(dataset['Age'], 10, labels=False)
    dataset['age_quantile'] 
    
train['age_quantile'].describe()


# In[ ]:


for dataset in full_data:
    dataset['age_quantile'] = dataset['age_quantile'].fillna(dataset['age_quantile'].mean())
    
train['age_quantile'].isnull().sum()


# In[ ]:


train['age_quantile'].value_counts().sort_index()


#  <b>Age</b> is not normally distributed a if we think that a Gaussian approach could be effective, our tests tell us otherwise. Intuition tells us that age could be a strong predictor of survivability. Let's take a look at the formal titles held by passengers (Mrs., Mr., Dr., etc.) and Sex. First, we need to create and map some clever features.

# In[ ]:


from statistics import mode
print('Embarked Mode: ' +  str(mode(train['Embarked'])))


# In[ ]:


print('Fare Mean: ' +  str(np.mean(train['Fare'])))


# In[ ]:


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

for dataset in full_data:
    # Mapping Sex
    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(float)
    
    # Mapping titles
    title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(float)
    dataset['Embarked'] = dataset['Embarked'].fillna(0)
    
    # Mapping Fare
    dataset['Fare'] = dataset['Fare'].fillna(32.20)
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(float).dropna(axis=0, how='any')
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] ;
    
    


# In[ ]:


train.isnull().sum()


# More than half of the <b>Cabin</b> data is missing, so we are going to drop it from our model, along with variables that no longer contain relevant information.

# In[ ]:


# Feature selection: remove variables no longer containing relevant information
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)


# ### Visualizing Processed Data

# In[ ]:


train.head(10)


# In[ ]:


colormap = plt.cm.plasma
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# This heatmap is very useful as an initial observation because you can easily get an idea of the predictive value of each feature. In this case, Sex and Title show the highest correlations (in absolute terms) with the class (Survived): 0.54 and 0.49 respectively. But the absolute correlation between both is also very high (0.86, the highest in our dataset), so they are probably carrying the same information and using the two as inputs for the same model wouldn't be a good idea. High chances are one of them will be used for the first node in our final decision tree, so let's first explore further these features and compare them.

# ### Title vs Sex
# You can easily compare features and their relationship with the class by grouping them and calculating some basic statistics for each group. The code below does exactly this in one line, and explains the meaning of each metric when working with a binary class.

# In[ ]:


train[['Title', 'Survived']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])
# Since "Survived" is a binary class (0 or 1), these metrics grouped by the Title feature represent:
    # MEAN: survival rate
    # COUNT: total observations
    # SUM: people survived

# title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5} 


# In[ ]:


xlab = ['N/A', 'Mr', 'Ms', 'Mrs', 'Master', 'Rare']
dp = sns.distplot(train['Title'], color='Red')
dp.set(xticks=range(0,6), xticklabels=xlab)
#dp.set_xticks([0,1,2,3,4,5,6])
#dp.set_xticklabels(xlab)
plt.show()


# In[ ]:


# normality test
stat, p = shapiro(train['Title'])
print('Shapiro-Wilk Statistic=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[ ]:


# normality test
stat, p = shapiro(train['Sex'])
print('Shapiro-Wilk Statistic=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')


# In[ ]:


train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).agg(['mean', 'count', 'sum'])
# Since Survived is a binary feature, this metrics grouped by the Sex feature represent:
    # MEAN: survival rate
    # COUNT: total observations
    # SUM: people survived
    
# sex_mapping = {{'female': 0, 'male': 1}} 


# The data shows that less 'Mr' survived (15.67%) than men in general (18.89%): Title seems therefore to be more useful than Sex for our purpose. This may be because Title implicitly includes information about Sex in most cases. To verify this, we can use the copy we made of the original training data without mappings and check the distribution of Sex grouped by Title.

# In[ ]:


# Let's use our 'original_train' dataframe to check the sex distribution for each title.
# We use copy() again to prevent modifications in out original_train dataset
title_and_sex = original_train.copy()[['Name', 'Sex']]

# Create 'Title' feature
title_and_sex['Title'] = title_and_sex['Name'].apply(get_title)

# Map 'Sex' as binary feature
title_and_sex['Sex'] = title_and_sex['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Table with 'Sex' distribution grouped by 'Title'
title_and_sex[['Title', 'Sex']].groupby(['Title'], as_index=False).agg(['mean', 'count', 'sum'])

# Since Sex is a binary feature, this metrics grouped by the Title feature represent:
    # MEAN: percentage of men
    # COUNT: total observations
    # SUM: number of men


# We find that, excepting for a single observation (a female with 'Dr' title), all the observations for a given Title share the same Sex. Therefore the feature Title is capturing all the information present in Sex. In addition, Title may be more valuable to our task by capturing other characteristics of the individuals like age, social class, personality, ...
# 
# It's true that by regrouping rare titles into a single category, we are losing some information regarding Sex. We could create two categories "Rare Male" and "Rare Female", but the separation will be almost meaningless due to the low occurrence of "Rare" Titles (2.6%, 23 out of 891 samples).
# 
# Thanks to this in-depth analysis of the Sex and Title features we've seen that, even if the correlation of the feature Sex with the class Survived was higher, Title is a richer feature because it carries the Sex information but also adds other characteristics. Therefore is very likely that Title is going to be the first feature in our final decision tree, making Sex useless after this initial split.

# ### Gini Impurity
# Before start working with Decision Trees, let's briefly explain how they work. The goal of their learning algorithms is always to find the best split for each node of the tree. But measuring the "goodness" of a given split is a subjective question so, in practice, different metrics are used for evaluating splits. One commonly used metric is Information Gain. The sklearn library we're gonna use implements Gini Impurity, another common measure, so let’s explain it.
# 
# Gini Impurity measures the disorder of a set of elements. It is calculated as the probability of mislabelling an element assuming that the element is randomly labelled according to the distribution of all the classes in the set. Decision Trees will try to find the split which decreases Gini Impurity the most across the two resulting nodes. For the titanic example it can be calculated as follows (code should be explicit enough):

# In[ ]:


def get_gini_impurity(survived_count, total_count):
    survival_prob = survived_count/total_count
    not_survival_prob = (1 - survival_prob)
    random_observation_survived_prob = survival_prob
    random_observation_not_survived_prob = (1 - random_observation_survived_prob)
    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob
    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob
    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob
    return gini_impurity


# Let's use our Sex and Title features as an example and calculate how much each split will decrease the overall weighted Gini Impurity. First, we need to calculate the Gini Impurity of the starting node including all 891 observations in our train dataset. Since only 342 observations survived, the survival probability is around 38,38% (342/891).

# In[ ]:


# Gini Impurity of starting node
gini_impurity_starting_node = get_gini_impurity(342, 891)
gini_impurity_starting_node


# We're now going to simulate both splits, calculate the impurity of resulting nodes and then obtain the weighted Gini Impurity after the split to measure how much each split has actually reduced impurity.
# 
# If we split by Sex, we'll have the two following nodes:
# 
# --Node with men: 577 observations with only 109 survived
# 
# --Node with women: 314 observations with 233 survived

# In[ ]:


# Gini Impurity decrease of node for 'male' observations
gini_impurity_men = get_gini_impurity(109, 577)
gini_impurity_men


# In[ ]:


# Gini Impurity decrease if node split for 'female' observations
gini_impurity_women = get_gini_impurity(233, 314)
gini_impurity_women


# In[ ]:


# Gini Impurity decrease if node split by Sex
men_weight = 577/891
women_weight = 314/891
weighted_gini_impurity_sex_split = (gini_impurity_men * men_weight) + (gini_impurity_women * women_weight)

sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_starting_node
sex_gini_decrease


# If we split by Title == 1 (== Mr), we'll have the two following nodes:
# 
# --Node with only Mr: 517 observations with only 81 survived
# 
# --Node with other titles: 374 observations with 261 survived

# In[ ]:


# Gini Impurity decrease of node for observations with Title == 1 == Mr
gini_impurity_title_1 = get_gini_impurity(81, 517)
gini_impurity_title_1


# In[ ]:


# Gini Impurity decrease if node split for observations with Title != 1 != Mr
gini_impurity_title_others = get_gini_impurity(261, 374)
gini_impurity_title_others


# In[ ]:


# Gini Impurity decrease if node split for observations with Title == 1 == Mr
title_1_weight = 517/891
title_others_weight = 374/891
weighted_gini_impurity_title_split = (gini_impurity_title_1 * title_1_weight) + (gini_impurity_title_others * title_others_weight)

title_gini_decrease = weighted_gini_impurity_title_split - gini_impurity_starting_node
title_gini_decrease


# We find that the Title feature is slightly better at reducing the Gini Impurity than Sex. This confirms our previous analysis, and we're now sure that Title will be used for the first split. Sex will therefore be neglected since the information is already included in the Title feature.
# 
# [For more on how decision trees work...](https://www.kaggle.com/c/titanic/discussion/10169)

# Rather than use the age feature as defined by the author of the original notebook, let's drop it and use the quantile variable that we created earlier. We will compare the results of this model with those of the model in the original notebook at the end.

# In[ ]:


del train['Age']
del test['Age']


# In[ ]:


try:
    train['Age']
except KeyError:
    var_exists = False
else:
    var_exists = True
    
print('Age feature exists in train: ' + str(var_exists))

try:
    test['Age']
except KeyError:
    var_exists = False
else:
    var_exists = True
    
print('Age feature exists in test: ' + str(var_exists))


# ### Finding the best tree depth with the help of cross-validation
# 
# After exploring the data, we're going to find of much of it can be relevant for our decision tree. This is a critical point for every Data Science project, since too much train data can easily result in bad model generalisation (accuracy on test/real/unseen observations). Over-fitting (a model excessively adapted to the train data) is a common reason. In other cases, too much data can also hide meaningful relationships either because they evolve with time or because highly correlated features prevent the model from capturing properly the value of each single one.
# 
# In the case of decision trees, the 'max_depth' parameter determines the maximum number of attributes the model is going to use for each prediction (up to the number of available features in the dataset). A good way to find the best value for this parameter is just iterating through all the possible depths and measure the accuracy with a robust method such as Cross Validation.
# 
# Cross Validation is a model validation technique that splits the training dataset in a given number of "folds". Each split uses different data for training and testing purposes, allowing the model to be trained and tested with different data each time. This allows the algorithm to be trained and tested with all available data across all folds, avoiding any splitting bias and giving a good idea of the generalisation of the chosen model. The main downside is that Cross Validation requires the model to be trained for each fold, so the computational cost can be very high for complex models or huge datasets.

# In[ ]:


cv = KFold(n_splits=10)            # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1, max_attributes + 1)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    # print("Current max depth: ", depth, "\n")
    for train_fold, valid_fold in cv.split(train):
        f_train = train.loc[train_fold] # Extract train data with cv indices
        f_valid = train.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 
                               y = f_train["Survived"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 
                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)
    # print("Accuracy per fold: ", fold_accuracy, "\n")
    # print("Average accuracy: ", avg)
    # print("\n")
    
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))


# The best max_depth parameter seems therefore to be 3 (80.5% average accuracy across the 10 folds), and feeding the model with more data results in worse results probably due to over-fitting. We'll therefore use 3 as the max_depth parameter for our final model.

# In[ ]:


# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
y_train = train['Survived']
x_train = train.drop(['Survived'], axis=1).values 
x_test = test.values

# Create Decision Tree with max_depth = 3
decision_tree = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree.fit(x_train, y_train)

# Predicting results for test dataset
y_pred = decision_tree.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)


#make sure you have graphviz executables in your system path
from graphviz import Source
# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = Source(tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(train.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True ))
     
# Convert .dot to .png to allow display in web notebook
check_call(['dot','-Tpng','tree1.dot','-o','tree1.png'])

# Annotating chart with PIL
img = Image.open("tree1.png")
draw = ImageDraw.Draw(img)
font = ImageFont.truetype('/usr/share/fonts/truetype/droid/DroidSans.ttf', 26)
draw.text((10, 0), # Drawing offset (position)
          '"Title <= 1.5" corresponds to "Mr." title', # Text to draw
          (0,0,255), # RGB desired color
          font=font) # ImageFont object with desired font
img.save('sample-out.png')
PImage('sample-out.png')


# In[ ]:


acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# Maybe if we had gone a few more levels down we would see the impact of the age quantile feature, but our test for best max depth parameter suggest we should leave it at 3. Slightly less accurate than the 82.38 from the tree in the original notebook, but now I have my first submission ready!
