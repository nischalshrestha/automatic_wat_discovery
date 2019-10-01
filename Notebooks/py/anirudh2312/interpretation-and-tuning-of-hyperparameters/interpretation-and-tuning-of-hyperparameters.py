#!/usr/bin/env python
# coding: utf-8

# # Introduction: Titanic- Machine Learning from Disaster
# 
# This notebook is intended for beginners who want to take a tour of how to approach a machine learning/data science problem, starting from observing and analyzing the data, infering some not so directly visible information in the data, correcting the given data, exporting and extracting out some new and useful information from the data (feature engineering), dealing with missing values in the data, pre-processing the data to make it fit for feeding into the ML algorithms, building the ML models, interpreting the role of various parameters in the ML algorithms, improving the performance of the ML models by tuning the parameters and applying the feature selection techniques. Finally, this notebook will analyse, why some of the better performing algorithms are not a good fit for this problem/dataset.      
# 
# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class. The challenge is to complete the analysis of what sorts of people were likely to survive. 
# 
#  This is a standard supervised classification task:
# - __Supervised:__ The labels are included in the training data and the goal is to train a model to learn to predict the labels from the features
# - __Classification:__ The label is a binary variable, 0 (not survived the tragedy), 1 (survived the tragedy)
# 
# The __evaluation metric__ for this problem is 'accuracy', i.e. how many of our predictions are correct.
# 
# __Note__: This is my first ever notebook in terms of a machine learning project. So, I am learning with this work, and if anybody has any suggestions or feedback with respect to this notebook, then please comment. And, if you like the work, then please upvote.  
# 
# 

# ## Data
# The data is taken from the kaggle's titanic competition website. The data contains features and labels-
# 
# - __Features:__ Passenger Id, Passenger Class, Passenger Name, Passenger Gender, Passenger Age, No. of Siblings, spouse, parent and children related to passenger on-board, Passenger Ticket, Ticket's Fare, Passenger Cabin, Embarked (location on the ship)
# - __Label:__ Survived
# 

# ## Imports

# In[ ]:


# for data manipulation
import pandas as pd
import numpy as np

# for plotting/visualising the distibution of data
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly import tools

import random
import re

# for pre-processing of the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import warnings


# ## Read in data

# In[ ]:


# load the data
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


train_df.head()


# In[ ]:


# Get the distribution of the data
train_df.describe(include='all')


# # Explorartory Data Analysis
# Exploratory Data Analysis (EDA) is an open-ended process where we calculate statistics and make figures to find trends, anomalies, patterns, or relationships within the data. The goal of EDA is to learn what our data can tell us. It generally starts out with a high level overview, then narrows in to specific areas as we find intriguing areas of the data. The findings may be interesting in their own right, or they can be used to inform our modeling choices, such as by helping us decide which features to use.

# ## Examine the Distribution of the 'Survived' Column
# 
# The survival is what we are asked to predict: either a 0 for the person who will not survive, or a 1 indicating the person will survive the tragedy. We can first examine the number of passengers falling into each category.

# In[ ]:


train_df['Survived'].astype(int).plot.hist();


# From this information, we can consider this problem as a balanced class problem.

# ## Examine Missing Values
# 
# Next we can look at the number and percentage of missing values in each column. 

# In[ ]:


# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


# Missing values statistics
missing_values = missing_values_table(train_df)
missing_values


# When it comes to building our machine learning models, we will have to fill in these missing values (known as imputation). Also, there are some models available such as LightGBM, XGBoost that can [handle missing values with no need for imputation](https://stats.stackexchange.com/questions/235489/xgboost-can-handle-missing-data-in-the-forecasting-phase). Another option would be to drop columns with a high percentage of missing values, which do not have high feature importance or high correlation with the labels.

# Since, Embarked column has very few number of missing values, only 2, we can impute it with the mode of the column. It will not affect the overall distribution of the column. 

# In[ ]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)


# ## Feature Engineering
#   

# Passenger Name is a categorical variable and it is obviously unique for each passenger. So we cannot use the Name variable directly in our model but we can extract some useful information from this variable like a new feature called 'Title' which can guide our model. 

# In[ ]:


# Define function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""
# Create a new feature Title, containing the titles of passenger names
train_df['Title'] = train_df['Name'].apply(get_title)
test_df['Title'] = test_df['Name'].apply(get_title)


# In[ ]:


train_df['Title'].value_counts()


# In[ ]:


test_df['Title'].value_counts()


# From this information, we see that some titles are very rare, so we will combine these titles under a single name - 'Others'

# In[ ]:


dict1 = {'Dr':'Others', 'Rev':'Others', 'Col':'Others', 'Mlle':'Others', 'Major':'Others', 'Capt':'Others', 'Ms':'Others', 
         'Don':'Others', 'Lady':'Others', 'Countess':'Others', 'Jonkheer':'Others', 'Mme':'Others', 'Sir':'Others'}
train_df['Title'] = train_df['Title'].replace(dict1)

dict2 = {'Col':'Others', 'Rev':'Others', 'Ms':'Others', 'Dona':'Others', 'Dr':'Others'}
test_df['Title'] = test_df['Title'].replace(dict2)


# In[ ]:


train_df['Title'].value_counts()


# From this information of titles, we can see that there are 5 classes. Leaving out the others, 'Miss' and 'Mrs' titles are used for females and the only information that we can get from these 2 titles is that there are 125 females (Mrs) who are married and 182 females (Miss) are not married. 'Mr' and 'Master' titles are used for males. Master is used for children (people with age less than 13). This means that there are 40 male children (age less than 13) and 517 males with age greater than 13.    

# ## Correction in the data
# If there is any male with age less than 13 and having a title of Mr, then that is incorrect entry and we need to correct the title, changing it to Master. Although, there is only one incorrect entry in our data of this type and changing it will not make much impact, but Data correction is an important step in building a machine learning model.  

# In[ ]:


# Checking the incorrect entry when age is less than 13 for male and the title is Mr.
df = train_df.loc[train_df['Title']=='Mr']
df = df.loc[df['Age']<13]
df.head()


# In[ ]:


# Correcting the entry
train_df.loc[[731],['Title']] = 'Master'


# ## Visualise the variables by plotting the classes on a graph

# In[ ]:


# Function to plot the classes of the variables
def random_color_generator(number_of_colors):
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color

def get_percent(df, temp_col, width=800, height=500):
    cnt_srs = df[[temp_col, 'Survived']].groupby([temp_col], as_index=False).mean().sort_values(by=temp_col)

    trace = go.Bar(
        x = cnt_srs[temp_col].values[::-1],
        y = cnt_srs['Survived'].values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        name = "Percent",
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace

def get_count(df, temp_col, width=800, height=500):
    cnt_srs = df[temp_col].value_counts().sort_index()

    trace = go.Bar(
        x = cnt_srs.index[::-1],
        y = cnt_srs.values[::-1],
        text = cnt_srs.values[::-1],
        textposition = 'auto',
        textfont = dict(
            size=12,
            color='rgb(0, 0, 0)'
        ),
        name = 'Count',
        orientation = 'v',
            marker = dict(
                color = random_color_generator(100),
                line=dict(color='rgb(8,48,107)',
                  width=1.5,)
            ),
            opacity = 0.7,
    )    
    return trace

def plot_count_percent_for_object(df, temp_col, height=400):
    trace1 = get_count(df, temp_col)
    trace2 = get_percent(df, temp_col)

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Count', 'Percent'), print_grid=False)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout']['yaxis1'].update(title='Count')
    fig['layout']['yaxis2'].update(range=[0, 1], title='% Survived')
    fig['layout'].update(title = temp_col, margin=dict(l=100), width=800, height=height, showlegend=False)

    py.iplot(fig)


# In[ ]:


# observe the distribution of title
warnings.simplefilter('ignore')
temp_col = train_df.columns.values[12]
plot_count_percent_for_object(train_df, temp_col)


# From this information, we can observe that Age (Mr vs Master), Family/relationship status (Mrs vs Miss), Gender/Sex (Mr vs Mrs) variables play reasonable role in the survival of the passenger.

# In[ ]:


# observe the distribution of Sex
temp_col = train_df.columns.values[4]
plot_count_percent_for_object(train_df, temp_col)


# There are 2 features in our dataset- 'SibSp' gives the information about the sibling or spouse of the passenger onboard and 'Parch' gives information about the parents and children of the passenger onboard. But both these variables basically indicate the family information of passenger onboard. So we will combine these 2 variables into one variable 'family'.  

# In[ ]:


# Making a new variable/feature 'family'
train_df['family'] = train_df['SibSp'] + train_df['Parch'] + 1
train_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)

test_df['family'] = test_df['SibSp'] + test_df['Parch'] + 1
test_df.drop(['SibSp', 'Parch'], axis=1, inplace=True)


# In[ ]:


# observe the distribution of family
temp_col = train_df.columns.values[11]
plot_count_percent_for_object(train_df, temp_col)


# From this information, we can see that there are many passengers who do not have any family member onboard. This directs us to make another variable called 'family_status' which will tell us whether the passenger was alone on the ship or with some family member (not alone).  

# In[ ]:


# Making a new variable/feature 'family_status' from the variable 'family' 
train_df['family_status'] = train_df['family']
test_df['family_status'] = test_df['family']
dict2 = {1:'Alone', 2:'NotAlone', 3:'NotAlone', 4:'NotAlone', 5:'NotAlone', 6:'NotAlone', 7:'NotAlone', 8:'NotAlone', 
         9:'NotAlone', 10:'NotAlone', 11:'NotAlone'}
train_df['family_status'] = train_df['family_status'].replace(dict2)
test_df['family_status'] = test_df['family_status'].replace(dict2)


# In[ ]:


train_df['family_status'].dtype


# In[ ]:


# observe the distribution of family_status
temp_col = train_df.columns.values[12]
plot_count_percent_for_object(train_df, temp_col)


# Now let's observe the distribution of Age. Age is a continuous variable with float data type. So, let's first plot the KDE (kernel distribution estimation) plot for age.   

# In[ ]:


plt.figure(figsize = (10, 8))

df = train_df[['Survived', 'Age']]
df = df.dropna()

# KDE plot of passengers who did not survive 
sns.kdeplot(df.loc[df['Survived'] == 0, 'Age'], label = 'survived == 0')

# KDE plot of passengers who survived
sns.kdeplot(df.loc[df['Survived'] == 1, 'Age'], label = 'survived == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Age');


# Although this graph is stating that children with age between 0-10 have more survival to non-survival ratio, this graph is not helping much in getting some useful information. We will examine this variable by dividing into different age groups- child, young and old. And for the null values in the age variable, we will make 'missing' class. 

# In[ ]:


# dividing the age variable into different classes
train_df['Age'] = train_df['Age'].fillna(200) # this is just indicating the missing values
train_df['Age'] = pd.cut(train_df['Age'], bins=[0,12,40,80, 250], labels = ['Child', 'Young', 'Old', 'Missing'])


# In[ ]:


train_df['Age'] = train_df['Age'].astype('O')


# In[ ]:


temp_col = train_df.columns.values[5]
plot_count_percent_for_object(train_df, temp_col)


# Now this plot gives us some useful information such as children are more likely to survive, followed by young and old. Maybe because there were limited lifeboats and they tried to save all the children first by sending them on the lifeboats with some adults. But Age variable has many missing entries (177 to be precise) and we need to impute them.   

# In[ ]:


age_train = pd.read_csv('../input/train.csv')
age_test = pd.read_csv('../input/test.csv')
train_df['Age'] = age_train['Age']
test_df['Age'] = age_test['Age']

# mean of the age variable
age_avg_train = train_df['Age'].mean()
age_avg_test = test_df['Age'].mean()
# standard deviation of the age variable 
age_std_train = train_df['Age'].std()
age_std_test = test_df['Age'].std()

age_null_count_train = train_df['Age'].isnull().sum()
age_null_count_test = test_df['Age'].isnull().sum()

# list of the random age values to be filled based on the distribution of the original age variable  
age_null_random_list_train = np.random.randint(age_avg_train - age_std_train, age_avg_train + age_std_train, size=age_null_count_train)
age_null_random_list_test = np.random.randint(age_avg_test - age_std_test, age_avg_test + age_std_test, size=age_null_count_test)

train_df['Age'][np.isnan(train_df['Age'])] = age_null_random_list_train
test_df['Age'][np.isnan(test_df['Age'])] = age_null_random_list_test


# Now, since we have imputed the missing age values, again divide the age variable into three groups- child, young and old.

# In[ ]:


# dividing the age variable into different classes
train_df['Age'] = pd.cut(train_df['Age'], bins=[0,12,40,80], labels = ['Child', 'Young', 'Old'])
test_df['Age'] = pd.cut(test_df['Age'], bins=[0,12,40,80], labels = ['Child', 'Young', 'Old'])


# In[ ]:


train_df['Age'] = train_df['Age'].astype('O')


# In[ ]:


temp_col = train_df.columns.values[5]
plot_count_percent_for_object(train_df, temp_col)


# We may infer from this plot that since there were limited lifeboats, they preferred children and old people in sending them on the lifeboats.

# In[ ]:


train_df.head()


# Passenger Id and Ticket are just random variables and they do not give any intuition for their relation with the chances of survival of a passenger. So, we would like to drop these variables. Cabin variable has more than 75% missing data, so we would want to drop this variable. Also, we have already extracted out the useful information from the Name variable, so we would drop this variable as well.  

# In[ ]:


drop = ['Name', 'PassengerId', 'Ticket', 'Cabin']
train_df = train_df.drop(drop, axis=1)
test_df = test_df.drop(drop, axis=1)


# ## Encode the categorical variable
# A machine learning model unfortunately cannot deal with categorical variables, except for some models such as LightGBM. Hence, we need to find a way to encode these variables as numbers before feeding them to our model. There are many types of encoding techniques but the two main techniques are:
# 
# - Label Encoding: assign each unique class in a categorical variable with an integer, randomly. No new columns are created
# - One Hot Encoding: create a new column for each unique class in a categorical variable. Each entry recieves a 1 in the column for its corresponding class and a 0 in all other new columns
# 
# The drawback of the label encoding is that it assigns arbitrary ordering to the classes of the categorical variables which do not reflect the inherent properties/information of that variable and label encoding may be misleading in this case. When we have only 2 unique classes (like male/female) for a categorical variable or if we have more than 2 unique classes and we already know the relative ordering of the classes (like, low/medium/high), then we can go with label encoding but here also, we need to ensure that the integer values assigned to these classes align with the relative order of the classes (like, low=0, medium=1 and high=2), otherwise one-hot encoding is the safe option.
# 
# The drawback of the one-hot encoding is that the number of new columns/features (dimensions of the data) can explode with the categorical variable having many unique classes because there will be a new column for each unique class in the variable. To deal with the feature exploding problem, we can use dimensionality reduction techniques like, Principal Component Analysis (PCA) after one-hot encoding.   

# ### Label Encoding 
# In this notebook, we will implement label encoding for any categorical variable which have 2 unique classes

# In[ ]:


# Create a label encoder object
encoder = LabelEncoder()
encoder_count = 0

# Iterate through the columns
for col in train_df:
    if train_df[col].dtype == 'object':
        # If 2 unique classes
        if len(list(train_df[col].unique())) <= 2:
            encoder.fit(train_df[col])
            train_df[col] = encoder.transform(train_df[col])
            test_df[col] = encoder.transform(test_df[col])
            # Keep track of how many columns were label encoded
            encoder_count += 1
            
print('%d columns were label encoded.' % encoder_count)


# Since, sex and family_status are 2 categorical variables with 2 unique classes, they are label encoded. We can also implement label encoding with the Age variable even if it has more than 2 unique classes, because we know the relative ordering of the classes, age_old > age_young > age_child 

# In[ ]:


dict1 = {'Child':1, 'Young':2, 'Old':3}
train_df['Age'] = train_df['Age'].replace(dict1)
test_df['Age'] = test_df['Age'].replace(dict1)


# ### One-Hot Encoding 

# In[ ]:


# one-hot encoding of categorical variables
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train = train_df
test = test_df

print(train.shape)
print(test.shape)


# In[ ]:


y = train['Survived']
x = train.drop('Survived', axis=1)


# Data preprocessing is complete. Here x represents features and y represents labels.

# # Build the Machine Learning Model
# The problem that we are solving is a binary classification problem and since we are given the target variable or the labels, it is a supervised learning. There is No Free Lunch (NFL) theorem in ,achine learning which states that no algorithm is the best for the generic case and all special cases. So, to build an efficient model, we need to compare the performance of various machine learning classification algorithms. But since we have identified our solution as a supervised learning classification algorithm, we can narrow down our list of choices.
# 
# __Machine Learning Classificaton Algorithms:__
# - Generalized Linear Models (GLM)
# - Support Vector Machines (SVM)
# - Discriminant Analysis
# - Nearest Neighbors
# - Naive Bayes
# - Decision Trees
# - Ensemble Methods

# ## Imports

# In[ ]:


from sklearn.model_selection import cross_validate, ShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn import ensemble, naive_bayes, svm, tree, discriminant_analysis, neighbors, feature_selection


# First, run the algorithms with the default parameters to get an idea about their performances on our data and then we would tune the parameters of better performing algorithms to improve their performances.

# In[ ]:


MLA = [    
        # Generalized Linear Models
        LogisticRegressionCV(),
    
        # SVM
        svm.SVC(probability = True),
        svm.LinearSVC(),
    
        # KNN
        neighbors.KNeighborsClassifier(weights='distance'),
    
        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
     
        # Naive Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
    
        #Trees    
        tree.DecisionTreeClassifier(),
    
        # Ensemble Methods
        ensemble.AdaBoostClassifier(),
        ensemble.BaggingClassifier(),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier()
     
    ]

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0)
MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean','MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    cv_results = cross_validate(alg, x,y, cv  = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
        
    row_index+=1
   

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)


# In[ ]:


MLA_compare


# From the above results, it can be seen that __Gradient Boosting Classifier__ has the highest test accuracy but it has a lot of __overfitting__ since its training accuracy is around 90%, while its test accuracy is only around 83.13%. So, this algorithm cannot be trusted as it is for our model in terms of its generalization ability. So, we will tune some of the better performing algorithms by searching for the best parameters for them.   

# ## Tuning the Model with Hyper-Parameters
# We will tune our model using Parameter Grid and Grid Search CV for different algorithms.

# ### Grid Search for SVM
# There are 3 important parameters when it comes to tuning with svm - __C__, __kernel__, and __gamma__. 
# 
# C represents the penalty parameter, larger the value of C, the more we are penalising the violations/misclassifications. So, there is a tradeoff with respect to C, the larger we make C, the smaller will the margin be but we will be getting more of the training data correct. So, if we make C very large to get most of the training data correct, then we may compromise with the generalisation property (robustness) of the model. In some cases, even when the data is truly linearly separable, we would like to tradeoff a small C for greater margins to make our model robust, especially in case of noisy data.
# 
# When our data is not linear in the given dimensions, then to make our classifier more powerful, we do basis transformation. Kernel trick does the same thing for us, it takes the data in the given dimensions and transform it to some higher dimensions to make it linear and then applies a linear classification. This facility is provided by the kernel parameter in the SVM algorithm. 
# 
# Gamma is a parameter which is associated with rbf or poly kernel and deals with the measure of complexity of the model. Small gamma means less complexity and large gamma means more complexity and very large gamma may eventually lead to overfitting.

# In[ ]:


# grid search for svm
classifier = svm.SVC()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

param_grid = {'C':[0.5,1.0,2.0, 3.0],  # penalty parameter C of the error term
              'kernel':['linear', 'rbf'], # specifies the kernel type to be used in the algorithm  
              'gamma':[0.02, 0.08,0.2,1.0] # kernel coefficient for 'rbf'
             }

# Grid Search
tune_model = GridSearchCV(svm.SVC(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    

print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)


# ### Grid Search for Decision Trees
# The parameters important for tuning are:
# - 'criterion': __gini or entropy__ -  measure of impurity (can be treated as a loss function)
# - 'splitter': __best or random__ - methodology to select the next feature to split on
# - 'max_depth': maximum height to restrict the growth of the tree beyond certain point (to avoid overfitting sometimes)
# - 'min_samples_split': minimum number of samples at a node required to split 
# 
# Decision tree grows by splitting the data based on the selected feature/variable (nodes in terms of trees), but a big question arises while implementing the trees that when should we stop. One answer to this question is Early Stopping. The tree grows in such a manner so as to minimize the loss function or the error by recursively splitting the variables, but at some point, the improvement in the error is not significant, then we can stop. This stopping is called as early stopping. 
# 
# Early stopping might not be a good idea in some situations. For example, let's say there are two variables to split on - x1 and x2. Consider all the possible ways where we can split x1. Suppose, we are not getting any improvement in the error by splitting on x1. Now, consider all the possible ways where we can split x2. Again we are not getting any error improvement. At this point, early stopping may stop this recursive algorithm and give the final results. But what if we can get error improvement by first splitting on x1 and then on x2. Early stopping may miss out on these interactions of variables such as in this case- first split on x1 and then on x2.
# 
# To avoid the problem created by early stopping, there is a second strategy to stop the tree algorithm, which is stop the splitting when the leaves are small. This means that when the number of data points in the region defined by the leaf of the tree are very small, then stop the further splitting of that region. This small number we can define by ourselves with the help of __min_samples_leaf__ parameter and we can also tune this parameter. 

# In[ ]:


# grid search for decision trees
dtree = tree.DecisionTreeClassifier(random_state = 0)
base_results_dtree = cross_validate(dtree, x, y, cv  = cv_split, return_train_score=True)
dtree.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results_dtree['train_score'], base_results_dtree['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', dtree.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results_dtree['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results_dtree['test_score'].mean()*100))
print('-'*10)

param_grid = {'criterion': ['gini','entropy'], 
              'splitter': ['best', 'random'], 
              'max_depth': [2,4,6,8,10,None], 
              #'min_samples_split': [2,5,7,10,12], 
              #'min_samples_leaf': [1,3,5,7, 10], 
              'random_state': [0] 
             }


tune_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    

print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)


# ### Check the Feature Importances returned by the Tuned Decision Tree

# In[ ]:


# train the model using tuned decision tree parameters
dtree = tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, random_state= 0, splitter= 'random')
base_results = cross_validate(dtree, x, y, cv  = cv_split, return_train_score=True)
dtree.fit(x, y)


# In[ ]:


def plot_feature_importances(df):
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
    
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()

    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), 
            df['importance_normalized'].head(15), 
            align = 'center', edgecolor = 'k')
    
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot labeling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df


# In[ ]:


importance = dtree.feature_importances_
feature = x.columns
fi = pd.DataFrame()
fi['importance'] = importance
fi['feature'] = feature
fi_sorted = plot_feature_importances(fi)


# ### Grid Search for Bagging Classifier
# The idea behind bagging is to create multiple training sets from the given dataset by bootstrapping, train a weak classifier on each of those sets abd then combine the output of each classifier either by majority vote (in case of 0/1) or by weighted average of probability. While tuning the parameters for bagging classifier, we will tune the base_estimator or the weak classifier that we want to bag and the number of such weak classifiers in the bag.  

# In[ ]:


# grid search for bagging classifier
classifier = ensemble.BaggingClassifier()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

cl1 = LogisticRegressionCV()
cl2 = tree.DecisionTreeClassifier()
cl3 = svm.LinearSVC()
cl4 = discriminant_analysis.LinearDiscriminantAnalysis()
cl5 = discriminant_analysis.QuadraticDiscriminantAnalysis()
param_grid = {'base_estimator':[cl1, cl2, cl3, cl4, cl5],
              'n_estimators':[10,13,17],
              #'warm_start':[False, True]
             }


tune_model = GridSearchCV(ensemble.BaggingClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)



# In[ ]:


# printing the results of bagging before and after tuning
epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    


print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)


# ### Grid Search for Adaboost
# The idea behind boosting is at every stage, try to look at previous stage and see which are the data points that were misclassified and try to get them correct in this stage. It is fine to make mistakes in this stage on the data points that have been correctly classified upto the previous stage because the previous classifiers can possibly adjust for that. Adaboost classifier is a boosting technique which is based on exponential loss function.
# 
# In the case of bagging, we create multiple datasets and train a weak classifier on each of those sets parallely. So, Bagging classifier is inherently parallel and that is why it is fast as well. 
# 
# In the case of boosting, first we assign weights to the data points. The data points that were misclassified in the previous stage will get higher weights in the current stage and those which were correctly classified will get lower weights. Then sample from those data points according to their weights. We create a training set by sampling from the data points given to us according to their weights. So, points for which weights are higher will get sampled more often and points for which weights are very low might not even appear in the dataset. So, if the point appears multiple times in the dataset and then we are trying to minimize the training error, we are likely to get that point correct. 
# 
# Since we are assigning weights to the data points at every stage according to the errors made in the previous stage, this makes boosting algorithm inherently serial, unlike bagging and that is why boosting is slow as well, in general.   
# 
# So, while tuning the adaboost classifier, we will tune the base estimators. Since adaboost requires sample weighting, those estimators which do not support sample_weights method cannot be used as base estimators, like LDA.  

# In[ ]:


classifier = ensemble.AdaBoostClassifier()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

cl1 = LogisticRegressionCV()
cl2 = tree.DecisionTreeClassifier()
cl3 = naive_bayes.GaussianNB()
param_grid = {'base_estimator':[cl1, cl2, cl3]
             }


tune_model = GridSearchCV(ensemble.AdaBoostClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    


print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)


# ### Grid Search for Random Forest
# Random forest is an improved version of bagging when we use decision trees as a base estimator. The goal of random forest is to reduce the correlation among trees. The idea is to start doing bagging as we normally do (create multiple training sets by bootstrapping). Then, start building trees on these datasets. So, in random forest what we do is at every node, randomly sample 't' features from 'p' feature set (t<p). Find out which is the best split variable among these 't' features alone and split on that. Generally, t=sqrt(p) or log(p) but we can select any value with the max_features parameter. 
# 
# If we had just done bagging, at the root level, it is highly likely that each one of the bagged tree would have picked the same attribute. So, at the higher levels of the tree, it will look very similar. But in random forest, we are getting rid of that. This leads to significant reduction in variance in the bagged estimate, especially when we have small feature set. 
# 
# Tuning random forest is just like tuning decision trees, except here, since we are bagging lot of decision trees, we need to tune the no. of estimators parameter. With small dataset and small feature set, large number of estimators can sometimes lead to overfitting.   

# In[ ]:


classifier = ensemble.RandomForestClassifier()
base_results = cross_validate(classifier, x, y, cv  = cv_split, return_train_score=True)
classifier.fit(x, y)

epoch=0
for train_score,test_score in zip(base_results['train_score'], base_results['test_score']):
        epoch +=1       
        print("epoch:",epoch,"train_score:",train_score, "test_score:",test_score)
print('-'*10)

print('BEFORE Tuning Parameters: ', classifier.get_params())
print("BEFORE Tuning Training w/bin score mean: {:.2f}". format(base_results['train_score'].mean()*100)) 
print("BEFORE Tuning Test w/bin score mean: {:.2f}". format(base_results['test_score'].mean()*100))
print('-'*10)

param_grid = {'n_estimators': [15,25,30,35],
              'criterion': ['gini','entropy'],  #scoring methodology; two supported formulas for calculating information gain - default is gini
              'max_depth': [2,4,6,None], #max depth tree can grow; default is none
              'min_samples_split': [2,5,7,10,12], #minimum subset size BEFORE new split (fraction is % of total); default is 2
              #'min_samples_leaf': [1,3,5], #minimum subset size AFTER new split split (fraction is % of total); default is 1
              'max_features': [2,3,'auto'], #max features to consider when performing split; default none or all
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }


tune_model = GridSearchCV(ensemble.RandomForestClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split, return_train_score=True)
tune_model.fit(x, y)

for i in range(10):
    print("epoch:",i,"train_score:",tune_model.cv_results_['split'+str(i)+'_train_score'][tune_model.best_index_],
    "test_score:",tune_model.cv_results_['split'+str(i)+'_test_score'][tune_model.best_index_])

print('-'*10)    


print('AFTER Tuning Parameters: ', tune_model.best_params_)
print("AFTER Tuning Training w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100))
print("AFTER Tuning Test w/bin score mean: {:.2f}". format(tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)


# ### Check the Feature Importances returned by the Random Forest

# In[ ]:


# train the model using tuned random forest parameters
random_forest = ensemble.RandomForestClassifier(criterion= 'entropy', max_depth= None, random_state= 0, min_samples_split= 10, n_estimators=25)
base_results = cross_validate(random_forest, x, y, cv  = cv_split, return_train_score=True)
random_forest.fit(x, y)


# In[ ]:


importance = random_forest.feature_importances_
feature = x.columns
fi = pd.DataFrame()
fi['importance'] = importance
fi['feature'] = feature
fi_sorted = plot_feature_importances(fi)


# ## Compare the performances of the tuned algorithms on our dataset

# In[ ]:


MLA = [    
        # Generalized Linear Models
        LogisticRegressionCV(),
    
        # SVM
        svm.SVC(probability=True, C=1.0, gamma=0.02, kernel='linear'),
        svm.LinearSVC(),
    
        # KNN
        neighbors.KNeighborsClassifier(weights='distance'),
    
        #Discriminant Analysis
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
     
        # Naive Bayes
        naive_bayes.BernoulliNB(),
        naive_bayes.GaussianNB(),
    
        #Trees    
        tree.DecisionTreeClassifier(criterion= 'entropy', max_depth= 4, random_state= 0, splitter= 'random'),
    
        # Ensemble Methods
        ensemble.AdaBoostClassifier(base_estimator = LogisticRegressionCV()),
        ensemble.BaggingClassifier(base_estimator=LogisticRegressionCV(), n_estimators=10),
        ensemble.ExtraTreesClassifier(),
        ensemble.GradientBoostingClassifier(),
        ensemble.RandomForestClassifier(criterion='entropy', min_samples_split=10, n_estimators=25, random_state=0, max_features=3)
     
    ]

cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0)
MLA_columns = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean','MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

row_index = 0
for alg in MLA:
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    cv_results = cross_validate(alg, x,y, cv  = cv_split, return_train_score=True)
    
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
         
    row_index+=1
   

MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)


# In[ ]:


MLA_compare


# ## Conclusion
# From the above table, it can be infered that the ensemble methods are not a good fit for our dataset. Although they are among the highest performers in terms of mean test accuracy, but they are facing the overfitting problem. 
# 
# Ensemble methods mainly works in two ways:
# 
# - either they sample the data multiple times and make many datasets (like bootstrapping), and on each dataset a classifier is being trained and then the combined result of each classifier is presented as output
# - or on each dataset produced, a classifier is being trained with a randomly chosen subset of the features set, and then the combined result of each classifier is presented as output. 
# 
# In our case the dataset as well as the feature set, both are very small. So, while building the ensemble model, there is high probability that the different datasets produced have almost the same type of samples or since there are very few features, the different classifiers are considering the same features again and again. Due to these two reasons, the different classifiers in an ensemble model maybe having high correlations with each other. Hence, they can create the problem of overfitting, when the results of all these classifiers are combined.     
# 
# Now, since decision tree algorithm is among the top performers, we can further try to improve the performance of this algorithm with sklearn's feature selection method by exploiting the feature_importance attribute of decision trees.  

# ## Tune the Decision Tree Model with Feature Selection
# We will use Recursive Feature Elimination (RFE) method which selects features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained through a feature_importances_ attribute. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached. 

# In[ ]:


print('BEFORE RFE Training Shape Old: ', x.shape) 
print('BEFORE RFE Training Columns Old: ', x.columns.values)

print("BEFORE RFE Training w/bin score mean: {:.2f}". format(base_results_dtree['train_score'].mean()*100)) 
print("BEFORE RFE Test w/bin score mean: {:.2f}". format(base_results_dtree['test_score'].mean()*100))
print('-'*10)

#feature selection
dtree_rfe = feature_selection.RFECV(tree.DecisionTreeClassifier(), step = 1, scoring = 'accuracy', cv = cv_split)
dtree_rfe.fit(x, y)

#transform x&y to reduced features and fit new model
X_rfe = x.columns.values[dtree_rfe.get_support()]
rfe_results = cross_validate(dtree, x[X_rfe], y, cv  = cv_split)

print('AFTER RFE Training Shape New: ', x[X_rfe].shape) 
print('AFTER RFE Training Columns New: ', X_rfe)

print("AFTER RFE Training w/bin score mean: {:.2f}". format(rfe_results['train_score'].mean()*100)) 
print("AFTER RFE Test w/bin score mean: {:.2f}". format(rfe_results['test_score'].mean()*100))
print('-'*10)

param_grid = {'criterion': ['gini','entropy'], 
              'splitter': ['best', 'random'], 
              'max_depth': [2,4,6,8,10,None], 
              #'min_samples_split': [2,5,7,10,12], 
              #'min_samples_leaf': [1,3,5,7, 10], 
              'random_state': [0] 
             }

#tune rfe model
rfe_tune_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=param_grid, scoring = 'accuracy', cv = cv_split)
rfe_tune_model.fit(x[X_rfe], y)

print('AFTER RFE Tuned Parameters: ', rfe_tune_model.best_params_)
print("AFTER RFE Tuned Training w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_train_score'][tune_model.best_index_]*100)) 
print("AFTER RFE Tuned Test w/bin score mean: {:.2f}". format(rfe_tune_model.cv_results_['mean_test_score'][tune_model.best_index_]*100))
print('-'*10)


# ## Closing Comments
# After tuning the decision tree model with respect to hyper-parameters as well as feature selection, we are able to reduce the overfitting but still we are getting the mean test accuracy of around 83%. Since, our dataset is very small, the decision tree model will likely have more variance. So, instead we can use either Linear Discriminant Analysis model or Logistic Regression model as both of them have similar accuracy, closer to decision trees. 
# 
# Ensemble models like Adaboost and Bagging can also be a good fit, but at the end, if we are able to achieve accuracy level close to that of complex models with a simple model, then we prefer a simple model for our problem. So, that is why we would like to prefer Logistic Regression model here.     

# ## Credits
# I have learnt a lot of things from the following people, that helped me in building this notebook. I have also borrowed some of the code in this notebook from them. So, I want to give credit, where the credit is due.

# - [Will Koehrsen - Start Here: A Gentle Introduction](https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction)
# 
# - [LD Freeman - A Data Science Framework: To Achieve 99% Accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# 

# In[ ]:




