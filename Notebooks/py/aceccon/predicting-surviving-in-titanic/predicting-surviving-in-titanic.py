#!/usr/bin/env python
# coding: utf-8

# # **Titanic: Machine Learning from Disaster**
# ## **Objective**

# This notebook aims to give an overview about the training dataset from the competition "Titanic: Machine Learning from Disaster". Parts of the code I based on the excelent notebook by Sergei Neviadomski (https://www.kaggle.com/neviadomski/titanic-data-exploration-starter).
# Now, I included a new part which is an basic introduction to predicting modeling, where I explore some models trying to predict who will survive in this tragic event.
# 
# This notebook is structured as follow:
# 1. Loading data
# 2. Basic dataset statistics
# 3. Checking for missing values
# 4. Data visualization
#         4.1 Label feature
#         4.2 Numeric features
#         4.3 Ordinal features
#         4.4 Nominal features
#         4.5 Correlation between attributes
# 5. Dealing with missing values
# 6. Encoding
# 7. Predictive Modeling
# 8. Analysis of Models
#         8.1 Feature Importance
#         8.2 Fine Tuning XGBoost
#         8.3 Fine Tuning SVM rbf
#         8.4 Fine Tuning Knn
#         8.5 Emsembling Models
# 9. Predicting Test Dataset
#         9.1 Preparing Test Dataset
#         9.2 Predicting and Submitting

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# # **1. Loading data**

# In[ ]:


#Reading data from CSV file
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# # **2. Basic dataset statistics**

# Let's take a peek in the first lines and some basic statistics of train dataset.

# In[ ]:


print("The train dataset contains %d examples and %d features" %(df_train.shape[0], df_train.shape[1]))
print(df_train.head())


# There are 12 variables: 5 categorical ("Survived" - our label; "Pclass"; "Sex"; "Cabin"; and "Embarked"); 3 Id ("PassengerId"; and "Name"; Ticket); 4 numerical ("Age"; "SibSp"; "Parch"; and "Fare"). A brief description of these variables: 
#  
#  - Survived: if the passanger survived or not (0 = No; 1 = Yes)
#  - Pclass: ticket class (1 = 1st; 2 = 2nd; 3 = 3rd)
#  - sex: sex of passanger (male or female)
#  - Age: age in years
#  - sibsp: number of siblings / spouses aboard the Titanic
#  - parch: number of parents / children aboud the Titanic
#  - ticket: ticket number
#  - fare: passanger fare
#  - cabin: cabin number
#  - Embarked: port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
#  
# There are a total of 891 examples to train our model.

# In[ ]:


df_train.describe()


# # **3. Checking for missing values**

# Let's check for missing values in the train dataset.

# In[ ]:


missing_df = pd.DataFrame(columns=["# missing", "% missing"])     #create two new columns
missing_df["# missing"] = df_train.isnull().sum()                 #compute number of missing values for each att
missing_df["% missing"] = missing_df["# missing"]/df_train.shape[0] * 100      #compute percentage of missing values
print(missing_df)


# There are a large number of missing values in the age (~20%) and in cabin (~77%). Some features don't affect the prediction, such as "PassengerId", "Name" and "Ticket". Together with "Cabin", which has too many missing values, we'll drop those attributes.

# In[ ]:


to_drop = ["Name", "Ticket", "Cabin"]     #columns to be dropped
    
df_train.drop(to_drop, axis=1, inplace=True);            #drop columns from train dataset
df_test.drop(to_drop, axis=1, inplace=True);             #drop columns from test dataset


# # **4. Data visualization**

# In[ ]:


# Separating features by type

# Id feature
ID_feat = ["PassengerId"]

# Target feature
target_feat = ["Survived"]

# Numeric features
num_feat = ["Age", "Fare"]

# Categorical Features
ordinal_feat = ["Pclass", "SibSp", "Parch"]
nominal_feat = ["Sex", "Embarked"]

df_train["target_name"] = df_train["Survived"].map({0: "Not Survived", 1: "Survived"})
print("The following features were separated by their types: ")
print("\tId: " + str(ID_feat))
print("\tTarget: " + str(target_feat))
print("\tNumeric features: " + str(num_feat) + "\n", end="")
print("\tOrdinal features: " + str(ordinal_feat) + "\n", end="")
print("\tNominal features: " + str(nominal_feat) + "\n", end="")


# > ## **4.1 Label feature**

# Learning about the label: Survived. Let's plot the distribution for "Survived".

# In[ ]:


ax = sns.countplot(df_train[target_feat[0]]);

###A dding percents over bars
# Getting heights of our bars
height = [p.get_height() if p.get_height()==p.get_height() else 0 for p in ax.patches]

# Counting number of bar groups
ncol = int(len(height)/2)

# Counting total height of groups
total = [height[i] + height[i + ncol] for i in range(ncol)] * 2

# Looping through bars
for i, p in enumerate(ax.patches):    
    # Adding percentages   
    ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 

plt.xlabel("Survived");
plt.ylabel("Occurrences");


# The majority of people didn't survived. By only guessing, we could get an accuracy of 62% (guessing that everyone had died). Therefore, our model should perform better than this

# ## **4.2 Numeric features**

# In[ ]:


#Plotting numeric features

for column in num_feat:
    fig = plt.figure(figsize=(18, 12));

    sns.distplot(df_train[column].dropna(), ax = plt.subplot(221));
    plt.xlabel(column, fontsize=14);
    plt.ylabel("Density", fontsize=14);
    plt.suptitle("Plots for " + column, fontsize=18);
    
    sns.distplot(df_train.loc[df_train.Survived==0, column].dropna(),
                color = "red", label="Not survived", ax = plt.subplot(222))
    sns.distplot(df_train.loc[df_train.Survived==1, column].dropna(),
                color = "blue", label="Survived", ax = plt.subplot(222))
    plt.legend(loc="best")
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Density", fontsize=14)
    
    sns.barplot(x="target_name", y=column, data=df_train, ax=plt.subplot(223))
    plt.xlabel("Survived", fontsize=14)
    plt.ylabel("Average " + column, fontsize=14)
    
    sns.boxplot(x="target_name", y=column, data=df_train, ax=plt.subplot(224))
    plt.xlabel("Survived", fontsize=14)
    plt.ylabel(column, fontsize=14)
    plt.show()


# Children have higher change of surviving. The average age of one that survived is slightly lower than the one who didn't. People who paid higher fares also have more chance of surviving. The average fare for the ones who survived are more than the double compared to the ones who didn't.

# ##  **4.3 Ordinal features**

# In[ ]:


###Plotting Ordinal Features
for column in ordinal_feat:
    #Figure initiation
    fig = plt.figure(figsize=(18, 18));
    
    sns.barplot(x="target_name", y = column, data = df_train, ax=plt.subplot(321))
    plt.xlabel("Survived", fontsize = 14)
    plt.ylabel("Average " + column, fontsize = 14)
    plt.suptitle("Plots for " + column, fontsize = 18)
    
    sns.boxplot(x="target_name", y=column, data=df_train, ax=plt.subplot(322))
    plt.xlabel("Survived", fontsize=14)
    plt.ylabel(column, fontsize=14)
    
    ax = sns.countplot(x=column, hue="target_name", data=df_train, ax=plt.subplot(312))
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Number of occurences", fontsize=14)
    plt.legend(loc =1)
    
    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() if p.get_height()==p.get_height() else 0 for p in ax.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax.patches):    
        # Adding percentages   
        ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14)
        
    sns.pointplot(x=column, y="Survived", data=df_train, ax=plt.subplot(313));
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Survived Percentage", fontsize=14)
    plt.show()


# People from 1st class have a higher change of surviving, while people from the lowest class have more chance of not surviving. The class with highest percentage of surviving is 1st.
# 
# People with fewer siblings aboard have higher change of surviving. Above 3 siblings aboard, the chances of surviving drop significantly. The only number of siblings that have higher chance of surviving is 1 sibling (yet not very substantial).
# 
# People with higher number of parents/children aboad have better chances of surviving. Above 3 parents/children aboard, the changes of surviving is almost none. There is no much difference in surviving or not in the range of 0 to 4 parents/children aboard.

# ## **4.4 Nominal features**

# In[ ]:


#Plotting nominal features

for column in nominal_feat:
    #Figure initiation
    fig = plt.figure(figsize=(18, 10));
    
    ax = sns.countplot(x=column, hue="target_name", data=df_train, ax=plt.subplot(221))
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Number of occurrences", fontsize=14)
    plt.suptitle("Plots for " + column, fontsize=18)
    plt.legend(loc=1)
    
    ### Adding percents over bars
    # Getting heights of our bars
    height = [p.get_height() for p in ax.patches]
    # Counting number of bar groups 
    ncol = int(len(height)/2)
    # Counting total height of groups
    total = [height[i] + height[i + ncol] for i in range(ncol)] * 2
    # Looping through bars
    for i, p in enumerate(ax.patches):    
        # Adding percentages
        ax.text(p.get_x()+p.get_width()/2, height[i]*1.01 + 10,
                '{:1.0%}'.format(height[i]/total[i]), ha="center", size=14) 
    
    sns.pointplot(x=column, y="Survived", data=df_train, ax=plt.subplot(222))
    plt.xlabel(column, fontsize=14)
    plt.ylabel("Survived Percentage", fontsize=14)
    plt.show()


# Among the sex, female seems to have more chances in surviving (74% for women vs only 19% for men). People who embarked at "C" por have higher chance of surviving, followed by "Q" and "S". I don't know how the port of embarkation can influence the chance of survival. Maybe there is something related with the location inside the ship.

# ## **4.5 Correlation between attributes**

# In[ ]:


plt.figure(figsize=(10, 10))
sns.heatmap(df_train.corr(), vmin=-1, vmax=1, annot=True, square=True);


# From above, "Survived" is negative correlated with "Pclass": higher class (therefore lower number: 1 is higher class than 2) tends to survived; also is positive correlated with "Fare": who paid more for the fare has more chance of surviving. Obviously, the ones who paid more for the fare are the ones who are from higher class.
# 
# "Patch" and "SibSp" are correlated: who has more parents/children aboard also tends to have more spouse/siblings aboard. Maybe we can create a variable called "Family size" = "Patch" + "SibSp".

# # **5. Dealing with missing values**

# Let's performe the replacement of missing values only after data visualization, so we don't let be mislead from it.

# In[ ]:


#Replace categorical missing value with most frequent category
df_train["Embarked"] = df_train["Embarked"].fillna(df_train["Embarked"].value_counts().index[0])
#Replace numerical missing values with mean
df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())


# Now, we have a dataset without missing values!

# # **6. Encoding**

# Let's replace nominal features encoding them.'

# In[ ]:


df_train['Sex'].replace(['male', 'female'], [0, 1], inplace = True)
df_train['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)


# Also, I will create a new variable called "Family Size", which is the sum of "Parch" and "SibSp".

# In[ ]:


df_train["Family_Size"] = 0
df_train["Family_Size"] = df_train["Parch"] + df_train["SibSp"]


# # 7. Predictive Modeling

# In[ ]:


# Importing algorithms
from sklearn.linear_model import LogisticRegression     # Logistic regression
from sklearn.naive_bayes import GaussianNB              # Naive Bayes
from sklearn import svm                                 # Support Vector Classification
from sklearn.tree import DecisionTreeClassifier         # Decision Tree
from sklearn.ensemble import RandomForestClassifier     # Random Forest
import xgboost as xgb                                   # Extreme Gradient Boost
from sklearn.neighbors import KNeighborsClassifier      #Knn

# Importing preprocessing modules
from sklearn.model_selection import cross_val_score    # Cross validation
from sklearn.model_selection import ShuffleSplit       # Shuffle Split
from sklearn.pipeline import make_pipeline             # Pipeline
from sklearn import preprocessing


# In[ ]:


# Separating train dataset into features and label
df_train_X = df_train.drop(["PassengerId", "Survived", "target_name"], axis=1);   # Selecting coluns of features
df_train_Y = df_train[df_train.columns[1]];                       # Selecting label

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)     # Dividing train dataset into folds

# List with models to be tested
models = [LogisticRegression(), 
          GaussianNB(), 
          svm.SVC(kernel="linear"),
          svm.SVC(kernel="rbf"),
          DecisionTreeClassifier(),
          RandomForestClassifier(),
          xgb.XGBClassifier(),
          KNeighborsClassifier()
         ]

# Create dataframe to store models and their performances
accuracy_models = pd.DataFrame(columns=["Model", "Accuracy", "Std"])
accuracy_models["Model"] = ["Logistic Regression", 
                            "Naive Bayes", 
                            "SVM linear", 
                            "SVM rbf", 
                            "Decision Tree",
                            "Randon Forest", 
                            "XGBoost", "Knn"]


j = 0

# Loop over the models
for model in models:
    clf_model = make_pipeline(preprocessing.StandardScaler(), model);
    scores_model = cross_val_score(clf_model, df_train_X, list(np.ravel(df_train_Y)), cv = cv);
    accuracy_models["Accuracy"][j] = scores_model.mean();
    accuracy_models["Std"][j] = scores_model.std();    
    j += 1
    


# In[ ]:


# Print table with models and performance sorting from the highest accuracy to the lowest
accuracy_models.sort_values(by="Accuracy", ascending=False)


# The best 3 models are XGBoost, SVM with kbr kernel and Knn. Remember that Knn performance is highly dependent of the k paramenter. Therefore, further analysis should compute how its performance develops according with k. 

# # 8. Analysis of Models

# ## 8.1 Feature importance

# From the GXBoost and the Random Forest models, let's check what are the most important features.

# In[ ]:


fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (15, 5))

model = RandomForestClassifier()
model.fit(df_train_X, df_train_Y)
pd.Series(model.feature_importances_, df_train_X.columns).sort_values(ascending = True).plot.barh(width = 0.8, ax=ax[0], color = "b");
ax[0].set_title("Feature importance in Random Forest");

model = xgb.XGBClassifier()
model.fit(df_train_X, df_train_Y)
pd.Series(model.feature_importances_, df_train_X.columns).sort_values(ascending = True).plot.barh(width = 0.8, ax=ax[1], color = "b");
ax[1].set_title("Feature importance in XGBoost");


# In both models, Fare and Age have high importance. If we think about them, we remember that both features have large values if we compare with the range of other attributes. Let's scale only those features and run the feature importance again.

# In[ ]:


fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (15, 5))

# Scaling Age and Fare
df_train[["Age", "Fare"]]
scaler = preprocessing.StandardScaler()
scaler.fit(df_train[["Age", "Fare"]])
pd.DataFrame(scaler.transform(df_train[["Age", "Fare"]]), columns = ["Age", "Fare"])
df_train_X[["Age", "Fare"]] = pd.DataFrame(scaler.transform(df_train[["Age", "Fare"]]), columns = ["Age", "Fare"])

model = RandomForestClassifier()
model.fit(df_train_X, df_train_Y)
pd.Series(model.feature_importances_, df_train_X.columns).sort_values(ascending = True).plot.barh(width = 0.8, ax=ax[0], color = "b");
ax[0].set_title("Feature importance in Random Forest");

model = xgb.XGBClassifier()
model.fit(df_train_X, df_train_Y)
pd.Series(model.feature_importances_, df_train_X.columns).sort_values(ascending = True).plot.barh(width = 0.8, ax=ax[1], color = "b");
ax[1].set_title("Feature importance in XGBoost");


# Even scaling, both features have high importance. While in the Randon Forest the order of importance changed, in XGBoost it doesn't. I believe we can conclude that scaling features in handed internally by XGBoost while for Random Forest doesn't.

# ## 8.2 Fine Tuning XGBoost

# In[ ]:


from sklearn.grid_search import GridSearchCV

df_train_X = df_train.drop(["PassengerId", "Survived", "target_name"], axis=1);
df_train_Y = df_train[df_train.columns[1]];


max_depth_range = list(range(1, 10))
min_child_weight_range = list(range(1, 6))
gamma_range = [i/10 for i in range(0, 5)]

param_grid = {"max_depth": max_depth_range,
              "min_child_weight": min_child_weight_range,
              "gamma": gamma_range
              }

grid_xgb = GridSearchCV(estimator=xgb.XGBClassifier(), 
                    param_grid=param_grid, 
                    cv = 5, 
                    scoring='accuracy', 
                    refit=True
                   )     #setting grid with estimator

xgb_model = make_pipeline(preprocessing.StandardScaler(), grid_xgb)    #creating preprocessing
xgb_model.fit(df_train_X, df_train_Y)      #fitting data

print("Accuracy of the tuned model: %.4f" %grid_xgb.best_score_)
print(grid_xgb.best_params_)


# ## 8.3 Fine Tuning SVM rbf

# In[ ]:


C_range = [0.01, 0.1, 1, 10, 100, 1000]       
gamma_range = [0.001, 0.01, 0.1, 1, 10, 100]

param_grid = {"C": C_range,
              "gamma": gamma_range
              }         #setting grid of parameters

grid_svm = GridSearchCV(estimator = svm.SVC(), 
                    param_grid = param_grid, 
                    cv = 5, 
                    scoring = 'accuracy', 
                    refit = True)   #setting grid with estimator

svm_model = make_pipeline(preprocessing.StandardScaler(), grid_svm)     #creating preprocessing
svm_model.fit(df_train_X, df_train_Y)       #fitting data

print("Accuracy of the tuned model: %.4f" %grid_svm.best_score_)
print(grid_svm.best_params_)


# ## 8.4 Fine Tuning Knn

# In[ ]:


weight_functions = ["uniform", "distance"]
p_values = [1, 2]
n_range = list(range(1, 51))

param_grid = {"n_neighbors": n_range,
              "weights": weight_functions,
              "p": p_values
              }         #setting grid of parameters

grid_knn = GridSearchCV(estimator = KNeighborsClassifier(), 
                    param_grid = param_grid, 
                    cv = 5, 
                    scoring = 'accuracy', 
                    refit = True)   #setting grid with estimator

knn_model = make_pipeline(preprocessing.StandardScaler(), grid_knn)     #creating preprocessing
knn_model.fit(df_train_X, df_train_Y)       #fitting data

print("Accuracy of the tuned model: %.4f" %grid_knn.best_score_)
print(grid_knn.best_params_)


# ## 8.5 Emsembling Models

# The only ensemble model I will try is the Voting Classifier.

# In[ ]:


from sklearn.ensemble import VotingClassifier

voting = make_pipeline(preprocessing.StandardScaler(), VotingClassifier(estimators=[("XGb", grid_xgb.best_estimator_),
                                                                                    ("SVM", grid_svm.best_estimator_),
                                                                                    ("knn", grid_knn.best_estimator_)]));
#scores_model = cross_val_score(clf_model, df_train_X, list(np.ravel(df_train_Y)), cv = cv);

#voting = VotingClassifier(estimators=[("XGb", grid_xgb.best_estimator_),
#                                      ("SVM", grid_svm.best_estimator_),
#                                      ("knn", grid_knn.best_estimator_)])

voting.fit(df_train_X, df_train_Y);

scores_model_ensemble = cross_val_score(voting, df_train_X, list(np.ravel(df_train_Y)), cv = cv);
print("Accuracy of ensemble model: %.3f (+/- %.2f)" %(scores_model_ensemble.mean(), scores_model_ensemble.std()))


# # 9. Predicting test dataset

# To run the same model I developed previously, I need to prepare the test dataset similarly. Thus, let's run the same operations we did for the train dataset: 
# 1. Checking for missing values
# 2. Drop same features we dropped from the train dataset (done in step 3)
# 3. Encoding variables

# In[ ]:


# Checking missing values
missing_df = pd.DataFrame(columns=["# missing", "% missing"])     #create two new columns
missing_df["# missing"] = df_test.isnull().sum()                 #compute number of missing values for each att
missing_df["% missing"] = missing_df["# missing"]/df_test.shape[0] * 100      #compute percentage of missing values
print(missing_df)


# In[ ]:


#Replace numerical missing values with mean from train dataset
df_test["Age"] = df_test["Age"].fillna(df_train["Age"].mean())
df_test["Fare"] = df_test["Fare"].fillna(df_train["Fare"].mean())


# In[ ]:


# Checking missing values
missing_df = pd.DataFrame(columns=["# missing", "% missing"])     #create two new columns
missing_df["# missing"] = df_test.isnull().sum()                 #compute number of missing values for each att
missing_df["% missing"] = missing_df["# missing"]/df_test.shape[0] * 100      #compute percentage of missing values
print(missing_df)


# No more missing values! Let's encode categorical features as the training dataset.

# In[ ]:


# Encoding features
df_test['Sex'].replace(['male', 'female'], [0, 1], inplace = True)
df_test['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)

# Creating new features
df_test["Family_Size"] = 0
df_test["Family_Size"] = df_test["Parch"] + df_test["SibSp"]


# ## 9.2 Predicting and Submitting

# I will use the ensemble model as the final model. It ensembles XGBoost, Knn and SVM with rbf kernel.

# In[ ]:


# Predicting
predict = voting.predict(df_test.drop(["PassengerId"], axis=1))

# Saving prediction data to dataframe
submission = pd.DataFrame(columns=["PassengerId", "Survived"])
submission["PassengerId"] = df_test.PassengerId
submission["Survived"] = predict

# Checking prediction
submission.head()


# In[ ]:


# Saving CSV
submission.to_csv("../working/submit.csv", index=False)
#submission.to_csv("Submission_XGBoost.csv", index=False)


# In[ ]:




