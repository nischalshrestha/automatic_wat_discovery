#!/usr/bin/env python
# coding: utf-8

# ### Titanic Dataset Analysis for Kaggle

# In this classic Kaggle challenge, we try to predict survival on the Titanic based on features of the passengers. 

# First,let's import some necessary functions:

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# ## Helper Functions

# In[ ]:


def plot_subplots(feature, data):
    fx, axes = plt.subplots(2,1,figsize=(15,10))
    axes[0].set_title(f"{feature} vs Frequency")
    axes[1].set_title(f"{feature} vs Survival")
    fig_title1 = sns.countplot(data = data, x=feature, ax=axes[0])
    fig_title2 = sns.countplot(data = data, x=feature, hue='Survived', ax=axes[1])
    
def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()

def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = df.corr()
    _ , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    _ = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
    )


# Get Titanic DataSet, group together for easy data cleaning

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

full = train.append(test, sort=False)

titanic = full.iloc[0:891,:]
full.shape


# In[ ]:


full.columns


# **VARIABLE DESCRIPTIONS:**
# 
# We've got a sense of our variables, their class type, and the first few observations of each. We know we're working with 1309 observations of 12 variables. To make things a bit more explicit since a couple of the variable names aren't 100% illuminating, here's what we've got to deal with:
# 
# 
# **Variable Description**
# 
#  - Survived: Survived (1) or died (0)
#  - Pclass: Passenger's class (1,2,3)
#  - Name: Passenger's name - includes Title of Mr/Miss/Captain etc.
#  - Sex: Passenger's sex 
#  - Age: Passenger's age
#  - SibSp: Number of siblings/spouses aboard
#  - Parch: Number of parents/children aboard
#  - Ticket: Ticket number
#  - Fare: Fare - numerical amount
#  - Cabin: Cabin - denoted by letter
#  - Embarked: Port of embarkation
# 
# [More information on the Kaggle site](https://www.kaggle.com/c/titanic/data)

# From the Kaggle site description, we are told that "Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class." Thus, we will make sure to keep an eye on any variables that identify these groups and include them in the final data analysis.

# First, let's note down the missing data. We can see that Age and Cabin are the main issues for this dataset. We'll keep that in mind and look for ways to deal with them as we explore the dataset

# In[ ]:


full.isnull().sum()


# ##### Let's first work on creating a Title category from Name. We can seperate the passengers by their titles by extracting from the Name column.

# In[ ]:


full.Name.head()


# In[ ]:


full['Title'] = full['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])


# In[ ]:


full.Title.sample(10)


# In[ ]:


full.Title.unique().tolist()


# By plotting against survival with our helper function, we can see that Title will be a useful predictor of survival as Mr. died a lot more and Mrs./Miss./Master. survived better on average.

# In[ ]:


plot_subplots('Title', full)


# What is the title 'the'? Let's fix that error.

# In[ ]:


full.loc[full.Title=='the']


# In[ ]:


full.iloc[759,-1] = "Countess"


# In[ ]:


full.iloc[759,:]


# By looking at a boxplot of Title vs Age, we can conclude that Title is a good variable to use to deal with the missing Age data. We will use it to impute Age later on.

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
sns.boxplot(x='Title', y='Age', data=full)


# In[ ]:


full.groupby("Title").Age.describe()


# In[ ]:


full.groupby("Title").Survived.describe()


# We can use a dictionary to map the rare titles into one:

# In[ ]:


full.Title.value_counts()


# In[ ]:


Title_Dictionary = {
                    "Mme.":        "Mrs",
                    "Mlle.":       "Mrs",
                    "Ms.":         "Mrs",
                    "Mr." :        "Mr",
                    "Mrs." :       "Mrs",
                    "Miss." :      "Mrs",
                    "Master.":     "Master",
                    "Countess":    "Lady",
                    "Dona.":       "Lady",
                    "Lady.":       "Lady"
                    }


# In[ ]:


Mapped_titles = full.Title.map(Title_Dictionary)


# In[ ]:


Mapped_titles.fillna("Rare", inplace=True)


# In[ ]:


full['Titles_mapped'] = Mapped_titles


# In[ ]:


full.Titles_mapped.value_counts()


# In[ ]:


full.Titles_mapped.unique()


# In[ ]:


plot_subplots('Titles_mapped', full)


# In[ ]:


target_columns = []
target_columns.append('Titles_mapped')


# For the Ticket variable, we can observe an interesting phenomenon - duplicate tickets!

# In[ ]:


Ticket = pd.DataFrame(full.Ticket)


# In[ ]:


Ticket.sample(10)


# In[ ]:


Ticket.Ticket.value_counts()


# ### Add column to identify multiple ticket holders

# In[ ]:


Ticket['Count'] = Ticket.groupby('Ticket')['Ticket'].transform('count')


# In[ ]:


Ticket.sample(10)


# In[ ]:


full['Ticket_Count'] = Ticket.Count


# In[ ]:


full.Ticket_Count.head()


# In[ ]:


plot_subplots('Ticket_Count', full)


# ### Seems like Single-ticket holders will more likely die alone... Let's keept this variable!

# In[ ]:


target_columns.append('Ticket_Count')


# Let's look at the other missing data - Cabin

# In[ ]:


full.isnull().sum()


# In[ ]:


cabin = pd.DataFrame()


# In[ ]:


cabin['Cabin'] = full.Cabin


# In[ ]:


cabin.Cabin.value_counts()


# We can suspect that perhaps the missing data means the passengers did not have a cabin. For now, let's fill in the missing data with 'U'. Later on we could also elect to take out this variable from our analysis if we think it isn't particularly predictable since there was so much missing data anyway.

# In[ ]:


cabin.Cabin.fillna("U", inplace=True)


# We can use regex to get the Cabin letter of the other passengers

# In[ ]:


import re


# In[ ]:


def findLetter(string, group):
    return re.match(r"([A-Z]{1})(\d*)", str(string)).group(group)


# In[ ]:


re.match(r"([A-Z]{1})(\d*)", 'U').group(1)


# In[ ]:


cabin.Cabin.sample(10)


# In[ ]:


cabin['Cabin_Letter']  = cabin.Cabin.apply(lambda x: findLetter(x,1))   


# In[ ]:


cabin.sample(10)


# In[ ]:


cabin['Survived'] = full['Survived']


# In[ ]:


cabin.head(10)


# In[ ]:


plot_subplots('Cabin_Letter', cabin)


# In[ ]:


target_columns.append('Cabin_Letter')


# In[ ]:


full['Cabin_Letter'] = cabin.Cabin_Letter


# In[ ]:


full.drop(columns='Cabin', inplace=True)


# ##### Family Size - this is a variable that we can engineer into the data by combining the Siblings and Parent/Children columns. And we can hypothesize that perhaps large families may be able to help each other and survive better

# In[ ]:


family = pd.DataFrame()


# In[ ]:


family["FamilySize"] = full.SibSp + full.Parch + 1


# In[ ]:


family.sample(10)


# In[ ]:


family.describe()


# In[ ]:


family.FamilySize.value_counts()


# In[ ]:


family['Survived'] = full.Survived


# Contrary to what we thought, we can see that it will be more likely that a passenger with NO family will be much more likely to die - which makes sense.

# In[ ]:


plot_subplots('FamilySize', family)


# In[ ]:


target_columns.append('FamilySize')


# In[ ]:


full['FamilySize'] = family.FamilySize


# Now let's look at the Fare variable

# In[ ]:


full[full.Fare.isnull()]


# By plotting Title and Embarked vs Fare, we can see that using the 2 variables to impute the missing value for Fare should be rather sufficient

# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
sns.barplot(x='Titles_mapped', y='Fare', data=full, hue="Embarked")


# In[ ]:


Mr_S_Fare_Mean = full[(full['Titles_mapped']=='Mr') & (full['Embarked']=='S')]['Fare'].mean()


# In[ ]:


Mr_S_Fare_Mean


# In[ ]:


full.loc[full.PassengerId==1044,'Fare'] = Mr_S_Fare_Mean


# In[ ]:


target_columns.append('Fare')


# In[ ]:


target_columns


# By looking at the data for Fare, we can also observe that there are duplicate Fare values. Maybe this has to do with the Ticket duplicates?

# In[ ]:


full.Fare.value_counts()


# In[ ]:


full.sort_values('Ticket').head(10)


# As shown above, seems like the same Ticket holders have a grouped Fare amount in the Fare variable. We can adjust it to Fare per person by dividing with the Ticket_Count variable we created earlier.

# In[ ]:


target_columns.remove('Fare')


# In[ ]:


full['Fare_adjusted'] = full.Fare / full.Ticket_Count


# In[ ]:


target_columns.append('Fare_adjusted')


# Of course, as suggested in the exploring of the Titles variable, it would seem like females have a much higher change of survival

# In[ ]:


plot_subplots('Sex', full)


# In[ ]:


target_columns.append('Sex')


# In[ ]:


full.isnull().sum()


# We still have a lot of missing data! Let's work on Age. In particular, we can look at the Titles and Passenger Class for more info

# In[ ]:


full[full.Age.isnull() == False].groupby(['Titles_mapped', 'Pclass']).describe()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(16, 9)
sns.boxplot(x='Titles_mapped', y='Age', data=full, hue="Pclass")


# Evidently, further separating by passenger class will give better estimates of the missing Age values.

# In[ ]:


full[full.Age.isnull()].groupby(['Titles_mapped', 'Pclass']).describe()


# In[ ]:


def get_Age_mean(title, pclass):
    return full.loc[(full.Age.isnull() == False) & (full.Titles_mapped==title) & (full.Pclass == pclass)].Age.mean()


# In[ ]:


get_Age_mean('Master', 3)


# In[ ]:


full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Master'), 'Age'] = get_Age_mean('Master', 3)


# In[ ]:


for pclass in [1,2,3]:
    full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Mr'), 'Age'] = get_Age_mean('Mr', pclass)


# In[ ]:


for pclass in [1,2,3]:
    full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Mrs'), 'Age'] = get_Age_mean('Mrs', pclass)


# In[ ]:


full.loc[(full.Age.isnull()) & (full.Titles_mapped == 'Rare'), 'Age'] = get_Age_mean('Rare', 1)


# In[ ]:


target_columns.append('Age')


# In[ ]:


full[full.Age.isnull() == False].groupby(['Titles_mapped', 'Pclass']).describe()


# For Embarked, we can just use the mode as it is by far the most common

# In[ ]:


full.Embarked.value_counts()


# In[ ]:


full[full.Embarked.isnull()]


# In[ ]:


full.Embarked.mode()[0]


# In[ ]:


full.loc[full.Embarked.isnull(), 'Embarked'] = full.Embarked.mode()[0]


# In[ ]:


full.loc[full.Embarked.isnull()]


# In[ ]:


full.isnull().sum()


# In[ ]:


target_columns.append('Embarked')


# In[ ]:


target_columns


# In[ ]:


fullfinal = full[target_columns]


# In[ ]:


fullfinal['Pclass'] = full.Pclass


# In[ ]:


fullfinal.dtypes


# * ##### Extra feature engineering - credit to Konstantin's kernel (https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83), he also has some detailed discussion and tips so definitely check it out.  It is likely that Families/Groups would survive/die together, and we can find that out based on their Last Names and Fare

# In[ ]:


full['Last_Name'] = full['Name'].apply(lambda x :str.split(x,',')[0])


# In[ ]:


full.Last_Name.sample(10)


# In[ ]:


DEFAULT_SURVIVAL_VALUE = 0.5


# In[ ]:


full['Family_Survival'] = DEFAULT_SURVIVAL_VALUE


# In[ ]:


for grp, grp_df in full[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId', 'Age',]].groupby(['Last_Name', 'Fare']):
    if (len(grp_df) != 1):
        #found Family group
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
print("Number of passengers with family survival information:", full.loc[full['Family_Survival']!=0.5].shape[0])


# In[ ]:


for _, grp_df in full.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival'] == 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin == 0.0):
                    full.loc[full['PassengerId'] == passID, 'Family_Survival'] = 0
print("Number of passenger with family/group survival information: " 
      +str(full[full['Family_Survival']!=0.5].shape[0]))


# In[ ]:


full.Family_Survival.describe()


# In[ ]:


fullfinal['Family_Survival'] = full.Family_Survival


# Now that we have our variables, we will need to encode the categorical variables. As one-hot encoding may lead to sparsity which virtually ensures that continuous variables are assigned higher feature importance, we will use label-encoding instead. 
# Source: https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/

# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X = fullfinal.copy()
for i in range(0,X.shape[1]):
    if X.dtypes[i]=='object':
        X[X.columns[i]] = le.fit_transform(X[X.columns[i]])
X.head()


# In[ ]:


X['Survived'] = full.Survived


# In[ ]:


plot_correlation_map(X)


# In[ ]:


full_bins = fullfinal.copy()


# In[ ]:


sns.distplot(full_bins.Age)


# In[ ]:


sns.distplot(full_bins.Fare_adjusted)


# In[ ]:


full_bins['AgeBin'] = pd.qcut(full_bins['Age'], 5)


# In[ ]:


full_bins['FareBin'] = pd.qcut(full_bins['Fare_adjusted'], 6)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()


# In[ ]:


full_bins['AgeBin_Code'] = label.fit_transform(full_bins.AgeBin)


# In[ ]:


full_bins['FareBin_Code'] = label.fit_transform(full_bins.FareBin)


# In[ ]:


full_bins['CabinBin_Code'] = label.fit_transform(full_bins.Cabin_Letter)


# In[ ]:


full_bins['EmbarkedBin_Code'] = label.fit_transform(full_bins.Embarked)


# In[ ]:


full_bins.head()


# In[ ]:


full_bin_final = full_bins.drop(columns=['Titles_mapped', 'Cabin_Letter','Fare_adjusted', 'Age', 'Embarked', 'AgeBin', 'FareBin'] )


# In[ ]:


full_bin_final.head()


# In[ ]:


full_bin_final.Sex = label.fit_transform(full_bin_final.Sex)


# In[ ]:


full_bin_final.head()


# In[ ]:


full_train = full_bin_final.copy()


# In[ ]:


full_train.describe()


# In[ ]:


full_train.columns


# In[ ]:


full_train.dtypes


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# In[ ]:


def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)


# In[ ]:


def recover_train_test_target(df):
    global combined
    
    targets = pd.read_csv('../input/train.csv', usecols=['Survived'])['Survived'].values
    train = df.iloc[:891]
    test = df.iloc[891:]
    
    return train, test, targets


# In[ ]:


train, test, targets = recover_train_test_target(full_train)


# In[ ]:


def checkFeatureImportance(dataset):
    train, test, targets = recover_train_test_target(dataset)
    clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
    clf = clf.fit(train, targets)
    
    features = pd.DataFrame()
    features['feature'] = train.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    
    features.plot(kind='barh', figsize=(10, 10))


# In[ ]:


checkFeatureImportance(full_train)


# By looking at the feature importance plot above, we can revise the variables that we will chooes for our predictions:
# 
# Embarked - as its feature importance is low, and it is unlikely that port of origin should affect survival, we will leave this out.
# 
# Cabin - there was a lot of missing data for this variable, so it may be best that we exclude this variable as well
# 
# Ticket Count - as we have already included this variable by adjusting Fare with it, we can exclude it
# 
# Pclass - we will choose to keep this variable as it should still be an important predictor (high correlation with Survival etc.)

# In[ ]:


full_train.columns


# In[ ]:


full_train2 = full_train.drop(columns=['EmbarkedBin_Code', 'CabinBin_Code', 'Ticket_Count'])


# In[ ]:


checkFeatureImportance(full_train2)


# In[ ]:


full_train2.head()


# In[ ]:


full_train2.describe()


# Scale features for better prediction:

# In[ ]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
train = std_scaler.fit_transform(train)
test = std_scaler.fit_transform(test)


# GridSearchCV to find best parameters:

# In[ ]:


# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
    parameter_grid = {
                 'max_depth' : [8,10,12],
                 'n_estimators': [45,47,48,50],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [2, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(n_splits=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)


# We can also use KNN, which had yielded better results

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier 

# turn run_gs to True if you want to run the gridsearch again.
run_gs = True

if run_gs:
    parameter_grid = {
                 'n_neighbors' : [6,7,8,9,10,11,12,114,16,18,20,22],
                 'algorithm': ['auto'],
                 'weights': ['uniform', 'distance'],
                 'leaf_size': list(range(1,50,5)),
                 }
    KNN = KNeighborsClassifier()
    cross_validation = StratifiedKFold(n_splits=10)

    grid_search = GridSearchCV(KNN,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation,
                               verbose=1
                              )

    grid_search.fit(train, targets)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(train, targets)


# In[ ]:


model


# Export results from best estimator to csv for submitting to Kaggle.

# In[ ]:


def to_Kaggle_csv(model, filename):
    output = model.predict(test).astype(int)
    df_output = pd.DataFrame()
    aux = pd.read_csv('test.csv')
    df_output['PassengerId'] = aux['PassengerId']
    df_output['Survived'] = output
    df_output[['PassengerId','Survived']].to_csv(filename, index=False)


# In[ ]:


to_Kaggle_csv(model, 'Family_Survival_KNN_GridSearch.csv')


# ## Score - 0.81339

# In[ ]:




