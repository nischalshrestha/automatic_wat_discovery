#!/usr/bin/env python
# coding: utf-8

# # TITANIC CHALLENGE - SURVIVABILITY PREDICTION
# ---

# <img src="https://preview.ibb.co/ehu58K/top_7_strangely_unique_things_that_sank_with_titanic.jpg" alt="top_7_strangely_unique_things_that_sank_with_titanic" border="0">

# The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.
# 
# One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.
# 
# In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

# ## 1. PREPARE PROBLEM
# ---

# ### 1.1 LOAD LIBRARIES

# In[ ]:


import os
import re
import math
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from matplotlib import pyplot as plt


# In[ ]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


# In[ ]:


from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score


# ### 1.2 CONFIGURE LIBRARIES

# In[ ]:


np.warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
sns.set_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])


# ### 1.3 LOAD DATA

# In[ ]:


INPUT_DIR = "../input"
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")


# In[ ]:


# Columns name in order of appearance
ID_COLUMNS = ["PassengerId"]
FEATURE_COLUMNS = ["Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
TARGET_COLUMNS = ["Survived"]

TRAIN_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMNS
TEST_COLUMNS = ID_COLUMNS + FEATURE_COLUMNS


# In[ ]:


TRAIN_DATA = pd.read_csv(TRAIN_CSV, usecols = TRAIN_COLUMNS, index_col = ID_COLUMNS)
TEST_DATA = pd.read_csv(TEST_CSV, usecols = TEST_COLUMNS, index_col = ID_COLUMNS)

ALL_DATA = pd.concat([TRAIN_DATA[FEATURE_COLUMNS], TEST_DATA[FEATURE_COLUMNS]])


# ### 1.4 SUMMARIZE DATA

# In[ ]:


TRAIN_COUNT = TRAIN_DATA.shape[0]
TEST_COUNT = TEST_DATA.shape[0]
ALL_COUNT = TRAIN_COUNT + TEST_COUNT

print("{:,d} total passengers | {:,d} passengers tagged ({:,.0%}) | {:,d} passengers untagged ({:,.0%})".format(
    ALL_COUNT, TRAIN_COUNT, TRAIN_COUNT / ALL_COUNT, TEST_COUNT, TEST_COUNT / ALL_COUNT
))


# In[ ]:


TRAIN_DATA.head()


# In[ ]:


TRAIN_DATA.dtypes


# #### Insights
# * The *Name* feature encapsulated the person's title.
# * The *Cabin* feature encapsulated the assigned deck of the passenger.
# * The *Embarked* feature refers to the Port of Embarkation of the passenger.
# * The *SibSp* feature refers to the number of sibling and spouses of the passenger.
# * The *Parch* feature refers to the number of parents and children of the passenger.

# #### 1.4.1 TITANIC DECKS

# <img src="https://preview.ibb.co/hFMQ8K/Titanic_Deck_Plan.jpg" alt="Titanic_Deck_Plan" border="0">

# #### 1.4.2 TITANIC ITINERARY

# <img src="https://preview.ibb.co/cUvtTK/Titanic_Route_Map.png" alt="Titanic_Route_Map" border="0">

# ## 2. CLEAN AND TRANSFORM DATA
# ---

# ### 2.1 COMPLETE DATA

# In[ ]:


def plot_missing_values(data):
    missing_values = data.isnull().sum().to_frame("Count")
    missing_values = missing_values[missing_values["Count"] > 0]
    missing_values = missing_values.sort_values(by = "Count", ascending = False)
    
    plt.figure(figsize = (15, 10))
    d = sns.barplot(x = missing_values.index, y = missing_values["Count"])
    
    total = len(data)
    for p in d.patches:
        y = p.get_height()
        x = p.get_x()
        
        d.text(x + p.get_width() / 2, y, "{:.1%}".format(y / total), va = "bottom", ha = "center") 
    
    plt.title("Missing Values By Feature")
    plt.xlabel("Feature")
    plt.ylabel("Frequency")
    plt.show()

plot_missing_values(ALL_DATA)


# **Insights**
# * The *Cabin* and *Embarked* can be categorized.
# * The missing *Age* and *Fare* values should be impute.

# #### 2.1.1 FARE

# In[ ]:


ALL_DATA[ALL_DATA["Fare"].isnull()]


# In[ ]:


def impute_missing_fares(data):
    missing_fare_rows = data[data["Fare"].isnull()]
    
    for index, row in missing_fare_rows.iterrows():
        similars = data[(data["Pclass"] == row["Pclass"]) &
                        (data["SibSp"]  + data["Parch"] == row["SibSp"] + row["Parch"]) &
                        (data["Embarked"] == row["Embarked"])]
        data.loc[index, "Fare"] = similars["Fare"].mean()
    
    return data

ALL_DATA = impute_missing_fares(ALL_DATA)


# #### 2.1.2 AGE

# In[ ]:


AGE_MISSING_COUNT = ALL_DATA["Age"].isnull().sum()

print("{:,d} total passengers > {:,d} missing age values ({:,.0%})".format(
    ALL_COUNT, AGE_MISSING_COUNT, AGE_MISSING_COUNT / ALL_COUNT
))


# Wich method should we use to impute the missing age values? Should we simply replace missing values by a statistic like the *mean*, *mode*, *gmean*, *hmean* or *median*. Or, should we use the *mean* of the age of similars passengers. Or, as last mesure, should we implement a mahcine learning algorithm to fill missing age values.

# In[ ]:


def plot_age_and_sex_distribution(data, subtitle, axis):
    copy = data.copy()
    copy["All"] = ""
    
    sns.violinplot(
        hue = "Sex",
        y = "Age",
        x = "All",
        data = copy,
        scale = "width",
        inner = "quartile",
        split = True,
        ax = axis
    )
    axis.set_xlabel("")
    axis.set_title("Age Distribution ({:s})".format(subtitle))


# In[ ]:


def plot_age_imputed_dist(data):
    fig, ax = plt.subplots(3, 2, figsize = (15, 15))
    plot_age_and_sex_distribution(data, "Original", ax[0][0])
    plot_age_and_sex_distribution(data.fillna(data["Age"].mean()), "Impute With Mean", ax[0][1])
    plot_age_and_sex_distribution(data.fillna(stats.gmean(data["Age"].dropna())), "Impute With GMean", ax[1][0])
    plot_age_and_sex_distribution(data.fillna(stats.hmean(data["Age"].dropna())), "Impute With HMean", ax[1][1])
    plot_age_and_sex_distribution(data.fillna(data["Age"].mode()[0]), "Impute With Mode", ax[2][0])
    plot_age_and_sex_distribution(data.fillna(data["Age"].median()), "Impute With Median", ax[2][1])
    sns.despine(trim = True)
    plt.tight_layout()
    plt.show()

plot_age_imputed_dist(ALL_DATA)


# In[ ]:


def impute_missing_ages_with_similars(data):
    missing_age_rows = data[data["Age"].isnull()]
    
    for index, row in missing_age_rows.iterrows():
        similars = data[(data["SibSp"] == row["SibSp"]) &
                        (data["Parch"] == row["Parch"] &
                        (data["Pclass"] == row["Pclass"]))]
        data.loc[index, "Age"] = similars["Age"].mean()
        
    return data


# In[ ]:


def plot_age_imputed_dist(data):
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    plot_age_and_sex_distribution(data, "Original", ax[0])
    
    filled = impute_missing_ages_with_similars(data.copy())
    plot_age_and_sex_distribution(filled, "Impute With Similars Mean", ax[1])
    sns.despine(trim = True)
    plt.tight_layout()
    plt.show()
    
plot_age_imputed_dist(ALL_DATA)


# In[ ]:


def impute_missing_ages_with_ml(data):
    target_cols = ["Age"]
    feature_cols = ["SibSp", "Parch", "Pclass"]
    
    missing_values = data[data["Age"].isnull()]
    present_values = data[~data["Age"].isnull()]
    model = ElasticNet()
    model.fit(pd.get_dummies(present_values[feature_cols]), present_values[target_cols])
    
    missing_values["Age"] = model.predict(missing_values[feature_cols])

    return pd.merge(missing_values, present_values, how = "outer")


# In[ ]:


def plot_age_imputed_dist(data):
    fig, ax = plt.subplots(1, 2, figsize = (15, 5))
    plot_age_and_sex_distribution(data, "Original", ax[0])
    
    filled = impute_missing_ages_with_ml(data.copy())
    plot_age_and_sex_distribution(filled, "Impute With ML (ElasticNet)", ax[1])
    sns.despine(trim = True)
    plt.tight_layout()
    plt.show()
    
plot_age_imputed_dist(ALL_DATA)


# In[ ]:


ALL_DATA = impute_missing_ages_with_ml(ALL_DATA)


# ### 2.2 CORRECT DATA

# The outliers will be handle by using a Robust Scaler while testing.

# ### 2.3 CREATE DATA

# #### 2.3.1 PASSENGER TITLE

# In[ ]:


ALL_DATA["Name"].head()


# **Insights**
# * Each passenger name encapsulated the passenger title.
# * The passenger title inform us of the passenger social group.

# In[ ]:


def get_title(name):
    return name.split(",")[1].split(".")[0].strip().replace("the", "").strip()


# In[ ]:


def create_title_feature(data):
    data["Title"] = data["Name"].apply(lambda name : get_title(name))
    
    return data

ALL_DATA = create_title_feature(ALL_DATA)


# In[ ]:


ALL_DATA["Title"].unique()


# In[ ]:


def plot_title_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["Title"],
        order = data["Title"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("Title Distribution")
    plt.show()
    
plot_title_dist(ALL_DATA)


# #### 2.3.2 PASSENGER SOCIAL GROUP

# In[ ]:


ALL_DATA["Title"].unique()


# **Insights**
# * The passenger title can be regrouped in social group.

# In[ ]:


SOCIAL_GROUP_BY_TITLES = {
    ("Major", "Col", "Capt") : "Officer",
    ("Lady", "Don", "Jonkheer", "Countess", "Dona") : "Royal",
    ("Dr",) : "Academic",
    ("Rev",) : "Clergy",
}

SOCIAL_GROUP_BY_TITLE = {}
for titles, social_group in SOCIAL_GROUP_BY_TITLES.items():
    for title in titles:
        SOCIAL_GROUP_BY_TITLE[title] = social_group


# In[ ]:


def create_socialGroup_feature(data):
    data["SocialGroup"] = data["Title"].map(SOCIAL_GROUP_BY_TITLE)
    
    return data
    
ALL_DATA = create_socialGroup_feature(ALL_DATA)


# In[ ]:


ALL_DATA["SocialGroup"].unique()


# In[ ]:


def plot_socialGroup_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["SocialGroup"],
        order = data["SocialGroup"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("SocialGroup Distribution")
    plt.show()
    
plot_socialGroup_dist(ALL_DATA)


# #### 2.3.3 DECK ASSIGNED

# In[ ]:


ALL_DATA["Cabin"].unique()


# **Insights**
# * From the passenger cabin number, we can deduce the deck assigned to the passenger.

# In[ ]:


def create_deck_feature(data):
    data["Deck"] = data["Cabin"].str[0]
    
    return data

ALL_DATA = create_deck_feature(ALL_DATA)


# In[ ]:


ALL_DATA["Deck"].unique()


# In[ ]:


def plot_deck_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["Deck"],
        order = data["Deck"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("Deck Assigned Distribution")
    plt.show()
    
plot_deck_dist(ALL_DATA)


# #### 2.3.4 CABIN COUNT

# In[ ]:


ALL_DATA["Cabin"].unique()


# **Insights**
# * In some case, more than one cabin can be assigned to a passenger.

# In[ ]:


def create_cabinCount_feature(data):
    data["CabinCount"] = data["Cabin"].str.split().str.len()
    data["CabinCount"] = data["CabinCount"].fillna(0).astype(int)
    
    return data

ALL_DATA = create_cabinCount_feature(ALL_DATA)


# In[ ]:


ALL_DATA["CabinCount"].unique()


# In[ ]:


def plot_cabinCount_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["CabinCount"],
        order = data["CabinCount"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("CabinCount Distribution")
    plt.show()
    
plot_cabinCount_dist(ALL_DATA)


# #### 2.3.4 FAMILY SIZE

# In[ ]:


ALL_DATA.columns


# **Insights**
# * The *SibSp* and *Parch* features encapsulated the number of the family.

# In[ ]:


def create_familySize_feature(data):
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    
    return data

ALL_DATA = create_familySize_feature(ALL_DATA)


# In[ ]:


ALL_DATA["FamilySize"].unique()


# In[ ]:


def plot_familySize_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["FamilySize"],
        order = data["FamilySize"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("FamilySize Distribution")
    plt.show()
    
plot_familySize_dist(ALL_DATA)


# #### 2.3.5 PASSENGER TRAVELING ALONE

# In[ ]:


ALL_DATA.columns


# **Insights**
# * Maybe if the passenger traveling alone, he has more chance to survive.

# In[ ]:


def create_isAlone_feature(data):
    data["TravelingAlone"] = (data["SibSp"] + data["Parch"] == 0) * 1
    
    return data

ALL_DATA = create_isAlone_feature(ALL_DATA)


# In[ ]:


ALL_DATA["TravelingAlone"].describe()


# #### 2.3.6 TICKET PREFIX

# In[ ]:


ALL_DATA["Ticket"].head()


# **Insights**
# * The ticket number can contains a prefix.

# In[ ]:


def create_ticketPrefix_feature(data):
    data["TicketPrefix"] = data["Ticket"].str.extract("(.*)\s+\d*$")
    data["TicketPrefix"] = data["TicketPrefix"].str.replace("\W", "", regex = True)
    data["TicketPrefix"] = data["TicketPrefix"].str.upper().str.strip()
    
    return data

ALL_DATA = create_ticketPrefix_feature(ALL_DATA)


# In[ ]:


ALL_DATA["TicketPrefix"].unique()


# In[ ]:


def plot_ticketPrefix_dist(data):
    plt.figure(figsize = (15, 10))
    sns.countplot(
        data["TicketPrefix"], 
        order = data["TicketPrefix"].value_counts().index
    )
    plt.xticks(rotation = 45)
    plt.title("TicketPrefix Distribution")
    plt.show()
    
plot_ticketPrefix_dist(ALL_DATA)


# ### 2.4 CONVERT DATA

# #### 2.4.1 PASSENGER IS MALE

# In[ ]:


ALL_DATA["Sex"].unique()


# In[ ]:


def convert_sex_feature(data):
    data["IsMale"] = (data["Sex"] == "male") * 1
    
    return data

ALL_DATA = convert_sex_feature(ALL_DATA)


# #### 2.4.2 PORT OF EMBARKATION

# In[ ]:


ALL_DATA["Embarked"].unique()


# In[ ]:


PORTS_MAP = {
    "S" : "Southampton",
    "C" : "Cherbourg",
    "Q" : "Queenstown"
}


# In[ ]:


def convert_embarked_feature(data):
    data["Port"] = data["Embarked"].map(PORTS_MAP)
    
    return data

ALL_DATA = convert_embarked_feature(ALL_DATA)


# ### 2.5 REMOVE DATA

# In[ ]:


ALL_DATA.head()


# **Insights**
# * The *Name*, *Sex*, *Ticket*, *Cabin* and *Embarked* doesn't add any informations. They can be remove safely.

# In[ ]:


ALL_DATA = ALL_DATA.drop(["Name", "Sex", "Ticket", "Cabin", "Embarked"], axis = 1)


# ### 2.6 PREPARE DATA

# In[ ]:


TRAIN_DATA = ALL_DATA[:TRAIN_COUNT].join(TRAIN_DATA[TARGET_COLUMNS], how = "inner")
TEST_DATA = ALL_DATA[TRAIN_COUNT:]

FEATURE_COLUMNS = TRAIN_DATA.columns.difference(TARGET_COLUMNS)


# In[ ]:


TRAIN_DATA.head()


# ## 3. EXPLORATORY DATA ANALYSIS
# ---

# ### 3.1 DESCRIBE DATA

# In[ ]:


FEATURE_COUNT = pd.get_dummies(ALL_DATA).shape[1]

print("{:,d} passengers tagged X {:,d} features = {:,d} entries".format(
    TRAIN_COUNT, FEATURE_COUNT, TRAIN_COUNT * FEATURE_COUNT
))


# ### 3.2 VISUALIZE DATA

# In[ ]:


# To be able to create more readable plots...
TRAIN_DATA["AgeRange"] = pd.cut(TRAIN_DATA["Age"], range(0, 90, 10))
TRAIN_DATA["FareRange"] = pd.cut(TRAIN_DATA["Fare"], range(0, 550, 25))


# #### 3.2.1 TARGET PLOT

# In[ ]:


def plot_target_dist(data):
    plt.figure(figsize = (14, 7))
    sns.countplot(data["Survived"])
    plt.title("Survavibility Distribution")
    
plot_target_dist(TRAIN_DATA)


# **Insights**
# * Because there's a significant difference between the number of survivors and the dead passengers, we will need to stratify our samples.

# **Insights**
# * More passengers died than survived.

# #### 3.2.1 UNIVARIATE PLOT

# In[ ]:


def plot_univariate_dist(data, feature_name, target_name):
    fig = sns.factorplot(
        data = data,
        x = feature_name,
        y = target_name,
        kind = "bar",
        height = 7,
        aspect = 2
    )
    sns.despine(trim = True)
    fig.set_xticklabels(rotation = 45)
    plt.title("Survavibility by {:s}".format(feature_name))


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "Pclass", "Survived")


# **Insights**
# * The upper class had more chance to survive.

# In[ ]:


plot_univariate_dist(TRAIN_DATA, "AgeRange", "Survived")


# **Insights**
# * The children had more chance to survive.
# * The oldery had less chance to survive.

# In[ ]:


plot_univariate_dist(TRAIN_DATA, "SibSp", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "Parch", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "Title", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "SocialGroup", "Survived")


# **Insights**
# * The academic passengers had more chance to survive.

# In[ ]:


plot_univariate_dist(TRAIN_DATA, "CabinCount", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "Deck", "Survived")


# **Insights**
# * The passengers assigned to the F deck had more chance to survived.

# In[ ]:


plot_univariate_dist(TRAIN_DATA, "TravelingAlone", "Survived")


# **Insights**
# * A passenger traveling alone doesn't seem to had more chance to survive.

# In[ ]:


plot_univariate_dist(TRAIN_DATA, "FamilySize", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "TicketPrefix", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "Port", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "IsMale", "Survived")


# In[ ]:


plot_univariate_dist(TRAIN_DATA, "FareRange", "Survived")


# ### 3.2.2 MULTIVARIATE PLOT

# In[ ]:


def plot_multivariate_dist(data, x, y, hue):
    fig = sns.factorplot(
        data = data,
        x = x,
        y = y,
        hue = hue,
        kind = "bar",
        legend_out = False,
        aspect = 2,
        height = 7,
    )
    sns.despine(trim = True)
    fig.set_xticklabels(rotation = 45)
    plt.title("Survavibility by {:s} and {:s}".format(x, hue))
    plt.show()


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "AgeRange", "Survived", "IsMale")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "AgeRange", "Survived", "SocialGroup")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "Deck", "Survived", "CabinCount")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "Deck", "Survived", "Pclass")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "Pclass", "Survived", "IsMale")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "Port", "Survived", "Pclass")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "FareRange", "Survived", "Port")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "AgeRange", "Survived", "TravelingAlone")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "TravelingAlone", "Survived", "IsMale")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "TicketPrefix", "Survived", "Port")


# In[ ]:


plot_multivariate_dist(TRAIN_DATA, "TicketPrefix", "Survived", "Pclass")


# In[ ]:


def plot_correlation_heatmap(data):
    corr = pd.get_dummies(data).corr()
    
    plt.figure(figsize = (25, 20))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap = True)

    sns.heatmap(corr, cmap = cmap, mask = mask, square = True, center = 0, robust = True, linewidths = .2)
    plt.show()


# In[ ]:


plot_correlation_heatmap(TRAIN_DATA[FEATURE_COLUMNS])


# ## 4. MODEL DATA
# ---

# ### 4.1 SPLIT-OUT DATA

# In[ ]:


X_TRAIN = pd.get_dummies(TRAIN_DATA[FEATURE_COLUMNS])
X_TEST = pd.get_dummies(TEST_DATA[FEATURE_COLUMNS])
Y_TRAIN = TRAIN_DATA[TARGET_COLUMNS]


# ### 4.2 DEFINE TEST OPTIONS AND METRICS

# In[ ]:


RANDOM_SEED = 123
K_FOLDS = StratifiedKFold(n_splits = 10, random_state = RANDOM_SEED)


# ### 4.3 SPOT-CHECK MODELS AND ENSEMBLES

# In[ ]:


SCALERS = [
    ("Standard", StandardScaler()),
    ("Robust", RobustScaler()),
    ("MinMax", MinMaxScaler()),
    ("Normalizer", Normalizer())
]


# In[ ]:


def get_scaled_models(model_tuples, scaler_tuple):
    scaler_name, scaler = scaler_tuple
    
    scaled_tuples = []
    for model_name, model in model_tuples:
        scaled_tuples.append((model_name, Pipeline([(scaler_name, scaler), (model_name, model)])))
            
    return scaled_tuples


# In[ ]:


def get_scaled_models_results(model_tuples, scaler_name, x, y, kfolds):
    results = pd.DataFrame([])
    
    for model_name, model in model_tuples:
        results[model_name] = cross_val_score(model, x, y, cv = kfolds)
        
    results = results.melt(var_name = "Model", value_name = "Precision")
    results["Scaler"] = scaler_name

    return results


# In[ ]:


def get_all_models_combinaisons_results(model_tuples, scaler_tuples, x, y, kfolds):
    results = get_scaled_models_results(model_tuples, "None", x, y, kfolds)

    for scaler_tuple in scaler_tuples:
        scaled_models = get_scaled_models(model_tuples, scaler_tuple)
        results = results.append(get_scaled_models_results(scaled_models, scaler_tuple[0], x, y, kfolds))
        
    return results


# In[ ]:


def plot_models_results(results):  
    plt.figure(figsize = (20, 10))
    plt.title('Algorithm Comparison')
    sns.boxplot(data = results, x = "Model", y = "Precision", hue = "Scaler")
    sns.despine(trim = True)
    plt.show()


# #### 4.3.1  MODELS

# In[ ]:


BASE_MODELS = [
    ("SVM", SVC()), 
    ("CART", DecisionTreeClassifier()),
    ("KNN", KNeighborsClassifier()),
    ("LR", LogisticRegression()),
    ("LDA", LinearDiscriminantAnalysis()),
    ("MLP", MLPClassifier()),
    ("GPC", GaussianProcessClassifier()),
]


# In[ ]:


RESULTS = get_all_models_combinaisons_results(BASE_MODELS, SCALERS, X_TRAIN, Y_TRAIN, K_FOLDS)


# In[ ]:


plot_models_results(RESULTS)


# **Insights**
# * Each model have better results when the data is normalized.

# #### 4.3.2 ENSEMBLE MODELS

# In[ ]:


BASE_ENSEMBLES = [
    ("ADABOOST", AdaBoostClassifier()),
    ("GRDBOOST", GradientBoostingClassifier()),
    ("BAGGING", BaggingClassifier()),
    ("FOREST", RandomForestClassifier()),
    ("VOTING", VotingClassifier(BASE_MODELS)),
    ("EXTRATREE", ExtraTreesClassifier()),
    ("XGBOOST", XGBClassifier()),
]


# In[ ]:


RESULTS = get_all_models_combinaisons_results(BASE_ENSEMBLES, SCALERS, X_TRAIN, Y_TRAIN, K_FOLDS)


# In[ ]:


plot_models_results(RESULTS)


# ## 5. IMPROVE ACCURACY
# ---

# ### 5.1 TUNE WITH HYPER-PARAMETERS

# #### 5.1.1 SVM

# ### 5.2 TUNE WITH FEATURES SELECTION

# ## 6. FINALIZE MODEL
# ---

# ### 6.1 CREATE STANDALONE MODEL

# ### 6.2 SAVE MODEL

# ## 7. CHANGE LOGS
# ---

# ## 8. REfERENCES AND CREDITS
# ---
