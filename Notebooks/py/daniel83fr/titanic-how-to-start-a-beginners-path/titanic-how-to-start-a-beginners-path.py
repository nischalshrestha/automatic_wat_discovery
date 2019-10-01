#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The purpose of this kernel is to formalize the different steps I've been through to understand the Titanic dataset,
# and few firsts submissions into Kaggle competition.
# *Note that i'm newbie, and it may not be the best way to achieve my goal (even maybe not the right way...)*
# 
# (work in progress... all comments are welcome,.)
# 
# # Data Overview
# First we will analyse the different files that are given, to see what is expected.
# 
# ## Loading Data
# We have three files:
# - train.csv
# - test.csv
# - gender_submission.csv (a sample submission file)

# In[7]:


import pandas as pd
pd.options.mode.chained_assignment = None #To hide some warnings
# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# Print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

TRAIN_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
SUBMISSION_FILE = "../input/gender_submission.csv"


# 
# 
# ### Train Data
# 
# 

# In[ ]:


train_data = pd.read_csv(TRAIN_FILE)
train_data.info()


# In[ ]:


train_data = pd.read_csv(TRAIN_FILE)

print("# 5 First lines")
print(train_data[1:6].to_csv(index=False))

print("# Data Infos")
train_data.info()


# - The following columns contains null values :** Age, Cabin, Embarked**
# - The following columns contains text values : ** Name, Sex, Ticket, Cabin, Embarked **
# These data will have to be cleaned before we can use them
# 
# ### Test Data
# we will do the same analyse with the test data

# In[ ]:


test_data = pd.read_csv(TEST_FILE)

print("# 5 First lines")
print(test_data[1:6].to_csv(index=False))

print("# Data Infos")
test_data.info()


# Test data doesn't contain the Survived column as it is the result.
# Else data is in the same format.
# - ** Fare** column contains a null value in test data, but not in train data.

# ### Submisson Data

# In[ ]:


submission_data = pd.read_csv(SUBMISSION_FILE)

print("# 5 First lines")
print(submission_data[1:6].to_csv(index=False))


# The expected format is straight forward, for a list of passenger (passengerId) we want to know if they survived (1) or not (0).

# # Data Prediction
# Now we can start to arrange the data in order to do the prediction.
# For this purpose, we will use Scikit-learn library.
# For the moment, we will choose one of the model that sounds suitable for the exerice => **KNeighborsClassifier**
# This page can be useful : http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html
# 
# Note: This model requires all data to be not null and numerical values. We will also need the data to be normalized.
# 
# we will split our train data using train_test_split and validate them using cross_val_score.
# First we will add a useless column with same value (0), and check how the model performs.
# 

# In[13]:


# Required imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing


# In[14]:


def create_useless_column(data):
    data["UselessColumn"] = 0
    data_new = data[["UselessColumn"]]
    return data_new

def extract_survived(data):
    return data["Survived"]

def apply_model(data, data_label):
    model = KNeighborsClassifier(n_neighbors=2)
    scores = cross_val_score(model, data, data_label, cv=5, verbose=1, scoring='accuracy')
    print(scores.mean())

data_label = extract_survived(train_data)
data = create_useless_column(train_data)
apply_model(data, data_label)


# ## PassengerId
# 

# In[45]:


import pandas as pd
import seaborn as sns
train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)


print("Train: {} null (on {})".format(test_data["PassengerId"].isnull().sum(), len(test_data)))
print("Test: {} null (on {})".format(train_data["PassengerId"].isnull().sum(), len(train_data)))
print("")

print("Correlation:")
corr = train_data[["PassengerId", "Survived"]].corr()

print(corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, vmin=0, center=0)


# There is no correlation between PassengerId and Survived.
# We can drop the feature

# In[38]:


def drop_survived(data):
    return data.drop("Survived", axis=1, errors="ignore")

def drop_passenger_id(data):
    return data.drop("PassengerId", axis=1, errors="ignore")

train_data = pd.read_csv(TRAIN_FILE)
data = train_data
data = drop_survived(data)
data = drop_passenger_id(data)[["Fare"]]
apply_model(data, data_label)


# ## Pclass
# Pclass represent the ticket class for the passengers
# > - TRAIN: Pclass       891 non-null int64
# > - TEST: Pclass         418 non-null int64

# In[62]:


print("Train: {} null (on {})".format(test_data["Pclass"].isnull().sum(), len(test_data)))
print("Test: {} null (on {})".format(train_data["Pclass"].isnull().sum(), len(train_data)))
print("")

print("Correlation:")
pclass = train_data[["Pclass", "Survived"]]
pclass["Class1"] = (pclass["Pclass"] == 1).astype(int)
pclass["Class2"] = (pclass["Pclass"] == 2).astype(int)
pclass["Class3"] = (pclass["Pclass"] == 3).astype(int)
pclass = pclass.drop("Pclass", axis=1)
corr = pclass.corr()
print(corr)

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, center=0)


# In[46]:


groups = test_data.groupby(['Pclass']).size()
groups.plot.bar()


# models expects data to be normalized in order to perform better.
# So we will try to normalize each features.
# 

# In[ ]:


data = train_data.copy()
data["Pclass"] = data["Pclass"] - 1
data["Pclass"] =  preprocessing.maxabs_scale(data["Pclass"])
print(data["Pclass"].value_counts())


# In[ ]:


def handle_pclass(data):
    new_data = data
    new_data["Pclass"] = new_data["Pclass"] -1
    new_data["Pclass"] = preprocessing.maxabs_scale(data["Pclass"])
    return new_data

data = train_data.copy()
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)[["Pclass"]] #Note: We will activate features one by one to avoid errors with columns that are not numerical
apply_model(data, data_label)


# ## Name
# 

# In[108]:


names = train_data["Name"]
print(names[1:10])

print("Train: {} null (on {})".format(test_data["Name"].isnull().sum(), len(test_data)))
print("Test: {} null (on {})".format(train_data["Name"].isnull().sum(), len(train_data)))
print("")


# The format seem to be *LastName, Title. firstname (maiden name)*
# Family name could be interesting to guess origins, or group passengers by family. But it would be pretty difficult to use.
# The title could be an intersting information to extract

# In[124]:


data = train_data.copy()
data["Name"] = data["Name"].str.replace(".",";")
data["Name"] = data["Name"].str.replace(",",";")
data["Name"] = data["Name"].str.split(';', expand=True)[1]

unique1 = data["Name"].unique()
print(unique1)

data2 = test_data.copy()
data2["Name"] = data2["Name"].str.replace(".",";")
data2["Name"] = data2["Name"].str.replace(",",";")
data2["Name"] = data2["Name"].str.split(';', expand=True)[1]
unique2 = data2["Name"].unique()
print(unique2)

for i in unique2:
    if not unique1.__contains__(i):
        print("Missing:" + i)

corr = data[["Name","Survived"]].corr()
print(corr)
dumies = pd.get_dummies(data["Name"])
dumies["Survived"] = data["Survived"]

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, center=0)

data[["Survived"]].groupby([data["Name"], data["Pclass"], data["Sex"], data["Survived"]]).count()


# In[ ]:


data = test_data.copy()
data["Name"] = test_data["Name"].str.replace(".",";")
data["Name"] = data["Name"].str.replace(",",";")
data["Name"] = data["Name"].str.split(';', expand=True)[1]
data["Name"] = data["Name"].str.replace("Capt","Mr")
data["Name"] = data["Name"].str.replace("Col","Mr")
data["Name"] = data["Name"].str.replace("Don","Mr")
data["Name"] = data["Name"].str.replace("Dr","Mr")
data["Name"] = data["Name"].str.replace("Jonkheer","Mr")
data["Name"] = data["Name"].str.replace("Rev","Mr")
data["Name"] = data["Name"].str.replace("Sir","Mr")
data["Name"] = data["Name"].str.replace("Mme","Mrs")
data["Name"] = data["Name"].str.replace("Mlle","Miss")
data["Name"] = data["Name"].str.replace("the Countess","Mme")
data["Name"] = data["Name"].str.replace("Mme","Mrs")
data["Name"] = data["Name"].str.replace("Ms","Mrs")
data["Name"] = data["Name"].str.replace("Major","Mr")
data["Name"] = data["Name"].str.replace("Master","Mr")
data["Name"] = data["Name"].str.replace("Lady","Miss")
groups = data.groupby("Name").size()
groups.plot.bar()


# In[ ]:


def handle_name(data):
    new_data = data
    new_data["Name"] = data["Name"]
    new_data["Name"] = new_data["Name"].str.replace(".",";")
    new_data["Name"] = new_data["Name"].str.replace(",",";")
    new_data["Name"] = new_data["Name"].str.split(';', expand=True)[1]
    new_data["Name"] = new_data["Name"].str.replace("Capt","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Col","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Don","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Dr","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Jonkheer","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Rev","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Sir","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Mme","Mrs")
    new_data["Name"] = new_data["Name"].str.replace("Mlle","Miss")
    new_data["Name"] = new_data["Name"].str.replace("the Countess","Mme")
    new_data["Name"] = new_data["Name"].str.replace("Mme","Mrs")
    new_data["Name"] = new_data["Name"].str.replace("Ms","Mrs")
    new_data["Name"] = new_data["Name"].str.replace("Major","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Master","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Lady","Miss")
    new_data["Miss"] = new_data["Name"].str.contains("Miss").astype(int)
    new_data["Mr"] = new_data["Name"].str.contains("Mr").astype(int)
    new_data["Mrs"] = new_data["Name"].str.contains("Mrs").astype(int)
    new_data = new_data.drop("Name", axis=1)
    return new_data

data = train_data
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data= handle_name(data)[["Pclass", "Miss", "Mr", "Mrs"]]
apply_model(data, data_label)


# ## Sex
# Data is either male or female
# we can simply convert to int
# 
# 

# In[ ]:


def handle_sex(data):
    new_data = data
    new_data["Sex"] = data["Sex"].str.contains("female").astype(int)
    return new_data

test_data = pd.read_csv(TRAIN_FILE)
groups = handle_sex(test_data).groupby("Sex").size()
groups.plot.bar()


# In[ ]:


data = train_data
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data= handle_name(data)
data= handle_sex(data)[["Pclass", "Sex",  "Miss", "Mr", "Mrs"]]
apply_model(data, data_label)


# In[131]:


print("Train: {} null (on {})".format(train_data["Sex"].isnull().sum(), len(train_data)))
print("Test: {} null (on {})".format(test_data["Sex"].isnull().sum(), len(test_data)))
print("")

print("Correlation:")
sex = train_data[["Sex", "Survived"]]
sex["Sex"] = (sex["Sex"] != "female").astype(int)
corr = sex.corr()
print(corr)

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values, center=0)


# ## Age
# For the moment we will just replace null values by the mean

# In[142]:


def handle_age(data):
    new_data = data
    new_data["Age"] = new_data["Age"].fillna(new_data["Age"].mean())
    new_data["Age"] = new_data["Age"]/15
    new_data["Age"] = new_data["Age"].astype(int)
    new_data["Age"] = preprocessing.maxabs_scale(data["Age"])
    return new_data

test_data = pd.read_csv(TRAIN_FILE)
groups.plot.bar()
groups = handle_age(test_data).groupby("Age").size()
groups.plot.bar()

print("Train: {} null (on {})".format(train_data["Age"].isnull().sum(), len(train_data)))
print("Test: {} null (on {})".format(test_data["Age"].isnull().sum(), len(test_data)))
print("")

print("Correlation:")
age = train_data[["Age", "Survived"]]
age["Age"] = age["Age"].fillna(age["Age"].mean())
age["LessThan5"] = (age["Age"] < 5).astype(int)
age["Between5And12"] = ((age["Age"] >= 5) & (age["Age"] < 12)).astype(int)
age["Between12And16"] = ((age["Age"] >= 12) & (age["Age"] < 16)).astype(int)
age["Between16And45"] = ((age["Age"] >= 16) & (age["Age"] < 45)).astype(int)
age["Between45And60"] = ((age["Age"] >= 45) & (age["Age"] < 60)).astype(int)
age["MoreThan60"] = (age["Age"] > 60).astype(int)
age = age.drop("Age", axis=1)

corr = age.corr()
print(corr)

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


data = train_data.copy()
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data= handle_name(data)
data= handle_sex(data)
data= handle_age(data)[["Pclass", "Miss", "Mr", "Mrs", "Sex", "Age"]]
apply_model(data, data_label)


# ## SibSp
# For the moment we will use this feature as is.
# It would be intersting to split this

# In[145]:


groups.plot.bar()
groups = test_data.groupby("SibSp").size()
groups.plot.bar()

print("Train: {} null (on {})".format(train_data["SibSp"].isnull().sum(), len(train_data)))
print("Test: {} null (on {})".format(test_data["SibSp"].isnull().sum(), len(test_data)))
print("")

print("Correlation:")
sib = train_data[["SibSp", "Survived"]]
corr = sib.corr()
print(corr)

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# ## Parch

# In[146]:


groups.plot.bar()
groups = test_data.groupby("Parch").size()
groups.plot.bar()

print("Train: {} null (on {})".format(train_data["Parch"].isnull().sum(), len(train_data)))
print("Test: {} null (on {})".format(test_data["Parch"].isnull().sum(), len(test_data)))
print("")

print("Correlation:")
sib = train_data[["Parch", "Survived"]]
corr = sib.corr()
print(corr)

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


def handle_sibsp(data):
    new_data = data
    new_data["SibSp"] = preprocessing.maxabs_scale(data["SibSp"])
    return new_data
    
def handle_parch(data):
    new_data = data
    new_data["Parch"] = preprocessing.maxabs_scale(data["Parch"])
    return new_data
                     
data = train_data.copy()
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data = handle_name(data)
data = handle_sex(data)
data = handle_age(data)
data = handle_sibsp(data)
data = handle_parch(data)
data = data[["Pclass", "Miss", "Mr", "Mrs", "Sex", "Age", "SibSp", "Parch"]]
apply_model(data, data_label)


# ## Ticket
# Will be excluded for now
# 
# 

# In[147]:


data = train_data.copy()
print(data["Ticket"].head(10))


# In[ ]:


def drop_ticket(data):
    return data.drop(["Ticket"], axis=1)
                        
data = train_data
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data = handle_name(data)
data = handle_sex(data)
data = handle_age(data)
data = handle_sibsp(data)
data = handle_parch(data)
data = drop_ticket(data)
data = data[["Pclass",  "Miss", "Mr", "Mrs", "Sex", "Age", "SibSp", "Parch"]]
apply_model(data, data_label)


# ## Fare
# 
# 

# In[ ]:


def handle_fare(data):
    new_data = data
    new_data["Fare"] = new_data["Fare"].fillna(new_data["Fare"].mean()) #some null values in test_data
    new_data["Fare"] = new_data["Fare"]/ 20
    new_data["Fare"] = new_data["Fare"].astype(int)
    new_data["Fare"] = preprocessing.maxabs_scale(data["Fare"])
    return new_data

test_data = pd.read_csv(TRAIN_FILE)
groups = handle_fare(test_data).groupby("Fare").size()
groups.plot.bar()


# In[ ]:


data = train_data
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data = handle_name(data)
data = handle_sex(data)
data = handle_age(data)
data = handle_sibsp(data)
data = handle_parch(data)
data = drop_ticket(data)
data = handle_fare(data)
data = data[["Pclass", "Mr", "Mrs", "Miss", "Sex", "Age", "SibSp", "Parch", "Fare"]]
apply_model(data, data_label)


# ## Cabin
# for the moment we will just check if cabin number is known or not

# In[ ]:


data = train_data.copy()
print(data["Cabin"].head(15))


# In[ ]:


def handle_cabin(data):
    new_data = data
    new_data["Cabin"] = new_data["Cabin"].isna().astype(int)
    return new_data
                       
data = train_data
data = drop_survived(data)
data = drop_passenger_id(data)
data = handle_pclass(data)
data = handle_name(data)
data = handle_sex(data)
data = handle_age(data)
data = handle_sibsp(data)
data = handle_parch(data)
data = drop_ticket(data)
data = handle_fare(data)
data = handle_cabin(data)
data = data[["Pclass", "Mr", "Mrs", "Miss", "Sex", "Age", "SibSp", "Parch","Fare", "Cabin"]]
apply_model(data, data_label)


# 
# 
# ## Embarked

# In[148]:


print("Train: {} null (on {})".format(train_data["Embarked"].isnull().sum(), len(train_data)))
print("Test: {} null (on {})".format(test_data["Embarked"].isnull().sum(), len(test_data)))
print("")

print("Correlation:")
embarked = train_data[["Embarked", "Survived"]]
embarked["NotEmbarked"] = embarked["Embarked"].isna().astype(int)
embarked["Embarked"] = embarked["Embarked"].fillna("")
embarked['Southampton'] = embarked["Embarked"].str.contains("S").astype(int)
embarked['Queenstown'] = embarked["Embarked"].str.contains("Q").astype(int)
embarked['Cherbourg'] = embarked["Embarked"].str.contains("C").astype(int)
embarked = embarked.drop("Embarked", axis=1)
corr = embarked.corr()
print(corr)

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)


# In[ ]:


def handle_embarked(data):
    new_data = data
    new_data["NotEmbarked"] = new_data["Embarked"].isna().astype(int)
    new_data["Embarked"] = new_data["Embarked"].fillna("")
    new_data['Southampton'] = new_data["Embarked"].str.contains("S").astype(int)
    new_data['Queenstown'] = new_data["Embarked"].str.contains("Q").astype(int)
    new_data['Cherbourg'] = new_data["Embarked"].str.contains("C").astype(int)
    new_data = new_data.drop("Embarked", axis=1)
    return new_data

print(handle_embarked(train_data)[["NotEmbarked", "Southampton", "Queenstown", "Cherbourg"]].head())


# In[ ]:


def process_data(data):
    data = drop_survived(data)
    data = drop_passenger_id(data)
    data = handle_pclass(data)
    data = handle_name(data)
    data = handle_sex(data)
    data = handle_age(data)
    data = handle_sibsp(data)
    data = handle_parch(data)
    data = drop_ticket(data)
    data = handle_fare(data)
    data = handle_cabin(data)
    data = handle_embarked(data)
    return data

data = train_data.copy()
data = process_data(data)
print(data.head())
apply_model(data, data_label)


# # Generate Output
# we have create a first implementation for our prediction.
# Now we can generate the output file.
# 
# So far we have only used part of the train_data to fit our model (as we are using one part for crossvalidation).
# Now we can fit our model with the the whole set, and run the prediction on the test_data.
# Format the result as expected in submission format.
# 
# 
# 
# 

# In[ ]:


model = KNeighborsClassifier(n_neighbors=2)
X_train = pd.read_csv(TRAIN_FILE)   
y_train = X_train["Survived"]
X_train = process_data(X_train) 

X_test = pd.read_csv(TEST_FILE)  
test_labels = X_test[["PassengerId"]]
X_test = process_data(X_test) 

model.fit(X_train, y_train)
result = model.predict(X_test)
print(len(result))
df = pd.DataFrame()
df['PassengerId'] = test_labels.astype(int)
df['Survived'] = result.astype(int)
print(df.head())
df.to_csv("submission.csv", index=False)
print("Done")



# # Tuning the model
# 

# First lets put all the code together:

# In[ ]:


import pandas as pd
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

pd.options.mode.chained_assignment = None #To hide some warnings

TRAIN_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

def drop_survived(data):
    return data.drop("Survived", axis=1, errors="ignore")

def drop_passenger_id(data):
    return data.drop("PassengerId", axis=1, errors="ignore")

def apply_model(data, data_label):
    model = KNeighborsClassifier(n_neighbors=2)
    scores = cross_val_score(model, data, data_label, cv=5, verbose=1, scoring='accuracy')
    print(scores.mean())
    
def extract_survived(data):
    return data["Survived"]

def handle_pclass(data):
    new_data = data
    new_data["Pclass"] = new_data["Pclass"] -1
    new_data["Pclass"] = preprocessing.maxabs_scale(data["Pclass"])
    return new_data

def handle_name(data):
    new_data = data
    new_data["Name"] = data["Name"]
    new_data["Name"] = new_data["Name"].str.replace(".",";")
    new_data["Name"] = new_data["Name"].str.replace(",",";")
    new_data["Name"] = new_data["Name"].str.split(';', expand=True)[1]
    new_data["Name"] = new_data["Name"].str.replace("Capt","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Col","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Don","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Dr","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Jonkheer","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Rev","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Sir","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Mme","Mrs")
    new_data["Name"] = new_data["Name"].str.replace("Mlle","Miss")
    new_data["Name"] = new_data["Name"].str.replace("the Countess","Mme")
    new_data["Name"] = new_data["Name"].str.replace("Mme","Mrs")
    new_data["Name"] = new_data["Name"].str.replace("Ms","Mrs")
    new_data["Name"] = new_data["Name"].str.replace("Major","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Master","Mr")
    new_data["Name"] = new_data["Name"].str.replace("Lady","Miss")
    new_data["Miss"] = new_data["Name"].str.contains("Miss").astype(int)
    new_data["Mr"] = new_data["Name"].str.contains("Mr").astype(int)
    new_data["Mrs"] = new_data["Name"].str.contains("Mrs").astype(int)
    new_data = new_data.drop("Name", axis=1)
    return new_data

def handle_sex(data):
    new_data = data
    new_data["Sex"] = data["Sex"].str.contains("female").astype(int)
    return new_data

def handle_age(data):
    new_data = data
    new_data["Age"] = new_data["Age"].fillna(new_data["Age"].mean())
    new_data["Age"] = new_data["Age"]/15
    new_data["Age"] = new_data["Age"].astype(int)
    new_data["Age"] = preprocessing.maxabs_scale(data["Age"])
    return new_data

def handle_sibsp(data):
    new_data = data
    new_data["SibSp"] = preprocessing.maxabs_scale(data["SibSp"])
    return new_data
    
def handle_parch(data):
    new_data = data
    new_data["Parch"] = preprocessing.maxabs_scale(data["Parch"])
    return new_data

def drop_ticket(data):
    return data.drop(["Ticket"], axis=1)

def handle_fare(data):
    new_data = data
    new_data["Fare"] = new_data["Fare"].fillna(new_data["Fare"].mean()) #some null values in test_data
    new_data["Fare"] = new_data["Fare"]/ 20
    new_data["Fare"] = new_data["Fare"].astype(int)
    new_data["Fare"] = preprocessing.maxabs_scale(data["Fare"])
    return new_data

def handle_cabin(data):
    new_data = data
    new_data["Cabin"] = new_data["Cabin"].isna().astype(int)
    return new_data

def handle_embarked(data):
    new_data = data
    new_data["NotEmbarked"] = new_data["Embarked"].isna().astype(int)
    new_data["Embarked"] = new_data["Embarked"].fillna("")
    new_data['Southampton'] = new_data["Embarked"].str.contains("S").astype(int)
    new_data['Queenstown'] = new_data["Embarked"].str.contains("Q").astype(int)
    new_data['Cherbourg'] = new_data["Embarked"].str.contains("C").astype(int)
    new_data = new_data.drop("Embarked", axis=1)
    return new_data

def process_data(data):
    data = drop_survived(data)
    data = drop_passenger_id(data)
    data = handle_pclass(data)
    data = handle_name(data)
    data = handle_sex(data)
    data = handle_age(data)
    data = handle_sibsp(data)
    data = handle_parch(data)
    data = drop_ticket(data)
    data = handle_fare(data)
    data = handle_cabin(data)
    data = handle_embarked(data)
    return data

data = train_data.copy()
data = process_data(data)
apply_model(data, data_label)

model = KNeighborsClassifier(n_neighbors=2)
X_train = pd.read_csv(TRAIN_FILE)   
y_train = X_train["Survived"]
X_train = process_data(X_train) 
    
X_test = pd.read_csv(TEST_FILE)  
test_labels = X_test[["PassengerId"]]
X_test = process_data(X_test) 
    
model.fit(X_train, y_train)
result = model.predict(X_test)
print(len(result))
df = pd.DataFrame()
df['PassengerId'] = test_labels.astype(int)
df['Survived'] = result.astype(int)
print(df.head())
df.to_csv("submission.csv", index=False)
  


# Some refactoring before continuing...
# Then we will try several models to have an idea about how accurate they are.

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None #To hide some warnings

TRAIN_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
train_data = pd.read_csv(TRAIN_FILE)
test_data = pd.read_csv(TEST_FILE)

   
def extract_survived(data):
    return data["Survived"]

def drop_survived(data):
    return data.drop("Survived", axis=1, errors="ignore")

def drop_passenger_id(data):
    return data.drop("PassengerId", axis=1, errors="ignore")

def handle_pclass(data):
    new_data = data
    new_data["Pclass"] = new_data["Pclass"] -1
    new_data["Pclass"] = preprocessing.maxabs_scale(data["Pclass"])
    return new_data

def replace_multi(string, separators, new_separator):
    for s in separators:
        string = string.replace(s, new_separator)
        
def filter_data_contains(data, column, contain, target_column=None):
    new_data = data
    if target_column == None:
        new_data[column] = new_data[column].str.contains(contain).astype(int)
    else:
        new_data[target_column] = new_data[column].str.contains(contain).astype(int)
    return new_data

def extract_title(x):
    return x.replace(".",";").replace(",",";").split(";")[1]

def handle_name(data):
    new_data = data
    new_data["Name"] = new_data["Name"].apply(extract_title)
    to_replace = {
        "Mr": ["Capt", "Col", "Don", "Dr", "Jonkheer", "Rev", "Sir", "Major", "Master" ],
        "Miss": ["Mlle", "Lady" ],
        "Mrs" :  ["Mme", "the Countess", "Ms" ]
    }
    
    for title in to_replace.keys():
        for t in to_replace[title]:
            new_data["Name"] = new_data["Name"].str.replace(t, title)
        new_data = filter_data_contains(new_data, "Name", title, title)
        
    new_data = new_data.drop("Name", axis=1)
    return new_data

def fill_na_with_mean(data, column):
    new_data = data
    new_data[column] = new_data[column].fillna(new_data[column].mean())
    return new_data

def fill_na_with_mean(data, column):
    new_data = data
    new_data[column] = new_data[column].fillna(new_data[column].mean())
    return new_data

def is_na(data, column):
    new_data = data
    new_data[column] = new_data[column].isna().astype(int)
    return new_data
    
def handle_sex(data):
    return filter_data_contains(data, "Sex", "female")

def handle_age(data):
    new_data = data
    new_data = fill_na_with_mean(new_data, "Age")
    new_data["Age"] =new_data["Age"] / 15
    new_data["Age"] = preprocessing.maxabs_scale(data["Age"])
    return new_data

def handle_sibsp(data):
    new_data = data
    new_data["SibSp"] = preprocessing.maxabs_scale(data["SibSp"])
    return new_data
    
def handle_parch(data):
    new_data = data
    new_data["Parch"] = preprocessing.maxabs_scale(data["Parch"])
    return new_data

def drop_ticket(data):
    return data.drop(["Ticket"], axis=1)

def handle_fare(data):
    new_data = data
    new_data["Fare"] = fill_na_with_mean(new_data, "Fare")
    new_data["Fare"] = new_data["Fare"]/ 20
    new_data["Fare"] = preprocessing.maxabs_scale(data["Fare"])
    return new_data

def handle_cabin(data):
    new_data = data
    new_data = is_na(new_data,"Cabin")
    return new_data

def handle_embarked(data):
    new_data = data
    new_data["NotEmbarked"] =  new_data["Embarked"].isna().astype(int)
    new_data["Embarked"] = new_data["Embarked"].fillna("")
    new_data = filter_data_contains(new_data, "Embarked", "S", "Southampton")
    new_data = filter_data_contains(new_data, "Embarked", "Q", "Queenstown")
    new_data = filter_data_contains(new_data, "Embarked", "C", "Cherbourg")
    new_data = new_data.drop("Embarked", axis=1)
    return new_data

def process_data(data):
    data = drop_survived(data)
    data = drop_passenger_id(data)
    data = handle_pclass(data)
    data = handle_name(data)
    data = handle_sex(data)
    data = handle_age(data)
    data = handle_sibsp(data)
    data = handle_parch(data)
    data = drop_ticket(data)
    data = handle_fare(data)
    data = handle_cabin(data)
    data = handle_embarked(data)
    return data

data = train_data.copy()
data_label = extract_survived(data)
data = process_data(data)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model

classifiers = {
    "Nearest Neighbors" : KNeighborsClassifier(3),
    "LinearRegression": linear_model.LinearRegression(),
    "Ridge": linear_model.Ridge(alpha = .5),
    "Lasso": linear_model.Lasso(alpha = 0.1),
    "ElasticNet": linear_model.ElasticNet(random_state=0),
    "Lars": linear_model.Lars(n_nonzero_coefs=1),
    "LassoLars": linear_model.LassoLars(alpha=.1),
    "Omp": linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=1),
    "BayesianRidge":linear_model.BayesianRidge(),
    "ARDRegression":linear_model.ARDRegression(),
    "LogisitcRegression":linear_model.LogisticRegression(),
    "SGDClassifier":linear_model.SGDClassifier(),
    "Perceptron": linear_model.Perceptron(),
    "PassiveAggressiveClassifier": linear_model.PassiveAggressiveClassifier(),
    "Theil-Sen": linear_model.TheilSenRegressor(random_state=42),
    "RANSAC": linear_model.RANSACRegressor(random_state=42),
    "Huber": linear_model.HuberRegressor(),
    "SVC linear": SVC(kernel="linear", C=0.025),
    "SVC": SVC(gamma=2, C=1, probability=True),
    "GuassianProcess":GaussianProcessClassifier(1.0 * RBF(1.0)),
    "DecisionTree":DecisionTreeClassifier(max_depth=5),
    "RandomForest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "NeutraNet":MLPClassifier(alpha=1),
    "ADABoost":AdaBoostClassifier(),
    "GaussianNB":GaussianNB(),
    "QDA":QuadraticDiscriminantAnalysis()
}

best_model_names = {}
for model_name in classifiers.keys():
    try:
        model = classifiers[model_name]
        scores = cross_val_score(model, data, data_label, cv=5, verbose=1, scoring='accuracy')
        score = scores.mean()
        if score > .8:
            best_model_names[model_name] = scores.mean()
            print("{} {}".format(model_name, scores.mean()))
    except:
        pass
            

print(best_model_names)


res = pd.DataFrame()
X_train = pd.read_csv(TRAIN_FILE)   
y_train = X_train["Survived"]
X_train = process_data(X_train) 
X_test = pd.read_csv(TEST_FILE)  
test_labels = X_test[["PassengerId"]]
X_test = process_data(X_test) 
res["PassengerId"] = test_labels["PassengerId"]

for model_name in best_model_names.keys():
    model =  classifiers[model_name]
    model.fit(X_train, y_train)
    result = model.predict_proba(X_test)[:,1]
    print("{}: {} rows".format(model_name, len(result)))
    res[model_name] = result

models_list = list(best_model_names.keys())
res['ProbaMin'] = res[models_list].min(axis=1)
res['ProbaMax'] = res[models_list].max(axis=1)
res['Accurate'] = (res['ProbaMin']< .20) | (res['ProbaMax']> .80)
res['Survived'] = (res['ProbaMax']-0.5) > (0.5-res['ProbaMin'])
res['Survived'] = res['Survived'].astype(int)
res.to_csv("submission_detail.csv", index=False)

res_filtered = res[["PassengerId","Survived"]]
res_filtered.to_csv("submission.csv", index=False)
res.head(20)

