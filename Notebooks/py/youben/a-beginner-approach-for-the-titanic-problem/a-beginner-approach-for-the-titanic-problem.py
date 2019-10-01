#!/usr/bin/env python
# coding: utf-8

# # Titanic
# 
# As a begginer to ML and Data Science, I tried through this Kernel to practice what I learnt on an End to End project and see if I can make a good prediction.
# 
# I learned how to deal with ML project from the [Hands-On Machine Learning with Sickit-learn and TensorFlow](https://github.com/ageron/handson-ml), my methodology is inspired from this book.
# 
# I will go through each one of this steps:
# 
# - [Prepare the workspace, load the data and take a quick look to it](#Prepare-the-workspace,-load-the-data-and-take-a-quick-look-to-it)
# - [Visualize the data and gain insights: think of new features](#Visualize-the-data-and-gain-insights:-think-of-new-features)
# - [Prepare the data](#Prepare-the-data)
# - [Select a model](#Select-a-model)
# - [Fine-tune the model](#Fine-tune-the-model)
# - [Run on test data](#Run-on-test-data)
# 
# You can find the problem description [here](https://www.kaggle.com/c/titanic).
# 
# Let's begin

# ### Prepare the workspace, load the data and take a quick look to it
# 
# first things first, import the modules needed

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# for the notebook
# %matplotlib inline
# disable warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# get the data and show basic information
train_set = pd.read_csv("../input/train.csv")
train_set.info()


# We can see that we have a lot of missing data in the 'Cabin' feature and some in the 'Age' one.
# We are gonna visualize the data so we can think of a way to impute our data after that.

# ### Visualize the data and gain insights: think of new features

# In[ ]:


train_set.hist(bins=20, figsize=(16,8))
plt.show()


# ##### Notes:
# - We can see here that we have a lot of passengers who are alone (Parch and SibSp)
# - More than 50% of the passengers are in the third class (Pclass)
# - The mean Age is around 18-35, we can check the exact value later 
# - Appromximately 30% of the passengers only survived :/, we can check the exact value later
# - The Fare is generally less than 50
# 
# we will now calculate the mean value for the Age and the Survived features

# In[ ]:


print("Age.mean =", train_set["Age"].mean())
print("Survived.mean =", train_set["Survived"].mean())


# So, the mean age is 29.69 and 38% of the passengers survived.
# 
# Since the Parch/SibSp features are not detailed we can't even think of "parent tend to save their childs" or "husband will be the hero of his spouse" so we will just try to merge them and see.

# In[ ]:


# we make a new copy of the data to work with
t = train_set.copy()
attrs_family_related = ["Family", "Parch", "SibSp"]
t["Family"] = t["Parch"] + t["SibSp"]
t[attrs_family_related].hist(bins=10, figsize=(16,8))
plt.show()


# More than 500 passengers are alone!

# Let us see how the number of Survived passengers is distributed according to our new Family feature

# In[ ]:


t[attrs_family_related].hist(bins=10, figsize=(16, 8), weights=t["Survived"])
plt.show()


# Wow! it's too obvious now that passengers who where alone has a greater chance to survive than families... wait, but we have seen that a lot of passengers were alone, so this histogram is not really representative... We should try something else

# We are gonna implement a function that will be useful to visualize the survival relatively to a certain feature

# In[ ]:


def plot_survival_per_feature(data, feature):
    grouped_by_survival = data[feature].groupby(data["Survived"])
    survival_per_feature = pd.DataFrame({"Survived": grouped_by_survival.get_group(1),
                                        "didnt_Survived": grouped_by_survival.get_group(0),
                                        })
    hist = survival_per_feature.plot.hist(bins=20, alpha=0.6)
    hist.set_xlabel(feature)
    plt.show()


# So what if we use this function to visualize more precisely what's happening

# In[ ]:


plot_survival_per_feature(t, "Family")


# okaay, now we can see that passengers who were alone have not a big chance to survive, actually, less than 30% of the alone passengers survived. You should be aware of what you visualize, you might conclude something wrong! so be careful.

# Apparently, medium sized families have a greater chance so survive than alone passengers and big families. We can think of it like "Alone passengers didn't survive because they haven't someone to help them and big families were unmanageable".

# Let's now get this awesome new function to work on other features

# In[ ]:


plot_survival_per_feature(t, "Age")


# Apparently, young passengers (~< 8) were the one who has the number of survivals greater than 50%.
# Passengers between 65 and 75 didn't survive at all but they were not that much, however a unique passenger who has 80 years survived. We may conclude that childs were rescued more than other passengers so we might add a feature like is_child which is set to the value 1 if the passenger have less than 8 years and the value 0 otherwise.

# What about the Pclass feature?

# In[ ]:


plot_survival_per_feature(t, "Pclass")


# Hmm... the more we increase the class is high, the more the survival/non-survival ratio decrease, it's surely because of the position of the passengers of each class, in the data description it's said :
# 
# `
# pclass: A proxy for socio-economic status (SES)
# 1st = Upper
# 2nd = Middle
# 3rd = Lower
# `
# 
# It's clear now that 1st class was closer to the lifeboats than the others.
# 

# Embarkation

# In[ ]:


t["Embarked"].hist(by=t["Survived"], sharey=True, figsize=(16,8))
plt.show()


# We can see that passengers who embarked at Cherbourg have the highest survival/non-survival ratio, I asked my mother who was next to me and she said with all simplicity, it's maybe because they embarked more on the 1st class so I checked that and ...
# 
# C = Cherbourg, Q = Queenstown, S = Southampton

# In[ ]:


t["Embarked"].groupby(t["Pclass"]).value_counts()


# Allright, the first thing that we conclude is: we should take advices from our mother seriously.

# In[ ]:


# Sex destribution
t["Sex"].value_counts().plot.pie(figsize=(8,8))
plt.show()
# What is the number of survival/non survival in each of the two sex?
t["Sex"].hist(by=t["Survived"], sharey=True, figsize=(16,8))
plt.show()


# There is much more male than female on the Titanic but what about the survivals? didn't guess that already? the survival/non-survival ratio is much greater for the female than the male, isn't that obvious? like we say "les femmes d'abord". I think of encoding this feature into binary with value 1 indicating a female and 0 indicating a male.

# In[ ]:


print(t["Cabin"].dropna())


# For the cabin feature, I started googling for a map or something that can help me visualize what the letters and numbers mean in the cabin number, and I found a good [discussion](https://www.kaggle.com/c/titanic/discussion/4693) on Kaggle that was talking exactly about that. In summary, the cabin feature is closely related to the Pclass feature and the more the letter is close to the 'A' the more the passenger is close to the lifeboats.
# 
# We have two choices for the cabin feature:
# 1. we can just keep the first letter and use the Pclass feature to impute it eg. maping 1st class to one of the letters A,B or C
# 2. we can just get rid of it since it's closely related to the Pclass feature.

# We will talk about the Fare now, but let's take a look at the correlation matrix first.

# In[ ]:


corr_matrix = t.corr()
corr_matrix["Fare"]


# There is a 3 features that correlate with the Fare: Survived, Pclass and Family (SibSp, Parch). We will check the correlation with the Survived with our magic function then we will see for the two other features.

# In[ ]:


plot_survival_per_feature(t, "Fare")


# The more a passenger have a high fare the more it has a chance to survive. The correlation with the Pclass make sense now, we already know that the more the class is low the more the passenger has a chance to survive, this is why the correlation value is negative.

# For the correlation with the family, I don't know if the food is accounted in the fare or the class or something else, I just assume that the more the family is bigger the more they consume, this is why the fare is high when a passenger has family, we may divide the Fare by the Family+1 (+1 for the person itself) to get the fare for one passenger and see if it correlate with other features.

# In[ ]:


t["personal_fare"] = t["Fare"] / (t["Family"] + 1)
plot_survival_per_feature(t, "personal_fare")


# Passengers with greater personal_fare tend to survive more, it doesn't seem to be harmfull so we will think of adding it as a new feature.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
sex = t["Sex"]
sex_encoded = encoder.fit_transform(sex)
t2 = t.copy()
t2["Sex"] = sex_encoded
embarked = t["Embarked"].fillna("C")
embarked_encoded = encoder.fit_transform(embarked)
t2["Embarked"] = embarked_encoded
t2.corr()["Embarked"]


# ### Prepare the data

# We will now build our pipeline of transormation that we will use to get prepared data. But we will first cut the label apart first

# In[ ]:


labels = train_set["Survived"]
features_data = train_set.drop("Survived", axis=1)


# In[ ]:


from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
# store the columns for the learning_data
COLUMNS = None

class Dropper(BaseEstimator, TransformerMixin):
    def __init__(self, to_drop=["PassengerId", "Name", "Ticket", "Cabin"]):
        self.to_drop = to_drop
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.to_drop, axis=1)
    
    
class AttributesExtension(BaseEstimator, TransformerMixin):
    def __init__(self, family=True, personal_fare=True, is_child=True, is_child_and_sex=True):
        self.family = family
        self.personal_fare = personal_fare
        self.is_child = is_child
        self.is_child_and_sex = is_child_and_sex
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.family:
            family = X["Parch"] + X["SibSp"]
            X["Family"] = family
        if self.personal_fare and self.family:
            personal_fare = X["Fare"] / (X["Family"] + 1)
            X["Personal_fare"] = personal_fare
        # is_child improved the model by 2% accuracy
        if self.is_child:
            X["is_child"] = X["Age"] <= 8
        if self.is_child_and_sex:
            X["is_child_and_sex"] = X["Sex"] * X["is_child"]
        
            
        #save columns
        global COLUMNS
        COLUMNS = X.columns.tolist()
        return X
    
    
class AttributesEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, sex=True, embarked=True):
        self.sex = sex
        self.embarked = embarked
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        encoder = LabelEncoder()
        if self.sex:
            sex_encoded = encoder.fit_transform(X["Sex"])
            X["Sex"] = sex_encoded
        if self.embarked:
            #impute with C
            embarked_encoded = encoder.fit_transform(X["Embarked"].fillna('C'))
            X["Embarked"] = embarked_encoded
        return X


# In[ ]:


from sklearn.impute import SimpleImputer
pipeline = Pipeline([
    ('dropper', Dropper()),
    ('encoder', AttributesEncoding()),
    ('extender', AttributesExtension()),
    ('imputer', SimpleImputer(strategy="mean")),
])
learning_data = pipeline.fit_transform(features_data)


# ### Select a model

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm.classes import SVC
from sklearn.metrics import accuracy_score

svc = SVC()
log_reg = LogisticRegression()
#log_reg.fit(learning_data, labels)
rand_for = RandomForestClassifier()
#rand_for.fit(learning_data, labels)

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rand_for,
    "SVM": svc,
}

for model in models.keys():
    scores = cross_val_score(models[model], learning_data, labels, scoring="accuracy", cv=10)
    print("===", model, "===")
    print("scores = ", scores)
    print("mean = ", scores.mean())
    print("variance = ", scores.var())
    models[model].fit(learning_data, labels)
    print("score on the learning data = ", accuracy_score(models[model].predict(learning_data), labels))
    print("")


# Looking at those results, I choosed the logitic regression because:
# 
# - the others looks like they are overfiting
# - it has a good mean score
# - it has a low variance

# ### Fine-tune the model

# This part will be implemented soon

# I was reading about how to select good feature from [here](https://www.kaggle.com/dansbecker/permutation-importance?utm_medium=email&utm_source=mailchimp&utm_campaign=ml4insights) so I decided to try it now that I can't add features on myself, so let's do it.

# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

log_reg.fit(learning_data, labels)
perm_imp = PermutationImportance(log_reg, random_state=1).fit(learning_data, labels)
eli5.show_weights(perm_imp,feature_names=COLUMNS)


# The features are ordered by impact on the model, so the Sex feature has the biggest impact on our model.
# 
# I repeated this process many time and tried to combine features to end up with adding is_child_and_sex feature.

# ### Run on test data

# In[ ]:


test_set = pd.read_csv("../input/test.csv")
pred = pipeline.fit_transform(test_set)


# In[ ]:


log_reg.fit(learning_data, labels)
sub = pd.DataFrame(test_set["PassengerId"], columns=("PassengerId", "Survived"))
sub["Survived"] = log_reg.predict(pred)


# In[ ]:


#write predicted data to submit it
#sub.to_csv("../input/sub.csv", index=False)

