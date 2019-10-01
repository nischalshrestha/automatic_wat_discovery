#!/usr/bin/env python
# coding: utf-8

# # Titanic Survivor Geometry
# 
# Titanic is the project for beginners on Kaggle. It is a relatively small dataset with a well known story and features.
# This is why most data analysis on the Titanic start just there: they look at the features single or in combination. 
# We sometimes find correlation matrices, but what is we could glimpse at the geometry as a whole?
# 
# ## Can we see the geometry of the Survivor Classification?
# 
# I recently came across an algorithm, that does just that: it projects the problem onto a 2 dimensional plain and preserves thereby as much of the distance between observations as possible.
# 
# ## Background
# The kernel uses the **t-SNE Algorithm**.
# 
# It is a dimensionality reduction algorithm, that has its focus on preserving the distances of the datapoints as much as possible, whille mapping them on a plain, that can be easily visualized by us humans.
# 
# If you want to know more about that algorithm: here is an article for you: It is free, but to read it, you need to login at Oreilly's website:
# [An illustrated introduction to the t-SNE algorithm](https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm)
# 
# # Content
# 1. Loading the Libraries
# 2. Loadinging and preparing the data
# 3. Applying the t-SNE algorithm
# 4. Visualization of the Survivor Geometry
# 5. Interpretation of the Result
# 6. K-Nearest Neighbor Algorithm

# ## 1. Loading Libraries
# We need only the standard libraries for this:
# - numpy
# - pandas
# - matplotlib
# - seaborn

# In[ ]:


# use numpy and pandas
import numpy as np
import pandas as pd

# We need sklearn for preprocessing and for the TSNE Algorithm.
import sklearn
from sklearn.preprocessing import Imputer, scale
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# WE employ a random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().magic(u'matplotlib inline')

# We import seaborn to make nice plots.
import seaborn as sns


# ## 2. Loading and preparing the data
# We need only the train data, since we want to visualize how well the festures separate the target.
# 
# ### The Algorithm expects
# - numeric input
# - no missing values are allowed
# - the data must be sorted by its target
# 
# ### Therefore we make some **basic transformations**:
# - Sex is encoded as 0/1 instead of 'male'/'female'
# - Embarked is encoded as 0/1/2 instead of 'S'/'C'/'Q': so the sequence of enbarkment is kept
# - Cabin is encoded with 0/1 depending whether it is filled or not
# - Name, TicketNr and PassengerId are excluded
# 
# ### We sort the data
# Our sort criteria is the 'Survived' colunm.

# In[ ]:


# import the data: we just need the training data for this visualization
# since we need a target!
X = pd.read_csv('../input/train.csv')

# sort by target
X.sort_values(by='Survived', inplace=True)

# separate the target
y = X['Survived']

# transform all fields that are not numeric:
def prepare_for_ml(X):
    # Cabin is 0 if nan or 1 if filled
    X.Cabin = X.Cabin.apply(lambda x: 0 if pd.isnull(x) else 1) 

    # Sex is turned into 0/1 for male/female
    X.Sex = X.Sex.apply(lambda x: 0 if x == 'male' else 1)

    # Embarked is encoded as 1,2,3 maintaining the order of ports S -> C -> Q
    def get_port_nr(embarked):
        if embarked == 'C':
            return 2
        elif embarked == 'Q':
            return 3 
        else: # cases nan or 'S'
            return 1

    X.Embarked = X.Embarked = X.Embarked.apply(lambda x: get_port_nr(x))

    # Name Ticket and PassengerId are dropped
    X.drop(['Survived', 'Name', 'Ticket', 'PassengerId'], inplace=True, axis=1)
    
    print("Features: ", X.columns)

    # now the missing values are imputed, which are Fare and Age, since Embarked has 
    # already been filled!
    imputer = Imputer()
    X = imputer.fit_transform(X)
    
    # scale the feature values 
    X = scale(X)
    
    return X

# apply transformation
X = prepare_for_ml(X)


# ## 3. Applying the t-SNE algorithm
# We are now ready for applying the algorithm:
# - X is numeric
# - there are no missing values

# In[ ]:


# run the TSNE Algorithm
titanic_proj = TSNE(random_state=RS).fit_transform(X)
titanic_proj.shape


# ## 4. Visualization of the Survivor Geometry
# 
# Now we can visualize the result
# - we use matplotlib and seaborn for this task

# In[ ]:


def scatter(x, colors):
    """this function plots the result
    - x is a two dimensional vector
    - colors is a code that tells how to color them: it corresponds to the target
    """
    
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 2))

    # We create a scatter plot.
    f = plt.figure(figsize=(10, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    
    ax.axis('off') # the axis will not be shown
    ax.axis('tight') # makes sure all data is shown
    
    # set title
    plt.title("Featurespace Visualization Titanic", fontsize=25)
    
    # legend with color patches
    survived_patch = mpatches.Patch(color=palette[1], label='Survived')
    died_patch = mpatches.Patch(color=palette[0], label='Died')
    plt.legend(handles=[survived_patch, died_patch], fontsize=20, loc=1)

    return f, ax, sc

# Use the data to draw ths scatter plot
scatter(titanic_proj, y)


# ## 5. Interpretation of the Result
# 
# ### Some charateristics coincidented with a higher chance of survival
# so some feature made survival more likely, such as
# - being female
# - young children
# - higher passenger class
# 
# ### For other characteristics survival was just fate
# - for other groups: survivers and non survivers could not be distinguished by there features.
# - so survival just boiled down to luck!
# 
# ### That makes survival hard to predict
# - we should meet a limit in our prediction accuracy, that can be overcome, that part is pure luck!

# ## 6. K-Nearest Neighbor Algorithm
# I am just trying one ML algorithm here: kNearest Neighbor, I might compare other algorithms later on.

# In[ ]:


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=test_size, random_state=seed)

# fit model no training data
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import roc_auc_score
roc_auc = roc_auc_score(y_test, predictions)
print(roc_auc)

