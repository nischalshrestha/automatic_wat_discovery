#!/usr/bin/env python
# coding: utf-8

# ## Titanic - Guided by a confusion matrix
# This walkthrough uses a [confusion matrix][1] to shape the reasoning behind our model construction. In the end, we'll submit a prediction and receive a **0.80861** score which currently lands in the top 500 (~10%) of the leaderboard. 
# 
# This notebook borrows ideas from other [kernels][2]. But it also explores a hypothesis that *children or husbands of perishing mothers likely did not survive*. The counter hypothesis, *children or wives of surviving fathers or husbands likely survived*, further boosts the model's efficiency.
# 
# Scroll to the bottom if you'd like to skip the boilerplate. 
# 
#   [1]: https://en.wikipedia.org/wiki/Confusion_matrix
#   [2]: https://www.kaggle.com/oysteijo/titanic/titanic-passenger-survival

# In[ ]:


from __future__ import division
import operator
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

from sklearn.metrics import confusion_matrix

from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


X_train = pd.read_csv('../input/train.csv', dtype={'Age': np.float64})
y_train = X_train['Survived']


# In[ ]:


X_test = pd.read_csv('../input/test.csv', dtype={'Age': np.float64})


# ## Functions
# Let's create some functions we'll use to help guide our thinking.

# In[ ]:


def plot_confusion_mat(null_mat, confused_mat, legend_loc=3, scores=None, filename=None):
    """
    Renders a confusion matrix and creates a bar graph representing the model's
    previous and current F1 scores.
    """
    
    new_style = {'grid': False}
    plt.rc('axes', **new_style)

    colors = ['#3182bd', '#fd8d3c', '#fdd0a2', '#c6dbef', '#9467bd', '#98df8a']
    line_alpha = 0.25
    plot_size = confused_mat.sum()
    quad = plot_size / 2
   
    fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))

    ##########
    # AXIS 1 #
    ##########
    
    axis1.tick_params(
        axis='both',
        top='off',right='off', bottom='off', left='off',
        labeltop='off', labelright='off', labelbottom='off', labelleft='off')
    axis1.spines['top'].set_linewidth(0)
    axis1.spines['right'].set_linewidth(0)
    axis1.spines['bottom'].set_linewidth(0)
    axis1.spines['left'].set_linewidth(0)

    axis1.set_xlim(0,plot_size)
    axis1.set_ylim(0,plot_size)

    # plot cross marks
    axis1.plot([quad, quad], [0, plot_size], color='0', alpha=line_alpha)
    axis1.plot([0, plot_size], [quad, quad], color='0', alpha=line_alpha)
    
    # draw confusion matrix
    
    total_tn = null_mat[0][0]
    total_fn = null_mat[1][0]
    
    tn = confused_mat[0][0]
    a = (tn/total_tn) * quad # percentage of total true negatives * quad
    axis1.bar([quad-a], [a], width=a, bottom=quad, lw=0, color=colors[0], label='True negative')
    
    fp = confused_mat[0][1]
    a = (fp/total_fn) * quad # percentage of total false negatives * quad
    axis1.bar([quad], [a], width=a, bottom=quad, lw=0, color=colors[1], label='False positive')
    
    fn = confused_mat[1][0]
    a = (fn/total_fn) * quad # percentage of total false negatives * quad
    axis1.bar([quad-a], [a], width=a, bottom=quad-a, lw=0, color=colors[2], label='False negative')
    
    tp = confused_mat[1][1]
    a = (tp/total_fn) * quad # percentage of total true negatives * quad
    axis1.bar([quad], [a], width=a, bottom=quad-a, lw=0, color=colors[3], label='True positive')

    # legend
    leg = axis1.legend(loc=legend_loc, framealpha=line_alpha, borderpad=1, labelspacing=1, handlelength=1, fontsize=11)
    
    # set the linewidth of each legend object
    for legobj in leg.legendHandles:
        legobj.set_linewidth(0)
    
    ##########
    # AXIS 2 #
    ##########
    
    lw = 0.75 if scores != None else 0.0
    ticks = 'on' if scores != None else 'off'
    axis2.tick_params(
        axis='both',
        top='off',right='off', bottom=ticks, left=ticks,
        labeltop='off', labelright='off', labelbottom=ticks, labelleft=ticks)
    axis2.spines['top'].set_linewidth(0)
    axis2.spines['right'].set_linewidth(0)
    axis2.spines['bottom'].set_linewidth(lw)
    axis2.spines['left'].set_linewidth(lw)
    
    if scores:
    
        bar_width=10
        bar_padding=5
        axis2.set_xlim(0, 40)
        axis2.set_ylim(0, 1)

        x = [1.5*bar_width, 2.5*bar_width]
    
        if len(scores) > 1:
            axis2.bar(bar_width, scores[-2]['f1'], width=bar_width, lw=0, color=colors[4], alpha=0.3)
        axis2.bar(2*bar_width, scores[-1]['f1'], width=bar_width, lw=0, color=colors[4])

        axis2.set_xticks(x)
        axis2.set_xticklabels(['previous', 'current'])
        axis2.set_ylabel('F1 score')
    
    plt.tight_layout(w_pad=6.0)
    if (filename):
        plt.savefig('plots/' + filename)
        


# In[ ]:


def check_classifiers(X_train, Y_train):
    """
    Returns a sorted list of accuracy scores from fitting and scoring passed data
    against several alogrithms.
    """
    _cv = 5
    classifier_score = {}
    
    scores = cross_val_score(LogisticRegression(), X, y, cv=_cv)
    classifier_score['LogisticRegression'] = scores.mean()
    
    scores = cross_val_score(KNeighborsClassifier(), X, y, cv=_cv)
    classifier_score['KNeighborsClassifier'] = scores.mean()
    
    scores = cross_val_score(RandomForestClassifier(), X, y, cv=_cv)
    classifier_score['RandomForestClassifier'] = scores.mean()
    
    scores = cross_val_score(SVC(), X, y, cv=_cv)
    classifier_score['SVC'] = scores.mean()
    
    scores = cross_val_score(GaussianNB(), X, y, cv=_cv)
    classifier_score['GaussianNB'] = scores.mean()

    return sorted(classifier_score.items(), key=operator.itemgetter(1), reverse=True)


# In[ ]:


def get_confused_mat(X, y):
    """
    Returns a 2X2 matrix listing true negatives, false postives, false negatives and
    true positives (a confusion matrix).
    """
    classifier = LogisticRegression()
    classifier.fit(X, y)
    return confusion_matrix(y, classifier.predict(X))


# In[ ]:


confusion_scores = []
def get_confusion_scores(clf, X, y):
    """
    Returns relevant confusion matrix scores.
    """
    
    # accuracy: fraction of the classifier's predictions that are correct.
    accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

    # precision: fraction of correct positive predictions.
    # A precision score of one indicates the classifier did not make any false positive predictions.
    precision = cross_val_score(clf, X, y, cv=5, scoring='precision')

    # recall: fraction of truly positive instances the classifier recognizes.
    # A recall score of one indicates the classifier did not make any false negative predictions.
    recall = cross_val_score(clf, X, y, cv=5, scoring='recall')

    # F1: harmonic mean bw precision and recall
    f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')
    
    return {'accuracy': np.mean(accuracy),
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1)}


# ### Missing data
# Before we get started, we need to fill missing ages for several passengers. We'll use mean age of all passengers by name prefix as fill values for missing ages.

# In[ ]:


_name = 'Name'
_age = 'Age'
_master = 'Master.'
_miss = 'Miss.'
_mrs = 'Mrs.'
_mr = 'Mr.'
_dr = 'Dr.'

mean_ages = {
    'Master.': int(X_train[X_train[_name].apply(lambda x: x.find(_master)) != -1][_age].dropna().mean()),
    'Miss.': int(X_train[X_train[_name].apply(lambda x: x.find(_miss)) != -1][_age].dropna().mean()),
    'Mrs.': int(X_train[X_train[_name].apply(lambda x: x.find(_mrs)) != -1][_age].dropna().mean()),
    'Mr.': int(X_train[X_train[_name].apply(lambda x: x.find(_mr)) != -1][_age].dropna().mean()),
    'Dr.': int(X_train[X_train[_name].apply(lambda x: x.find(_dr)) != -1][_age].dropna().mean())
}

def approx_age(nm):
    if (nm.find(_master) != -1):
        return mean_ages[_master]
    elif (nm.find(_miss) != -1):
        return mean_ages[_miss]
    elif (nm.find(_mrs) != -1):
        return mean_ages[_mrs]
    elif (nm.find(_mr) != -1):
        return mean_ages[_mr]
    else:
        return mean_ages[_dr]

# where Age is nan, get the Name and apply approx_age; return a Series as index: Age
updated_ages = X_train[np.isnan(X_train[_age])][_name].apply(approx_age)

# where Age is nan, update values
X_train[_age].fillna(updated_ages, inplace=True)

# do the same for the test set
updated_ages = X_test[np.isnan(X_test[_age])][_name].apply(approx_age)
X_test[_age].fillna(updated_ages, inplace=True)


# <a id='the_destination'></a>

# ### Start with a null hypothesis
# As a starting point, let's create the most simple model possible and assert everyone died. From here on out, if updates to our model improve accuracy over the null hypothesis, we know we're doing something right.

# In[ ]:


null_hypothesis = 1 - X_train.Survived.mean()
null_hypothesis


# Let's think in terms of error. Our current training error is **0.39** (1 - 0.61). That's a lot of error. Fortunately, a good portion is *reducible error*. We will construct generalizations about our data and incrementally transfer error to accuracy. 
# 
# Some of this error is *irreducible error*. Take some time to read first hand accounts of [people who jumped into lifeboats][1] as they were lowered, or [jumped into the water][2] from the deck of the ship, or [exited a lifeboat before it was lowered][3] to look for missing family members. Chance was a large factor determining passengers' survival. If we create an efficient model, our error will largely describe random chance that we'll never be able to generalize.
# 
# 
#   [1]: https://www.encyclopedia-titanica.org/titanic-survivor/sahid-nakid.html
#   [2]: https://www.encyclopedia-titanica.org/titanic-survivor/john-borland-thayer-jr.html
#   [3]: https://www.encyclopedia-titanica.org/titanic-victim/bessie-waldo-allison.html

# In[ ]:


# run thru our classifiers just for fun
X_train['null_hypo'] = 0.0
X = DataFrame(X_train['null_hypo'])
y = X_train['Survived']
scores = check_classifiers(X, y)
scores


# In[ ]:


# supress warnings about 0.0 scores
import warnings
warnings.filterwarnings('ignore')
confusion_scores.append(get_confusion_scores(LogisticRegression(), X, y))


# In[ ]:


confused_mat = get_confused_mat(X, y)
null_mat = confused_mat.copy()
plot_confusion_mat(null_mat, confused_mat, legend_loc=1, scores=confusion_scores)


# Using the matrix above, we can see what happens when we label everyone as a victim. The top row represents 549 victims in our training set. The bottom row represents 342 survivors. 
# 
# Since our model labels *everyone* as a victim, we misclassify (orange) all the survivors. Our confusion matrix renders all 342 survivors as false negatives.  
# 
# Clearly, not everyone died. Let's identify some features to account for survivors and pull some of these false negatives over to true positives.

# ### Sex
# Accounts of the evacuation effort indicate the ship's officers applied the 'Women and children first' doctrine when organizing people in lifeboats. Simply asserting *all women survived and all men died* should improve our model. Of course we know some women died and some men survived. You can get details in this [gender as a feature analysis](https://gist.github.com/vinceallenvince/06c83b0a1c62a10b296cac79f249e394). Meanwhile, we're starting with broad generalizations and will refine our features along the way.

# In[ ]:


# create dummy variables
_feature = 'Sex'

# train
dummies = pd.get_dummies(X_train[_feature])
X_train = X_train.join(dummies)

# test
dummies = pd.get_dummies(X_test[_feature])
X_test = X_test.join(dummies)


# In[ ]:


# drop male bc it has the lower average of survived passengers
# https://gist.github.com/vinceallenvince/06c83b0a1c62a10b296cac79f249e394
features = ['female']


# In[ ]:


X = DataFrame(X_train[features])
y = y_train
scores = check_classifiers(X, y)
scores


# In[ ]:


confusion_scores.append(get_confusion_scores(LogisticRegression(), X, y))
plot_confusion_mat(null_mat, get_confused_mat(X, y), scores=confusion_scores)


# Our confusion matrix reflects a big change. The light blue square represents the female survivors in 'female' feature (true positives). It's not full because some women died. We can find these women in the orange square at the top right (false positives).
# 
# We still have some false negatives. The "Women and children first" doctrine accounted for male children. When we assert all males died, we misclassify a large group of male children as victims who actually survived.
# 
# Also, notice we have a non-zero F1 score. An [F1 score](https://en.wikipedia.org/wiki/F1_score) is the harmonic mean between [precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall) and is a measure our model's accuracy.
# 
# Precision is the fraction of true positive predictions as opposed to false positives. High precision reflects high quality results. Recall is the fraction of true positive predictions as opposed to false negatives. High recall reflects high quantity results.
# 
# As we improve our F1 score, the sum of the area of the orange boxes in the confusion matrix will get smaller.

# ### Person
# We can account for the misclassified children if we engineer a 'Person' feature with the labels 'male_adult', 'female_adult' and 'child'. As a refinement to our model, we'll assert *all adult females and all children survived while all adult males died*. Please check out a [Person feature EDA](https://gist.github.com/vinceallenvince/9b1cd6bfbf5b8ad49e696715a8c8f0ad) for survival rates of these groups.

# In[ ]:


feature = 'person'

# It's likely the age threshold for adults was younger in the early 1900s.
# Account of a 9 year-old boy almost getting refused a lifeboat:
# https://www.encyclopedia-titanica.org/titanic-survivor/winnie-coutts.html
# May want to try younger ages here.
child_age = 14

def get_person(passenger):
    """
    Returns a person value of 'female_adult', 'male_adult', 'child'.
    """
    age, sex = passenger
    
    if (age < child_age):
        return 'child'
    elif (sex == 'female'):
        return 'female_adult'
    else:
        return 'male_adult'
    
X_train = X_train.join(DataFrame(X_train[['Age', 'Sex']].apply(get_person, axis=1), columns=['person']))
X_test = X_test.join(DataFrame(X_test[['Age', 'Sex']].apply(get_person, axis=1), columns=['person']))

X_train['person'].value_counts().sort_index()


# In[ ]:


_feature = 'person'

_columns = ['male_adult', 'female_adult', 'child']

# train
dummies = pd.get_dummies(X_train[_feature])
X_train = X_train.join(dummies)

# test
dummies = pd.get_dummies(X_test[_feature])
X_test = X_test.join(dummies)


# In[ ]:


# drop male_adult as it has the lower average of survived passengers
# https://gist.github.com/vinceallenvince/9b1cd6bfbf5b8ad49e696715a8c8f0ad
features = ['female_adult', 'child']


# In[ ]:


X = DataFrame(X_train[features])
y = y_train
scores = check_classifiers(X, y)
scores


# In[ ]:


confusion_scores.append(get_confusion_scores(LogisticRegression(), X, y))
plot_confusion_mat(null_mat, get_confused_mat(X, y), scores=confusion_scores)


# Our score slightly improved but not as much as we may have anticipated. By accounting for children, we regained 23 male children misclassified as victims. These passengers were transferred from false negatives to true positives.
# 
# At the same time, our improvements were offset by an increase in false positives. Not all children survived. To account for this increase, we need to add dimension to our model.

# ### Class
# Looking at survival rates, first and second class had much higher survival rates for women and children. With a class feature, our model will assert *all female adults and children in classes 1 and 2 survived and everyone else died*. Look for overall survival rates in this [Pclass feature EDA](https://gist.github.com/vinceallenvince/58099082334230e90b3aec1bfd4f3804).

# In[ ]:


# create dummy variables for person column. 
_feature = 'Pclass'

# train
dummies = pd.get_dummies(X_train[_feature], prefix='class')
X_train = X_train.join(dummies)

# test
dummies = pd.get_dummies(X_test[_feature], prefix='class')
X_test = X_test.join(dummies)


# In[ ]:


# drop class3 bc it has the lower average of survived passengers across female_adults, male_adults and children
# https://gist.github.com/vinceallenvince/58099082334230e90b3aec1bfd4f3804
features = ['female_adult', 'child', 'class_1', 'class_2']


# In[ ]:


X = DataFrame(X_train[features])
y = y_train
scores = check_classifiers(X, y)
scores


# In[ ]:


confusion_scores.append(get_confusion_scores(LogisticRegression(), X, y))
plot_confusion_mat(null_mat, get_confused_mat(X, y), scores=confusion_scores)


# Our score improved again. Given the low survival rate in class 3, we increased our true negatives and reduced our false positives. We're now accounting for women and children who died in class 3.
# 
# However, our false negatives are back up again. And our F1 decreased. Contrary to our model's assertion, not all class 3 passengers died.
# 
# Can we find something in our data to account for surviving class 3 passengers?
# 
# It's tricky.
# 
# First, surviving class 3 passengers were very lucky. All other passengers had more time, more access and less people to compete with for a seat in the lifeboats. We can't generalize what's largely attributable to chance.
# 
# Second, whatever feature we identify to account for surviving class 3 passengers needs to apply to class 1 and 2 passengers as well. If not, we'll trade false negatives between features.
# 
# We need a featured shared by all passengers to more accurately account for passengers in class 3. Let's start with a correlation matrix and introduce two new features 'SibSp' and 'Parch'. These indicate number of Siblings/Spouses and number of Parents/Children respectively. You can also check out a [Family feature EDA](https://gist.github.com/vinceallenvince/7d5ccc1210b29840dde31d15dd885103).

# In[ ]:


corr = X_train[['female_adult', 'male_adult', 'SibSp', 'Parch']].corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, annot=True)


# We can see female adults were more likely to travel with family. Knowing the "Women and children first" doctrine was in place, if a female passenger died, *it's highly likely her spouse or children died as well*. The inverse is also likely true, if a male passenger survived, *it's highly likely his spouse or children survived*.
# 
# We'll likely find perishing female adults in class 3. If we can create a feature 'perishing_mother_spouse', it's likely we can better account for victims in class 3 and lower our false positives.
# 
# We can also create a 'surviving_father_husband' feature and account for surviving adult males and their families. Adult males had a very low survival rate, especially in class 3. However, if we can find some with families, it should improve our true positives and lower our false negatives.

# ### Surnames
# To connect males and females with their families, we're going to engineer a feature called **surname** to make it easier to work with passengers' last names.

# In[ ]:


def process_surname(nm):
    return nm.split(',')[0].lower()

X_train['surname'] = X_train['Name'].apply(process_surname)
X_test['surname'] = X_test['Name'].apply(process_surname)


# ### Surviving fathers and husbands, perishing mothers and wives
# We'll first create a feature called **perishing_wife_mother** to account for children or husbands of female victims.

# In[ ]:


# find female_adult victims with family
perishing_female_surnames = list(set(X_train[(X_train.female_adult == 1.0) &
                                     (X_train.Survived == 0.0) &
                                     ((X_train.Parch > 0) | (X_train.SibSp > 0))]['surname'].values))

print('Total female adult victims with family:', len(perishing_female_surnames))


# In[ ]:


# Return 1 if passenger shares a surname w a perishing female adult.
# We're not checking if passengers w matching surname are in the same Pclass. Could introduce error.
def perishing_mother_wife(passenger): 
    surname, Pclass, person = passenger
    return 1.0 if (surname in perishing_female_surnames) else 0.0

X_train['perishing_mother_wife'] = X_train[['surname', 'Pclass', 'person']].apply(perishing_mother_wife, axis=1)
X_test['perishing_mother_wife'] = X_test[['surname', 'Pclass', 'person']].apply(perishing_mother_wife, axis=1)


# In[ ]:


print('Total passengers traveling with a female adult victim: ', X_train['perishing_mother_wife'].sum())
print('Note: includes the female adult victim.')


# Next, we'll create a feature called **surviving_father_husband** to account for children or wives of male survivors.

# In[ ]:


surviving_male_surnames = list(set(X_train[(X_train.male_adult == 1.0) &
                                     (X_train.Survived == 1.0) &
                                     ((X_train.Parch > 0) | (X_train.SibSp > 0))]['surname'].values))

print('Total male adult survivors with family: ', len(surviving_male_surnames))


# In[ ]:


# Return 1 if passenger shares a surname w a surviving male adult.
# We're not checking if passengers w matching surname are in the same Pclass. Could introduce error.
def surviving_father_husband(passenger): 
    surname, Pclass, person = passenger
    return 1.0 if (surname in surviving_male_surnames) else 0.0

X_train['surviving_father_husband'] = X_train[['surname', 'Pclass', 'person']].apply(surviving_father_husband, axis=1)
X_test['surviving_father_husband'] = X_test[['surname', 'Pclass', 'person']].apply(surviving_father_husband, axis=1)


# In[ ]:


print('Total passengers traveling with a male adult survivor: ', X_train['surviving_father_husband'].sum())
print('Note: includes the male adult survivor.')


# In[ ]:


features = ['female_adult', 'child', 'class_1', 'class_2', 'perishing_mother_wife']


# In[ ]:


X = DataFrame(X_train[features])
y = y_train
scores = check_classifiers(X, y)
scores


# In[ ]:


confusion_scores.append(get_confusion_scores(LogisticRegression(), X, y))
plot_confusion_mat(null_mat, get_confused_mat(X, y), scores=confusion_scores)


# By accounting for perishing mothers/wives, we greatly improved our accuracy and reduced our false positives. It makes sense our model reduces false positives since we're making connections between people who likely died. The trade off between false positive and true negative is likely male children traveling in class 3 who we misclassified as survivors.

# In[ ]:


features = ['female_adult', 'child', 'class_1', 'class_2', 'perishing_mother_wife', 'surviving_father_husband']


# In[ ]:


X = DataFrame(X_train[features])
y = y_train
scores = check_classifiers(X, y)
scores


# In[ ]:


confusion_scores.append(get_confusion_scores(LogisticRegression(), X, y))
plot_confusion_mat(null_mat, get_confused_mat(X, y), scores=confusion_scores)


# By connecting male survivors (surviving_father_husband) with their families, we reduced our false negatives. The trade off between false negative and true positives is likely the male adults and male children in class 3 we misclassified as victims.

# ### Submit
# Let's submit and see where we stand.

# In[ ]:


def param_grid_search(X, y, clf, parameter_grid):
    skf = StratifiedKFold(n_splits=5)
    splits = skf.get_n_splits(X, y)
    grid_search = GridSearchCV(clf,
                           param_grid=parameter_grid,
                           cv=splits)
    grid_search.fit(X, y)
    return grid_search

def print_scores(grid_search):
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    print('Best estimator: {}'.format(grid_search.best_estimator_))


# In[ ]:


parameter_grid = [{
        'penalty': ['l1'],
        'C':(0.01, 0.1, 1, 10, 100, 1000),
        'fit_intercept': [True, False],
        'solver': ['liblinear']
    },
    {
        'penalty': ['l2'],
        'C':(0.01, 0.1, 1, 10, 100, 1000),
        'fit_intercept': [True, False],
        'solver': ['newton-cg', 'lbfgs', 'sag']
    },
    {
        'penalty': ['l2'],
        'C':(0.01, 0.1, 1, 10, 100, 1000),
        'fit_intercept': [True, False],
        'solver': ['lbfgs'],
        'multi_class': ['ovr', 'multinomial']
}]
grid_search = param_grid_search(X, y, LogisticRegression(), parameter_grid)
print_scores(grid_search)
output = grid_search.predict(X_test[features]).astype(int)
df_output = pd.DataFrame()
df_output['PassengerId'] = X_test['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('titanic_submit.csv',index=False)


# Our submitted score returned: **0.80861**

# We are definitely overfitting which is no surprise considering our strategy uses surnames from the training set to link family members. But looking at our coefficients, we have a strong model.

# In[ ]:


# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(X.columns)
coeff_df.columns = ['Features']
classifier = LogisticRegression()
coeff_df["Coefficient Estimate"] = pd.Series(classifier.fit(X, y).coef_[0])

# preview
coeff_df


# ### Where to go from here
# Connecting families to females who died or males who survived is a powerful pattern that gave our score a nice boost. Can we increase it further? Since we're using surname to link these passengers, we could look at the test set for more surnames. However, how to go about it deserves discussion beyond this notebook 

# ### Finally
# Here's a gif rendering our confusion matrix's progress as we constructed our model.
# 
# ![Titanic confusion matrix][1]
# 
# Good luck!
# 
# [1]: http://i.imgur.com/mgxOomU.gif
