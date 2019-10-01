#!/usr/bin/env python
# coding: utf-8

# ### In this notebook we study Titanic survival data (from [Kaggle](https://www.kaggle.com/c/titanic)) and try to predict who will survive and who won't. 

# #### This notebook is organized as follows:
# 
# 1. Data analysis and visualization
# 2. Predictions

# ### 1. Data Analysis and visualization
# 
# #### As usual, we start by importing the basic libraries and loading the csv data file into a pandas dataframe. After that we will analyze the data thoroughly. 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


titanic_train_data = pd.read_csv('../input/train.csv', index_col='PassengerId')


# In[ ]:


titanic_train_data.info()


# In[ ]:


titanic_train_data['Survived'].value_counts()


# #### To get a better feeling which kind of values the features have in the data, we take a look at the first 50 rows of the dataframe.
# 

# In[ ]:


titanic_train_data.head(50)


# #### Let us check if the Ticket numbers are unique...

# In[ ]:


titanic_train_data['Ticket'].value_counts(normalize=True)


# #### In order to find out which features are significant in predicting whether a passenger survived (having 1 in the 'Survived' column) or not (0) let us inspect some crosstables and their countplots. Since, presumably, for each name there is exactly one passenger and from above we see that ticket numbers are almost one-to-one, we leave Name and Ticket out. On the other hand, cabins are missing from most of the passengers so we will exclude Cabin also. From the known facts of Titanic's story, we expect that the Sex and Pclass will have significant effect on one's survival.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Pclass'])


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Pclass vs. Survival")
sns.countplot(titanic_train_data['Pclass'], hue=titanic_train_data['Survived'])


# #### As expected, the higher the class (i.e., the lower the Pclass number), the better is the survival rate. Since the correspondence is linear, there is no need for changing the numeric values or one-hot-encoding at this point (*).
# 
# ##### (*) It could be that Pclass together with some other feature (e.g. Sex) doesn't have linear correspondence to survival, but for the sake of simplicity we ignore this possibility at least for now.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Sex'])


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Sex vs. Survival")
sns.countplot(titanic_train_data['Sex'], hue=titanic_train_data['Survived'])


# #### It really does seem that the sex of the passenger is a strong indicator for one's survival. However, we must change the categorical values (male, female) to numerical ones. We will do this later.

# #### There is no point in analyzing Age with all different values separately, so let's divide the passengers to somewhat suitable bins by their age for the crosstab.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], pd.cut(titanic_train_data['Age'], bins=[0,16,30,45,60,80]))


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Age vs. Survival")
sns.countplot(pd.cut(titanic_train_data['Age'], bins=[0,16,30,45,60,80]), hue=titanic_train_data['Survived'])


# #### As we see, more than half of the children (16 years or less), around one third from young adults, around 40% of both adults (30-45 yrs) and middle-aged (45-60yrs), and around one fifth from older people (60-80 yrs) survived. In the data set there seems to be a couple of hundred people with unknown age, but since we don't want to drop Age out, we will impute the data later.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['SibSp'])


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Having siblings/spouse vs. Survival")
sns.countplot(titanic_train_data['SibSp'], hue=titanic_train_data['Survived'])


# #### As the data suggests, most of the passengers travelled without their siblings/spouse and quite a few had more than 1 sibling/spouse. Let's take a closer look with 3 bins (0, 1 or more siblings/spouse).

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], pd.cut(titanic_train_data['SibSp'], bins=[-1,0,1,8]))


# #### So, having 1 sibling/spouse on board increases the changes of survival but travelling without them (as perhaps most young men did) or having 2 or more siblings/spouse decreases it. Maybe this feature together with Age would tell more. Lets see next if having parents or children on board has an impact to survival.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Parch'])


# #### Since a vast majority of passengers travels without parents or children, we will divide the passengers to 2 bins in order to have a better idea of the impact of having family onboard.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], pd.cut(titanic_train_data['Parch'], bins=[-1,0,6]))


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Having parents/children vs. Survival")
sns.countplot(pd.cut(titanic_train_data['Parch'], bins=[-1,0,6]), hue=titanic_train_data['Survived'])


# #### There is power in the family, however, we must notice that the difference in the sizes of these 2 bins is noticeable.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], pd.qcut(titanic_train_data['Fare'], 4))


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Fare vs. Survival")
sns.countplot(pd.qcut(titanic_train_data['Fare'], 4), hue=titanic_train_data['Survived'])


# #### Dividing feature Fare to quantiles shows clearly that the higher the price of the ticket, the better chance the passenger has for surviving.

# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Embarked'])


# In[ ]:


plt.figure(figsize=(15,5))
plt.title("Embarked vs. Survival")
sns.countplot(titanic_train_data['Embarked'], hue=titanic_train_data['Survived'])


# #### What comes to the place of embarking, it looks like C has been the best place to hop onboard. Note, however, that most passengers came from S.

# #### One could easily think that the name of the passenger does not have any effect on his/hers survival. Nevertheless, the (relationship) status might have, and therefore we will take a closer look to the feature 'Name'. Indeed, at a glimpse, it looks like all passengers have a title (Mr./Mrs./Master./Miss./Don.). Recall that "Master." was used for underage boys.

# In[ ]:


titanic_train_data['Name'].head(50)


# #### For digging out the title from the names, we will define some functions and add columns which tell whether (1=true, 0=false) the passenger has certain title. The end result corresponds pretty much one-hot-encoding.

# In[ ]:


def titleMrs(name):
    return 1*("Mrs." in name)

def titleMiss(name):
    return 1*("Miss." in name)

def titleMr(name):
    return 1*("Mr." in name)

def titleMaster(name):
    return 1*("Master." in name)

def titleDon(name):
    return 1*("Don." in name)


# In[ ]:


titanic_train_data['TitleMrs.']=titanic_train_data['Name'].apply(titleMrs)
titanic_train_data['TitleMiss.']=titanic_train_data['Name'].apply(titleMiss)
titanic_train_data['TitleMr.']=titanic_train_data['Name'].apply(titleMr)
titanic_train_data['TitleMaster.']=titanic_train_data['Name'].apply(titleMaster)
titanic_train_data['TitleDon.']=titanic_train_data['Name'].apply(titleDon)


# In[ ]:


titanic_train_data.head(10)


# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMrs.'])


# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMr.'])


# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMiss.'])


# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMaster.'])


# In[ ]:


pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleDon.'])


# #### There is only 1 person having title Don. so we can just as well drop the TitleDon column away.

# In[ ]:


titanic_train_data.drop('TitleDon.', axis=1, inplace=True)


# #### So, our target is 'Survived' and for the model(s) we will drop the columns 'Name', 'Ticket' and 'Cabin' altogether from the data. Since there are not too many features left (12 to be exact), we will keep them at least at this point and, if necessary, see more closely correlation values later.

# In[ ]:


y=titanic_train_data.Survived
X=titanic_train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


X.head(30)


# #### Our dataframe looks pretty good at this point, however, we must change the categorical values in Sex to numerical ones and one-hot-encode the Embarked column. We could also one-hot-encode Sex, but since there are only 2 possible values, we can just as well replace 'female' by 1 and 'male' by 0. Let's not forget that we must impute the data (since there are lots of missing values in Age) but we'll get back to this later.

# In[ ]:


X.loc[X.Sex == 'male', 'Sex'] = 0
X.loc[X.Sex == 'female', 'Sex'] = 1


# In[ ]:


X.head(20)


# In[ ]:


X_one_hot = pd.get_dummies(X)


# In[ ]:


X_one_hot.head(50)


# #### For final feature selection we will check the mutual infos and the correlations.

# In[ ]:


from sklearn.feature_selection import mutual_info_classif


# In[ ]:


# We have to drop Age temporarily just for mutual info since it has NaN values.
X_one_hot.columns.drop('Age')


# In[ ]:


np.round(mutual_info_classif(X_one_hot.drop('Age', axis=1), y, discrete_features=True),2)


# In[ ]:


dict(zip(X_one_hot.columns.drop('Age'),
         np.round(mutual_info_classif(X_one_hot.drop('Age', axis=1), y, discrete_features=True),2)))


# #### From mutual information (see more info from [Wikipedia](https://en.wikipedia.org/wiki/Mutual_information)) we see that Embarked_Q and TitleMaster are the least significant and Age and TitleMr. the most. Below we see the correlations from each feature to the target separately.

# In[ ]:


df = X_one_hot.copy()


# In[ ]:


df['Survived'] = titanic_train_data['Survived']


# In[ ]:


df.corr()['Survived'].sort_values()


# #### If we look at the impact of each feature to the target, the features SibSp and Embarked_Q have the smallest correlation. These features have also really small impact given by the mutual information so we will drop these two as well as TitleMaster and Parch.

# In[ ]:


X_one_hot.drop(['SibSp', 'Embarked_Q', 'TitleMaster.', 'Parch'], axis=1, inplace=True)


# #### Before imputing the data, we will split the data to train and test sets. This is done in this order to avoid any leakage from the test set to train set since Imputer uses mean as a default strategy.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_one_hot, 
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=2)


# In[ ]:


from sklearn.preprocessing import Imputer


# In[ ]:


my_imputer = Imputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_train.columns = X_one_hot.columns

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X_test.columns = X_one_hot.columns


#  ### 2. Predictions
#  
#  #### In this section we predict the survivors and analyze our model in terms of its reliability.

# #### Since this is a classification problem, we will use Random Forest Classifier for our model.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=None, random_state=2)
rf.fit(imputed_X_train,y_train)


# In[ ]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


pred = rf.predict(imputed_X_test)


# In[ ]:


accuracy_score(y_test, pred)


# #### The accuracy score with this model is around 80%. One reason for our model not to perform better is perhaps that the data set is relatively small. Also, one must take into account that in this type of case/scenario there is certainly some natural randomness whether person survives or not. Next we see how well our model predicts one's probability to survive.

# In[ ]:


pred_proba = rf.predict_proba(imputed_X_test)[:,1]


# In[ ]:


df_results = pd.DataFrame()


# #### We are interested in passengers' probabilities to survive. Naturally we expect that for those who did not survive, also the probability to survive is small.

# In[ ]:


df_results['Survived'] = y_test
df_results['PredProba'] = np.round(pred_proba,4)


# In[ ]:


df_results[df_results.Survived==0]['PredProba']


# #### Below we see a histogram of survival probabilities for all passengers.

# In[ ]:


plt.figure(figsize=(15,5))
plt.title('Histogram of survival probabilities')
plt.ylabel('Number of passengers')
sns.distplot(df_results[df_results.Survived==0]['PredProba'], kde=False, bins=10, label='Deceased')
sns.distplot(df_results[df_results.Survived==1]['PredProba'], kde=False, bins=10, label='Survived')
plt.legend(loc=1)


# #### Let us check the confusion matrix and try to analyze where the model did wrong.

# In[ ]:


confusion_matrix(y_test, pred)


# #### From the confusion matrix we see that in the test set there were 145 true negatives (not survived and predicted to not survive), 15 false positives (not survived but predicted to survive), 34 false negatives and 74 true positives.

# #### Let us see if there are any common features in the cases where our model did not succeed to predict right. We will do this by creating a data frame by copying X_test and adding columns y_test and pred to it. Then we will drop the rows where the prediction went right.

# In[ ]:


X_comparison = X_test.copy()


# In[ ]:


X_comparison['Survived'] = y_test
X_comparison['Predicted'] = pred


# In[ ]:


X_comparison


# In[ ]:


Wrong_predicted = X_comparison[X_comparison.Survived != X_comparison.Predicted]
Wrong_predicted


# #### At a first glimpse, it looks like there is no clear similarities between the passengers gotten wrong prediction.  Let's see how reliable our model is by using cross validation.  Since cross validation splits the data to different train and test sets each time and does not accept NaN values, we must impute the whole data first.

# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


imputed_X_one_hot = pd.DataFrame(my_imputer.fit_transform(X_one_hot))
imputed_X_one_hot.columns = X_one_hot.columns


# In[ ]:


scores = cross_val_score(rf, imputed_X_one_hot, y, cv=5, scoring='accuracy')


# In[ ]:


scores


# #### The cross validation scores correspond pretty much in average our accuracy score which means that our model does not over/underfit. To improve our model we could test different hyperparameters in our random forest classifier. Also survival probabilities could be analyzed more carefully, for instance, by adjusting the threshold.
