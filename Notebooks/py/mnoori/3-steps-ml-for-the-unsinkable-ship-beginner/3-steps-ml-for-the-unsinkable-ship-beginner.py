#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Titanic: The Unsinkable Ship! Jack B. Thayer, one of the Titanic survivors said:
# > "There was peace and the world had an even tenor to it's way. Nothing was revealed in the morning the trend of which was not known the night before. It seems to me that the disaster about to occur was the event that not only made the world rub it's eyes and awake but woke it with a start keeping it moving at a rapidly accelerating pace ever since with less and less peace, satisfaction and happiness. To my mind the world of today awoke April 15th, 1912." 
# 
# I have always been fascinated with Machine Learning techniques. To me, it sounds interesting to be able to predict the chance of survival of anybody on that ship by analyzing their attributes. And that's I'm about to start. This analysis is in three separate parts. I tried to explain steps as much as possible.
# 
# **Please note,** I am new to Kaggle, and I am working on my Data Science skills. However, I'd like to think systematically when it comes to solving problems. Therefore, if you have any ideas about the way I think, please feel free to leave a comment. I do appreciate it!
# 
# 
# # Part 1: Data cleaning and fitting our first model
# 
# In this part, I will start the process of data cleaning. As you will see, I have selected Age, Sex, and Pcalss as my primary variables in this part of analysis. Eventually, I wil fit a model to see how things look. 
# In the next part of analysis, I will explain feature engineering, and eventually I will try to tweak the fitted model to reach to the maximum possible accuracy.

# In[ ]:


#importing modules and reading the train and test sets.
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings("ignore")


train=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/test.csv')

print('Shape of train set is:',train.shape)
print('Shape of test set is:',test.shape)
train.head()


# ### Background research
# 
# Background research is very very important. Getting some domain knowledge about the incident, I realized that women and kids were given priority in lifeboats. Also, there has been three classes among passengers. Therefore, sex, age, and class could be good variables to look at.

# In[ ]:


#using pivot tables to get an idea how much of each sex survived.
sex_pivot=train.pivot_table(index='Sex',values='Survived')
sex_pivot


# **74%** of female survived. It means, according to [this post on Kaggle Kernels](https://www.kaggle.com/pliptor/how-am-i-doing-with-my-score/notebook), if we predicted all women survive and all men parished, we would at least get an accuracy of 0.76. 

# In[ ]:


#pivot table of Pclass
class_pivot=train.pivot_table(index='Pclass',values='Survived')
class_pivot


# In[ ]:


train.Age.describe()


# Age has missing values (compare 714 with 891 shape of train set). Age is fractional for kids younger than one year (we know this by looking at dataset description). Therfore, unlike sex and pclass which are categorical variables, age should be treated differently. 
# One way to look at continous data is through lens of histograms:

# In[ ]:


survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
fig, ax=plt.subplots(figsize=(8,6))
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


# Red surpasses blue more in younger ages. In order to convert age to categorical items, we will cut the Age column in categories. But we should not forget about the null values. Let's fill nulls with -0.5 and use pandas.cut() function for cutting the Age column.
# **Please note,** whatever changes to dataset we make on train set, we should do it test set as well.

# In[ ]:


#I define a function, so I can re-use it on test set as well.
def cut_age(df,cut_limits,label_names):
    df['Age']=df['Age'].fillna(-.5)
    df['Age_cats']=pd.cut(df['Age'],cut_limits,labels=label_names)
    return df

cut_limits=[-1,0,5,12,18,35,60,100] #These limits are something to alter in the future
label_names=['Missing','Infant','Child','Teenager','Young Adult','Adult','Senior']

#we defined a function to apply to both train and test sets.
train=cut_age(train,cut_limits,label_names)
test=cut_age(test,cut_limits,label_names)

train.pivot_table(index='Age_cats',values='Survived').plot.bar()


# Okay! now we have Age as a categorical variable.
# 
# Machine learning algorithms handle numerical variables better than text. For the Pcalss, although the unique values are 1,2,3, these are not mathematically related, meaning a class 2 is not worth two times of class 1, for instance. 
# 
# We will use pd.get_dummies() function to create dummy variables.

# In[ ]:


#again, defining a function in order to be able to reuse on test set.
def create_dummies(df,col_name):
    dummies=pd.get_dummies(df[col_name],prefix=col_name)
    df=pd.concat([df,dummies],axis=1)
    return df
train = create_dummies(train,"Pclass")
test = create_dummies(test,"Pclass")
train = create_dummies(train,"Age_cats")
test = create_dummies(test,"Age_cats")
train = create_dummies(train,"Sex")
test = create_dummies(test,"Sex")

#let's see how our columns look now:
train.columns


# ## Fitting our first model
# 
# The goal is to predict whether a passenger survives or not. This is a binary classification problem. One way to approach these problems is using Logistic Regression models. These models are easy to implement (using Scikit-learn), and can be meaningfully intrepreted.
# 
# Let's fit a logistic regression model for now, and see what insights could we draw from it.

# In[ ]:


#importing LogiticRegression class from sklearn
from sklearn.linear_model import LogisticRegression

#Although we have a test set, but that is only for submission purposes. We should still split our train set into...
#...two seperate sets. This helps us measure the accuracy of our model.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

lr=LogisticRegression()

#Let's only select sex, pclass, and age related columns.
features=['Pclass_1','Pclass_2', 'Pclass_3','Age_cats_Missing', 'Age_cats_Infant','Age_cats_Child', 'Age_cats_Teenager', 
          'Age_cats_Young Adult','Age_cats_Adult', 'Age_cats_Senior', 'Sex_female', 'Sex_male']
target='Survived'

all_X=train[features]
all_y=train[target]

#we will hold out our original test model
holdout=test

#splitting train set into two seperate sets. I use 80% of data for training and 20% for testing.
train_X,test_X,train_y,test_y=train_test_split(all_X,all_y,test_size=0.2,random_state=0)

#let's now fit the model and make predictions.
lr.fit(train_X,train_y)
predictions=lr.predict(test_X)

#calculating accuracy using sklearn function
accuracy=accuracy_score(test_y,predictions)
print('Accuracy of model is {0:.2f} percent'.format(accuracy*100))


# Well, well, well. Seems like we did a good job, right? But wait! Isn't this a little too high for a first time model fitting? We should check if our model is overfitting or not by cross validation?
# 
# What is overfitting overall? It means how optimistic our model is. Although hoping for the passenger's survivals are helpful in this case, we don't want to be too much hopeful!
# 
# ## Cross validation

# In[ ]:


#I use the cross validation score function of sklearn.
from sklearn.model_selection import cross_val_score
import numpy as np

lr=LogisticRegression()
scores=cross_val_score(lr,all_X,all_y,cv=10) #10 folds
accuracy=np.mean(scores)
print(scores)
print('Cross-validated accuracy of model is {0:.2f} percent'.format(accuracy*100))


# That's actually very close! Therefore, our cross validated model shows similar accuracy level to our original model. 
# 
# Let's now train the model on the original train set, and test it on the original test set.

# In[ ]:


lr=LogisticRegression()
lr.fit(all_X,all_y)
holdout_predictions=lr.predict(holdout[features])


# ### Submission file
# 
# What I'm about to here? I will submit these results and see when I stand.

# In[ ]:


holdout_ids=holdout['PassengerId']
submission_df={
    'PassengerId':holdout_ids,
    'Survived':holdout_predictions,
                  }

submission=pd.DataFrame(submission_df)
submission_file=submission.to_csv('TitanicSubmission.csv',index=False)


# Okay! I just submitted and it seems like our actual accuracy is around %75...well, let's work on that to improve our model. There are two ways we could do that:
# * Improving our features
# * Improving our model itself
# 
# In the next section, I will explain how can we work on our features and improve our model.
# 
# 
# # Part 2: Feature Engineering
# 
# ## Feature selection
# Selecting proper features is very important. By feature selection helps to exclude those that are not related to those that are realted to each other.
# 
# Let's see if we can find meaningful features to include in our analysis. As you noticed, we only included sex, age, and calss so far. We will use DafaFrame.describe() method to find meaningful information on values in the remaining columns.

# In[ ]:


cols=['SibSp','Parch','Fare','Cabin','Embarked']
train[cols].describe(include='all',percentiles=[])


# AMong these features, `Cabin` has a lot of missing values (only 204 out of 891 observations). And it seems like most of the values are unique. Therefore, we will drop this column for now.
# `Embarked` seems like a categorical column, has only 3 unique values, and 2 missing values. We can simply replace these two with most frequent value in this column, `S`, which has repeated 644 times!
# The rest of columns seem like regular numerical features with no missing values. However, the range of `SibSp` and `Parch` are different than `Fare`. Therefore, we would need to **rescale** these columns so we give them equal weights in our model fitting.

# In[ ]:


train['Embarked']=train['Embarked'].fillna('S')
#As you remember, whatever we do on train data set, we will do the same on test (holdout).
holdout['Embarked']=holdout['Embarked'].fillna('S')


#holdout has one missing value in Fare columns, let's replace it with mean of that column.
holdout['Fare']=holdout['Fare'].fillna(train['Fare'].mean())

holdout[cols].describe(include='all',percentiles=[])


# In[ ]:


#creating dummy variables for Embarked
train = create_dummies(train,"Embarked")
holdout = create_dummies(holdout,"Embarked")


# For rescaling the numerical columns, I will use sklearn's minmax_scale function.

# In[ ]:


#rescaling the numerical columns
from sklearn.preprocessing import minmax_scale
cols=['SibSp','Parch','Fare']
for col in cols:
    train[col + "_scaled"] = minmax_scale(train[col])
    holdout[col + "_scaled"] = minmax_scale(holdout[col])


# In[ ]:


train.columns


# After rescaling, we can fit a new model with all the features and try to select the most important variables. We can do this by sorting out the coefficients in Logistic Regression algorithm.

# In[ ]:


columns=['Pclass_1', 'Pclass_2', 'Pclass_3', 'Age_cats_Missing', 'Age_cats_Infant',
       'Age_cats_Child', 'Age_cats_Teenager', 'Age_cats_Young Adult',
       'Age_cats_Adult', 'Age_cats_Senior', 'Sex_female', 'Sex_male',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',
       'Parch_scaled', 'Fare_scaled']
       
lr=LogisticRegression()
lr.fit(train[columns],train['Survived'])

#finding the coefficients
coeffs=lr.coef_
importance_of_features=pd.Series(coeffs[0],index=train[columns].columns).abs().sort_values()
importance_of_features.plot.barh()


# Well, great! We could rank the features based on their coefficients in the model. However, we should forget about **Collinearity**. Let's form the heatmap of correlation matrix.
# But before doing that, let's finalize our feature enegineering process. We can also work with Fare columns, to convert it into a categorical item. This approach is called **Bining**. It is similar to what we did fo age in first part of analysis.
# 
# Let's first take a look at the histogram of Fare column.

# In[ ]:


train['Fare'].hist(bins=20,range=(0,100))


# In[ ]:


# defining a function for binning fare column
def process_fare(df,cut_points,lebel_names):
    df['Fare_cats']=pd.cut(df['Fare'],cut_points,labels=label_names)
    return df

cut_points=[0,12,50,100,1000]
label_names=['0-12','12-50','50-100','100+']

#cutting the fare column using our function
train=process_fare(train,cut_points,label_names)
holdout=process_fare(test,cut_points,label_names)

#creating dummy columns:
train=create_dummies(train,'Fare_cats')
holdout=create_dummies(test,'Fare_cats')


# Now, let's take a look at the name column in more detail.

# In[ ]:


train[['Name','Cabin']].head(10)


# We see a trend of Mr., Mrs, Miss, etc. By looking at the entire records, we can come up with some unique titles, listed in the below dictionary. 
# How can we parse these titles from the name column? The answer is the `extract` method of dataframe! We will also use `regular expressions`. Please refer to [this](www.regex101.com) amaizng website when you have questions how a regex syntaxt will end up in your code.
# 
# We can also see that the first character of `Cabin` column could be a categorical item, let's extract that one as well.

# In[ ]:


#creating a mapping dictionary
titles={
    "Mr" :         "Mr",
    "Mme":         "Mrs",
    "Ms":          "Mrs",
    "Mrs" :        "Mrs",
    "Master" :     "Master",
    "Mlle":        "Miss",
    "Miss" :       "Miss",
    "Capt":        "Officer",
    "Col":         "Officer",
    "Major":       "Officer",
    "Dr":          "Officer",
    "Rev":         "Officer",
    "Jonkheer":    "Royalty",
    "Don":         "Royalty",
    "Sir" :        "Royalty",
    "Countess":    "Royalty",
    "Dona":        "Royalty",
    "Lady" :       "Royalty"    
}

def titles_cabin_process(df):
    #extracting titles from 'Name' column
    extracted_titles=df['Name'].str.extract(' ([A-Za-z]+)\.',expand=False)
    df['Title']=extracted_titles.map(titles)
    
    #extracting first letter of 'Cabin' column
    df['Cabin_type']=df['Cabin'].str[0]
    df['Cabin_type']=df['Cabin_type'].fillna('Unknown')
    
    #creating dummy variables
    df=create_dummies(df,'Title')
    df=create_dummies(df,'Cabin_type')
    
    return df
    
train=titles_cabin_process(train)
holdout=titles_cabin_process(holdout)


# In[ ]:


train.columns


# #### Checking for Collinearity
# 
# I will do that by looking at the correlation matrix of our features. We will use Seaborn's heatmap for this.

# In[ ]:


#writing a function that graphs nice heatmaps
def plot_corr_heatmap(df):
    import seaborn as sns
    corrs=df.corr()
    sns.set(style='white')
    mask=np.zeros_like(corrs,dtype=np.bool)
    mask[np.triu_indices_from(mask)]=True
    
    f,ax=plt.subplots(figsize=(11,9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corrs, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


# Let's create a column list from the features that we think are ready to go into model fitting process.

# In[ ]:


ready_columns=['Pclass_1',
       'Pclass_2', 'Pclass_3', 'Age_cats_Missing', 'Age_cats_Infant',
       'Age_cats_Child', 'Age_cats_Teenager', 'Age_cats_Young Adult',
       'Age_cats_Adult', 'Age_cats_Senior', 'Sex_female', 'Sex_male',
       'Embarked_C', 'Embarked_Q', 'Embarked_S', 'SibSp_scaled',
       'Parch_scaled', 'Fare_cats_0-12',
       'Fare_cats_12-50', 'Fare_cats_50-100', 'Fare_cats_100+',
        'Cabin_type_A', 'Cabin_type_B', 'Cabin_type_C',
       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',
       'Cabin_type_T', 'Cabin_type_Unknown', 'Title_Master', 'Title_Miss',
       'Title_Mr', 'Title_Mrs', 'Title_Officer', 'Title_Royalty']

plot_corr_heatmap(train[ready_columns])


# Beautiful! What we can easily realize is that sex_female has a lot of correlation with sex_female! it is comething called `dummy variable trap`, becuase we created complimentary set of variables, setting the value of one to True, the other one becomes False automatically. Therefore, let's remove these collinear columns. but let's remove the ones with the least amount of data.

# In[ ]:


final_ready_columns=['Pclass_1','Pclass_3', 'Age_cats_Missing', 'Age_cats_Infant',
       'Age_cats_Child', 'Age_cats_Teenager', 'Age_cats_Young Adult',
       'Age_cats_Adult', 'Embarked_C', 'Embarked_S', 'SibSp_scaled',
       'Parch_scaled', 'Fare_cats_0-12', 'Fare_cats_12-50', 'Fare_cats_50-100',
        'Cabin_type_A', 'Cabin_type_B', 'Cabin_type_C',
       'Cabin_type_D', 'Cabin_type_E', 'Cabin_type_F', 'Cabin_type_G',
       'Cabin_type_Unknown', 'Title_Master', 'Title_Miss',
       'Title_Mr', 'Title_Mrs', 'Title_Officer']


# Also, instead of fitting a model again, sorting the features, and showing them on a barchart, we will use `recursive feature elemination` with cross-validation, using sklearn's `RFECV` class.

# In[ ]:


from sklearn.feature_selection import RFECV

all_X=train[final_ready_columns]
all_y=train['Survived']

lr=LogisticRegression()

#just like any other sklearn class, we will instantiate the class first, then fit the model
selector=RFECV(lr,cv=10)
selector.fit(all_X,all_y)

#usuing RFECV.support_ we can find the most import features. It provides a boolean list.
optimized_columns=all_X.columns[selector.support_]
optimized_columns


# Now let's fit the model and see the score.

# In[ ]:


all_X = train[optimized_columns]
all_y = train["Survived"]
lr=LogisticRegression()
scores=cross_val_score(lr,all_X,all_y,cv=10)
accuracy=scores.mean()
print('Cross-validated accuracy of model is {0:.2f} percent'.format(accuracy*100))


# It's s decent improvement compared to our last try!

# ## Part 3: Model selection and tuning
# 
# Until now, we have been working with Logistoc Regression approach. Let's use a different algorithm now. 
# `random forest` perfomrs well in non-linear situations. Let's see if we can have a better acuracy with that algorithm.
# 
# We will perform a grid search in hyper paramter optimization.

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

hyperparameters = {"criterion": ["entropy", "gini"],
                   "max_depth": [5, 10],
                   "max_features": ["log2", "sqrt"],
                   "min_samples_leaf": [1, 5],
                   "min_samples_split": [3, 5],
                   "n_estimators": [6, 9]
}

clf = RandomForestClassifier(random_state=1)
grid = GridSearchCV(clf,param_grid=hyperparameters,cv=10)

grid.fit(all_X, all_y)

best_params = grid.best_params_
best_score = grid.best_score_

print('Cross-validated accuracy of model is {0:.2f} percent'.format(best_score*100))


# ------------------------------Thank you--------------------------
# 
# I would love to hear your thoughts.
# Also, These steps are aligned with what has been done at [Dataquest](dataquest.io)
