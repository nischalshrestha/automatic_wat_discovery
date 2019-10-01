#!/usr/bin/env python
# coding: utf-8

# # Beginning in Kaggle with Titanic (Ensemble)
# ### **André Koeppl**
# 
# 
#     

# ## 1. Introduction
# 
# This is my first kernel at Kaggle. I got inspiration for this kernel from the folowing kernels:
# * Yassine Ghouzam, PhD: "Titanic Top 4% with ensemble modeling
# *  LD Freeman, from his kernel: "A Data Science Framework: To Achieve 99% Accuracy". 
# 
# Thank you a lot guys for your willingness to share knowledge.
# I hope I can help some people to have more insights and receive critics ans suggestions. They will be welcome!
#  

# In[ ]:


#Importing initial libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

#including other classifiers
from sklearn.naive_bayes import GaussianNB
from mlxtend.classifier import StackingClassifier # for Stacking Ensembles

sns.set(style='white', context='notebook', palette='deep')


# ## 2. Load and check data
# ### 2.1 Load data

# In[ ]:


# Load data
##### Load train and Test set
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
IDtest = df_test["PassengerId"]
df_train.head()


# In[ ]:


print(df_train.shape)
print(df_test.shape)


# ### 2.2 Joining data frames

# I joint the 2 data frames ensemble to assure that the dummy variables I am going to create are the same for both data frames

# In[ ]:


## Join train and test datasets in order to obtain the same number of features during categorical conversion
train_len = len(df_train)
dataframe =  pd.concat(objs=[df_train, df_test], axis=0).reset_index(drop=True)
dataframe.tail()


# ## 3. Filling Missing values

# ### 3.1 Simple missings substitution

# In[ ]:


#Checking missing values
print(dataframe.isnull().sum())


# I am going to use the following strategy to treat the missing variables:
# 
# * Fare missing values are little representative in the data frame so, lets fulfill them with the median of the df_train 
# * Cabin missing values are very representative, so they will be fulfiled with a prediction from a multonomial model
# * Embarked missing  values are little representative in the data frame so, lets fulfill them with the most frequent value
# * Age missing values are very representative in this dara frame so I am creating a model to better predict values to fulfill the NaNs

# In[ ]:


Fare_median = df_train.Fare.median()
dataframe.loc[dataframe.Fare.isnull(),'Fare'] = Fare_median


# Lets check the categorical variable Embarked.

# In[ ]:


g = sns.factorplot("Embarked",  data=dataframe,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Since S is the most frequent Embarkation port, let's assume the missing values to be S

# In[ ]:


dataframe.loc[dataframe.Embarked.isnull(),'Embarked'] = 'S'


# ### 3.2 Missing substitution by a predicted model

# Let's take a look  at the NaNs of the categorical variable Cabin:

# In[ ]:


print("% of NaN values in Cabin Variable for train test:")
print(dataframe.Cabin.isnull().sum()/len(dataframe))


# Very representative.

# In[ ]:


# Replace the Cabin categorical variable for its first letter
dataframe["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else np.nan for i in dataframe['Cabin']])

g = sns.factorplot("Cabin",  data=dataframe,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# Multinomial logistic regression doen't accept one of its nominal levels to have just one observation. Let's check "T".

# In[ ]:


dataframe[dataframe.Cabin=='T']


# Let's check the most frequent level too

# In[ ]:


dataframe[dataframe.Cabin=='C'].head()


# Replacing "T" by the most frequent level "C"

# In[ ]:


dataframe.loc[dataframe.Cabin=='T',"Cabin"]='C'
g = sns.factorplot("Cabin",  data=dataframe,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")


# So, let's train a model for 'Cabin'

# In[ ]:


#Splitting the base in two:
df_Cabin_not_NaN = dataframe[dataframe.Cabin.notnull()]
df_Cabin_NaN = dataframe[dataframe.Cabin.isnull()]

#Splitting the dataframe in X and Y
X_train_Cabin = pd.concat([df_Cabin_not_NaN[['SibSp', 'Parch', 'Pclass','Fare']]], axis=1)
y_train_Cabin = df_Cabin_not_NaN.Cabin

# Importing the libraries
from sklearn import linear_model

# Train multinomial logistic regression model
mul_lr = LogisticRegression(multi_class='multinomial',
                                         solver='newton-cg',max_iter = 100) .fit(X_train_Cabin, y_train_Cabin)

scores = cross_val_score(mul_lr, X_train_Cabin, y_train_Cabin, cv=5, scoring='accuracy')
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std()))


# Then let's run the prediction.

# In[ ]:


#Splitting the dataframe in X, using the same variables we used to train the model
X_train_Cabin_Nan =  pd.concat([df_Cabin_NaN[['SibSp', 'Parch', 'Pclass','Fare']]], axis=1)

# Predict y
mul_lr.fit(X_train_Cabin, y_train_Cabin)
y_train_Cabin_Nan = mul_lr.predict(X_train_Cabin_Nan)

#Checking results
auxiliar = pd.DataFrame(y_train_Cabin_Nan,columns=['Cabin'])

g = sns.factorplot("Cabin",  data=auxiliar,
                   size=6, kind="count", palette="muted")
g.despine(left=True)
g = g.set_ylabels("Count")   


# In[ ]:


#Rebuilding the dataset
df_Cabin_NaN.loc[:,'Cabin']= y_train_Cabin_Nan
dataset = pd.concat([df_Cabin_NaN,df_Cabin_not_NaN], axis =0).reset_index(drop=True)
dataset.shape


# Let's check the variable 'Age'

# In[ ]:


print("% of NaN values in Age Variable for train test:")
print(dataset.Age.isnull().sum()/len(dataset))


# The remaing variable is Age and it is considerably representative in the dataset. Let's take a look in a correlation heat map to understant it better.

# In[ ]:


# Correlation matrix between numerical values (SibSp Parch Age and Fare values) and Survived 
g = sns.heatmap(df_train[["Survived","SibSp","Parch","Age","Fare"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")


# So, Age is more correlated with SibSp and Parch, but one os my guess is it maybe be correlated with the categorical variables as well. Let's understand and treat the variable Name.

# In[ ]:


#Lets take a look into the name's variable
dataset.Name.head()


# Let's separate the pronoum of treatment from de data

# In[ ]:


dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)

j = sns.countplot(x="Title",data=dataset)
j = plt.setp(j.get_xticklabels(), rotation=45) 


# In[ ]:


# Exploring Survival probability
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar", size = 10 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Captain has not survived. The major and colonel are military titles and have ~50% chance of survival (with high dipsersion). It seems relevant to group possible dynamics of profession, crew membership, nobility title, and so on..

# In[ ]:


# Let's group the military and crew titles ensemble
dataset["Title"] = dataset["Title"].replace(['Capt', 'Col','Major'], 'Crew/military')
# Let's group the profession title ensemble
dataset["Title"] = dataset["Title"].replace(['Master', 'Dr'], 'Prof')
# Let's change the type of women title
dataset["Title"] = dataset["Title"].replace(['Miss', 'Mme','Mrs','Ms','Mlle'], 'Woman')
# Let's change the type of nobility title
dataset["Title"] = dataset["Title"].replace(['the Countess', 'Sir', 'Lady', 'Don','Jonkheer', 'Dona'], 'Noble')
# Let's change the type of religious title
dataset["Title"] = dataset["Title"].replace(['Rev'], 'Religious')

j = sns.countplot(x="Title",data=dataset)
j = plt.setp(j.get_xticklabels(), rotation=45) 


# 

# In[ ]:


# Exploring Survival probability
g = sns.factorplot(x="Title",y="Survived",data=dataset,kind="bar", size = 6 , 
palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")


# Clearly we understand that they prioitize Woman, followed by the Nobles and then Professional titles. We can't make a clear conclusion regaring nobility titles because of the high sigma, but I bet it's because of the percentage of men in this group. Obviously Religious title make the moral decision in this case. Regarding officers, the dispersion indicates part of the crew decided to remain onboard and others went on the survival boats. Ordinary men had the least chance of survival.

# Age: let's create a simple model to predict values for the fullfilment of the NaNs

# In[ ]:


# Creating the Dummies
dataset = pd.concat([dataset, pd.get_dummies(dataset[['Sex', 'Embarked','Title']])], axis=1)
dataset.head()


# In[ ]:


#Splitting the base in two:
df_Age_not_NaN = dataset[dataset.Age.notnull()]
df_Age_NaN = dataset[dataset.Age.isnull()]

#Splitting the dataframe in X and Y
X_train = pd.concat([df_Age_not_NaN[['SibSp', 'Parch', 'Pclass','Fare','Sex_female','Sex_male',
                                     'Embarked_C', 'Embarked_Q', 'Embarked_S','Title_Crew/military',
                                     'Title_Mr','Title_Noble','Title_Prof','Title_Religious',
                                     'Title_Woman']]], axis=1)
y_train = df_Age_not_NaN.Age

# Importing the libraries
from sklearn import linear_model
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

# Initializing models
lr = LinearRegression()
svr_lin = SVR(kernel='linear')
ridge = Ridge(random_state=1)
svr_rbf = SVR(kernel='rbf')
rf = RandomForestRegressor(random_state=1)
knn = KNeighborsRegressor()
gbr = GradientBoostingRegressor()
dtr = DecisionTreeRegressor(max_depth=4)
adr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=1)
                           
for clf, label in zip([lr, svr_lin, ridge, rf, knn,gbr,dtr,adr, svr_rbf], ['Linear regression',
                                                                           'Linear support vector machine',
                                                                           'Ridge','Random Forest',
                                                                           'KNeighbors','Gradient Boosting',
                                                                           'Decision Tress', 'Ada Boosting',
                                                                           'Rbf support vector machine']):
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='r2')
    print("R2: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


# Every model resulted in a R2 higher than zero, so they predict better if I've used an average. So let's take the best estimator, the Gradient Boosting, and refine it using the Grid Search

# In[ ]:


#GBC
params = [{ 'n_estimators' : [25, 50,100,200],
              'learning_rate': [0.5, 0.1, 0.05, 0.01],
              'max_depth': [2, 3, 4],
              'min_samples_leaf': [1, 2, 3],
              'max_features': [None, "auto"]}]

from sklearn.model_selection import GridSearchCV

grid_gbr = GridSearchCV(GradientBoostingRegressor(random_state=41), params, cv=5, scoring = 'r2')
grid_gbr.fit(X_train, y_train)

print ("Gradient Boosting")
print("Best parameters set found on development set:")
print()
print(grid_gbr.best_params_)
print()
print("Best score set found on development set:")
print()
print(grid_gbr.best_score_)


# A little improvement in R2, which for me it's enough to stop tuning the model. Let's apply it to the NaNs in 'Age'

# Predicting missing values...

# In[ ]:


#Splitting the dataframe in X, using the same variables we used to train the model
X_train_Nan = pd.concat([df_Age_NaN[['SibSp', 'Parch', 'Pclass','Fare','Sex_female','Sex_male',
                                     'Embarked_C', 'Embarked_Q', 'Embarked_S','Title_Crew/military',
                                     'Title_Mr','Title_Noble','Title_Prof','Title_Religious',
                                     'Title_Woman']]], axis=1)
#Initializing the best model
from sklearn import ensemble
best_gbr = GradientBoostingRegressor(learning_rate=0.1, max_depth=4, max_features= None, 
                                     min_samples_leaf= 1, n_estimators= 50, random_state=41)                             

# Predict y
best_gbr.fit(X_train, y_train)
y_train_Nan = best_gbr.predict(X_train_Nan)

#Checking results
auxiliar = pd.DataFrame(y_train_Nan,columns=['Age'])
print("Number of Nan: %0.2f" %auxiliar.isnull().sum()) # no Nans
print("-"*10)
print("Min value of y_train_Nan: %0.2f" % y_train_Nan.min())   #We don't have negative 'Age'            


# Let's insert the replaced data from 'Age' in dataset again

# In[ ]:


#Rebuilding the dataset
df_Age_NaN['Age']= y_train_Nan
df = pd.concat([df_Age_NaN,df_Age_not_NaN], axis =0).reset_index(drop=True)



# In[ ]:


#Checking the missings
print(df.isnull().sum())


# So we have cleaned all the missings from the  input variables

# # 4. Feature engineering

# I have engineered Cabin and the Name variables.
# Also I created dummies for  'Sex', 'Embarkation' and 'Title'.
# 
# Now I am going to engineer the variable 'Fsize', which is the combination of 'Parch' and 'SibSp' and  'Ticket' as well, despite my suspition that it has few effect in the surival rate.
# 
# Then I am going to create dummies for them too for 'Cabin' and 'Tiket' and discretize the numerical variables

# Working on 'Cabin'. Let's see it.

# In[ ]:


df.Cabin.head(10)


# And create the dummies for it...

# In[ ]:


df = pd.get_dummies(df, columns = ["Cabin"],prefix="Cabin")


# Let's do the same for 'Ticket'

# In[ ]:


df.Ticket.head(40)


# Let's extract the first letter of the ticket, which may be indicative of group reservations, and whenever it's not possible to find one letter, input 'X' value, as I've done with the 'Cabin' missing treatment.

# In[ ]:


## Treat Ticket by extracting the ticket prefix. When there is no prefix it returns X. 

Ticket = []
for i in list(df.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0]) #Take prefix
    else:
        Ticket.append("X")
        
df["Ticket"] = Ticket
df["Ticket"].head(40)


# In[ ]:


#Generating the dummies under the prefix 'T'
df = pd.get_dummies(df, columns = ["Ticket"], prefix="T")
#plotting a graph
df.head()


# Engineering 'Family' size

# We can imagine that large families will have more difficulties to evacuate, looking for theirs sisters/brothers/parents during the evacuation. So, in the file I forked, the author choosed to create a "Fize" (family size) feature which is the sum of SibSp , Parch and 1 (including the passenger). It's a good idea, let's do it.

# In[ ]:


# Create a family size descriptor from SibSp and Parch
df["Fsize"] = df["SibSp"] + df["Parch"] + 1


# In[ ]:


g = sns.factorplot(x="Fsize",y="Survived",data = df)
g = g.set_ylabels("Survival Probability")


# Some models work better if we 'help' them segmenting the space of the possible levels for the variable. Let's create 4 types of Fsize:
# * Single individuals (1)
# * Couples or Parent-kid, Parent-rellative (2)
# * Medium families (up to 4 individuals)
# * Large Families (5+)
# 

# In[ ]:


# Create new feature of family size
df['Single'] = df['Fsize'].map(lambda s: 1 if s == 1 else 0)
df['SmallF'] = df['Fsize'].map(lambda s: 1 if  s == 2  else 0)
df['MedF'] = df['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
df['LargeF'] = df['Fsize'].map(lambda s: 1 if s >= 5 else 0)

g = sns.factorplot(x="Single",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")


# Now it's time for discretizing numerical variables, like 'Age'

# In[ ]:


# Explore Age vs Survived
g = sns.FacetGrid(df, col='Survived')
g = g.map(sns.distplot, "Age")


# Visually, I am defining segments of 'Age', trying to capture possible dynamics of suvival probability

# In[ ]:


# Create new feature of Age
df['Child'] = df['Age'].map(lambda s: 1 if s<=10 else 0)
df['Young'] = df['Age'].map(lambda s: 1 if  10 < s <= 30   else 0)
df['Senior'] = df['Age'].map(lambda s: 1 if 30 < s <= 40 else 0)
df['Elder'] = df['Age'].map(lambda s: 1 if s > 40 else 0)

g = sns.factorplot(x="Child",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="Young",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="Senior",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="Elder",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")


# It's time for the numerical variable 'Fare':

# In[ ]:


# Explore Fare vs Survived
g = sns.FacetGrid(df, col='Survived', size=10)
g = g.map(sns.distplot, "Fare")


# Applying the same logic for 'Fare'
# 

# In[ ]:


# Create new feature of Fare
df['F_Low'] = df['Fare'].map(lambda s: 1 if s<=50 else 0)
df['F_Med'] = df['Fare'].map(lambda s: 1 if  50 < s <= 100   else 0)
df['F_High'] = df['Fare'].map(lambda s: 1 if 100 < s <= 200 else 0)
df['F_Ultra'] = df['Fare'].map(lambda s: 1 if s > 200 else 0)

g = sns.factorplot(x="F_Low",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="F_Med",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="F_High",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="F_Ultra",y="Survived",data=df,kind="bar")
g = g.set_ylabels("Survival Probability")


# The value paid really made difference when the subject is surviving onboard Titanic.

# ## 5. Dataset preparation for Model

# I am not going to explore very well the possible exploratory analysis. Let's just ensure there's no missing values in the dataset, all dummies were created and we have sufficiently engineered all possible variables.
# 
# If you want to understand better the variables, consult the following kernels, which inspired me a lot:
# * Yassine Ghouzam, PhD: "Titanic Top 4% with ensemble modeling"
# * LD Freeman, from his kernel: "A Data Science Framework: To Achieve 99% Accuracy". 
# 
# 

# ### 5.1 Macro overview

# In[ ]:


df.describe()


# ### 5.2 Outlier detection

# Outliers do have an effect in training your model, let's take a look:
# 

# In[ ]:


# Outlier detection 

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5* IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(df,2,["Age","SibSp","Parch","Fare"])


# It's Tukey method for detecting outliers. We could use multiples of standard deviation, but this method seemed to me a good first approach.

# In[ ]:


df.loc[Outliers_to_drop] # Show the outliers rows


# In[ ]:


len(df.loc[Outliers_to_drop])/len(df)


# We detect 19 outliers and they represent less than 1.45% of the dataset. I think personally the effect can be negleted because the most impacting variables were 'Fare' with high values and 'SibSp' particulary with the value 8. It's a decision, we can always go back and refine the model aplying some type of outlier filtering
# 
# 

# ### 5.3 Train x test dataframe splitting

# In[ ]:


## Separate train dataset and test dataset
train = df[df.Survived.notnull()]
test = df[df.Survived.isnull()]
train.head(20)


# In[ ]:


#let's check the test dataframe
test.Survived.head(10)


# ### 5.4 Dropping irrelevant variables 

# Understanding wich are the categorical variables to drop. Let's check if we have created dummies for all of them.

# In[ ]:


print(train.info())


# In[ ]:


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)

Y_train = train["Survived"]

X_train = train.drop(labels = ["Survived"],axis = 1)

# Drop useless input variables 
X_train.drop(labels = ["PassengerId", "Sex","Name", "Title", "Embarked"], axis = 1, inplace = True)
print(X_train.info())


# All variables are numeric. Great!

# ## 6. MODELING

# ### 6.1 Simple modeling
# #### 6.1.1 Cross validate models
# 
# I compared the most  popular classifiers and evaluate the mean accuracy of each of them by a stratified kfold cross validation procedure.
# 

# In[ ]:


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# In[ ]:


# Modeling step Test differents algorithms 
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())


cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = Y_train, scoring = "accuracy", 
                                      cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,
                       "CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree",
                                                             "AdaBoost","RandomForest","ExtraTrees",
                                                             "GradientBoosting","MultipleLayerPerceptron",
                                                             "KNeighboors","LogisticRegression",
                                                             "LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")


# In[ ]:


cv_res


# I decided to choose the Linear Discriminant analysis, Random Forest, Logistic regression and Gradient Boosting for tuning by GradientSearchCV.

# #### 6.1.2 Hyperparameter tunning for best models
# 
# I performed a grid search optimization for the tree-based algorithms.

# In[ ]:


# RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,Y_train)

RFC_best = gsRFC.best_estimator_
#Best set of parameters
print(RFC_best)
# Best score
print(gsRFC.best_score_)


# In[ ]:


# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,Y_train)

GBC_best = gsGBC.best_estimator_
#Best set of parameters
print(GBC_best)
# Best score
print(gsGBC.best_score_)


# ### 6.2 Ensemble modeling
# #### 6.2.1 Combining models
# 
# I choosed a voting classifier to combine the predictions coming from the 4 classifiers.
# 
# I preferred to pass the argument "soft" to the voting parameter to take into account the probability of each vote.

# In[ ]:


#ignore warnings
import warnings
warnings.filterwarnings('ignore')
print('-'*25)

LDA =LinearDiscriminantAnalysis()
logitR = LogisticRegression(random_state = random_state)
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('logitR', logitR),
('LDA', LDA),('gbc',GBC_best)], voting='soft', n_jobs=4,weights=[1,1,1,1])

votingC = votingC.fit(X_train, Y_train)

for clf, label in zip([RFC_best, logitR, LDA, GBC_best, votingC],
                      ['rfc','logitR', 'lda','gbc', 'soft voting']):
    scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy',pre_dispatch=4)
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


#     You can see that in some cases voting classifier can be no better than the best model you've just tuned. Let's change the weights...

# In[ ]:


LDA =LinearDiscriminantAnalysis()
logitR = LogisticRegression(random_state = random_state)
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('logitR', logitR),
('LDA', LDA),('gbc',GBC_best)], voting='soft', n_jobs=4,weights=[1,1,1,2])

votingC = votingC.fit(X_train, Y_train)

for clf, label in zip([RFC_best, logitR, LDA, GBC_best, votingC],
                      ['rfc','logitR', 'lda','gbc', 'soft voting']):
    scores = cross_val_score(clf, X_train, Y_train, cv=5, scoring='accuracy',pre_dispatch=4)
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


# Let's try another strategy: Stacking

# In[ ]:


lgr = LogisticRegression()
sclf = StackingClassifier(classifiers=[RFC_best, logitR, LDA, GBC_best, votingC], 
                          meta_classifier=lgr)

print('10-fold cross validation:\n')

for clf, label in zip([RFC_best, logitR, LDA, GBC_best, votingC, sclf], 
                      ['rfc','logitR', 'lda','gbc', 'voting', 'stacking']):

    scores = cross_val_score(clf, X_train, Y_train, 
                                              cv=5, scoring='accuracy')
    print("Accuracy: %0.4f (+/- %0.4f) [%s]" 
          % (scores.mean(), scores.std(), label))


# If we run an ANOVA we couldn't say one model predicts better than other, so let's take the voting Classifier as the final model...

# ### 6.3 Prediction
# #### 6.3.1 Predict and Submit results

# In[ ]:


test.head()


# In[ ]:


## Separate train features and label 
X_test = test.drop(labels = ["Survived"],axis = 1)

# Drop useless input variables 
X_test.drop(labels = ["PassengerId", "Sex","Name", "Title", "Embarked"], axis = 1, inplace = True)
print(X_test.info())


# In[ ]:


test['Survived']=votingC.predict(X_test)

results =  pd.concat([test[['PassengerId', 'Survived']]], axis=1)

results.to_csv("output_python_voting.csv",index=False)


# In[ ]:


results.head()


# Thank you very much for reading it!!
