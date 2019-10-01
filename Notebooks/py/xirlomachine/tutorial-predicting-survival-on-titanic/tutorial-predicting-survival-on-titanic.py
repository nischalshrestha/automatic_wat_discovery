#!/usr/bin/env python
# coding: utf-8

# #Tutorial 
# This is a tutorial for me to get familiar with Machine Learning concepts.
# I will mainly be following this tutorial: https://www.kaggle.com/startupsci/titanic-data-science-solutions
# Please note that I do not claim that the work below is mine, I am following a tutorial and full credit goes to the author Manav Sehgal. I have done some exploratory reading already and have followed a few beginner tutorials, mainly from http://machinelearningmastery.com - I will try to follow steps as indicated there.
# 
# 1. Define problem
# The goal of this tutorial is to predict if a passenger on the Titanic will survive based on a number of variables. For instance, their sex, age, ticket class, fare, whether they had siblings or parents onboard and their cabin number. More details can be found on the data tab. Of course, we also know whether each passenger survived or not. The project description mentions a 38% survival rate.
# 

# In[2]:


# import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Step 2: Prepare data
# Next, read in input data files into their respective pandas dataframes. Note that we have a training data set to train our model and a test dataset which will be used to validate our dataset. This can also be accomplished if one splits the training data up and reserves a portion of it for later.

# In[3]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
combine_df = [train_df, test_df]


# Lets have a look at the data to see what we're working with.

# In[4]:


# Have a look at the data
train_df.head()


# At this stage, I'd like to get to know the data a bit better.  What type is each variable? How complete is the data?
# At that point, we'll be in a better position to perform the proper data munging and know which ML models are most likely to give the best results. Pandas has some handy tools to have a first look at the data very easily. 

# In[5]:


train_df.info()
print('_'*40)
test_df.info()


# In[6]:


# This will perform basic statistics on numerical values only
train_df.describe()


# Some observations.
# We see that the survival rate is the 38% mentioned earlier. 
# Pclass has 3 integer values (also mentioned in description)
# It looks like we have some incomplete values for Age (714 non-null values of a total 891) and Cabin (only 204 non-null values)

# What about the categorical values?

# In[7]:


# We can use nunique to find the number of unique values for each feature
train_df.nunique()
# Or use describe again with the flag to include 0's and hence consider categorical values as well
train_df.describe(include=['O'])


# At this point in the tutorial described above, the "5Cs" are mentioned before proceeding further into the analysis.
# Correlation, Completing, Correcting, Creating, Classifying.
# 
# 1. Correlation
# Some features will be more correlated with Survival than others.  We should determine these, but be careful to revisit in case we are proven wrong later. 
# 
# 2. Completing 
# We can see above than some of the features contain null values. The tutorial points to Age and Embarked as features that should be completed. 
# 
# 3. Correcting
# Can we drop any information safely from our analysis? Name is likely not correlated with survival, and all values are unique anyway. Others are Ticket (Ticket Class, information such as First, Second etc), Cabin (Cabin Number, mostly incomplete data) and PassengerID (just a numeric identifier for each passenger).
# 
# 4. Creating
# This is an interesting step- here we take a look at the features we already have and see if we can derive anything more useful. As mentioned in the tutorial. Family -> combine Parch (parent-child) and Sibsp (number of siblings on board). Title --> obtain from Name (as a proxy for socio-economic status perhaps?)
# Age --> bin values to create an ordinal categorical feature
# Fare range?
# 
# 5. Classifying
# Women, children and upper-class passengers were more likely to have survived.
# (Sex= Female, Age <?? Pclass=1 respectively
# 
# ## Pivot features against Survival
# Perform some exploratory analysis of various features against survival to see how well they are correlated. The features below are selected as they are complete, categorical/discretely valued.

# In[8]:


# Careful of syntax here!!
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)


# In[9]:


train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[10]:


train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[11]:


train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Some patterns are starting to become evident from the pivot tables above - FirstClass passengers are more likely to have survived, as are women. SibSp and Parch need further investigation.
# 
# # Data visualization
# Investigate a bit further using histograms. Plot the numerical features vs. survival
# We will use the Seaborn package (FacetGrid)
# 

# In[12]:


# use a facetgrid (FacetGrid is used to draw plots with multiple Axes 
# where each Axes shows the same relationship conditioned on different levels of some variable.
# It’s possible to condition on up to three variables by assigning variables 
# to the rows and columns of the grid and using different colors for the plot elements.)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# ##Observations
# 
#  - Young Children had a high survival rate (>20%) 
#  - Many 15-25 year olds did not survive. 
#  - Majority of passengers are in the 15-35 age range.
# 
# 

# 

# In[ ]:





# ## Correlating numerical and ordinal features
# Below we will combine multiple features to identify correlations with one plot. We will use numerical and categorical features (eg. Pclass)

# In[13]:


grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6 )
grid.map(plt.hist, 'Age')
grid.add_legend()


# ## Observations
# Pclass=3 had the most passengers but most of them did not survive. 
# The youngest passengers (ie less than 5) in Pclass=2 and Pclass=3 mostly survived. 
# Most passengers in Pclass=1 survived. 
# Pclass varies in terms of Age distribution of passengers.
# 
# Based on these observations, Pclass is good to consider for model training
# 
# #Correlating categorical features
# Let us examine the sex of the passenger, the class and where they embarked with survival.
# We will use line/point plots on the same facetgrid for each embarkation point.

# In[14]:


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# ## Observations
# - PClass 1 and 2 generally had higher survival rates
# - Females in 2 of the 3 embarkment points had significantly higher survival rates than males. A similar pattern is observed for the other embarkment point for males (C)
# - Point of embarkment does have a varying effect on survival for males.
# 
# Based on this, Embarked and Sex will be added to model training.
# Embarked has 889 non null values so the remaining two will need to be completed.
# 
# # Correlating categorical with numerical features
# We can now compare some of the categorical features (Embarked, Sex) with the numeric feature Fare as it seems that these all had an impact on survival.

# In[15]:


grid = sns.FacetGrid(train_df, row='Embarked', col='Survived',size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
grid.add_legend()


# ## Observations
# - Embarkment point affects survival
# - Again, in general females have higher survival rate
# - Higher paying passengers had higher survival rate
# 
# Moving forward, we can bin our Fare data in order to see more meaningful trends in different fare classes.

# In[16]:


train_df['Fare']


# # Data wrangling
# Things get more interesting here. We now have a better idea of what features have a strong impact on survival and those that don't. Also, we have some idea of how to combine our existing features to possibly give a single feature that is more meaningful.
# 
# The tutorial suggests dropping Cabin and Ticket features.
# These should be done on both training and test data sets. 

# In[17]:


print("Before", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine_df = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)


# # Combine existing features to create new ones
# The Name field contains another possibly useful piece of information - the title. The title of an individual could tell us about their age, socio-economic status, job etc. Based on the correlations we have seen above, there could be useful information here that is correlated with survival. Let's extract the Title from Name.

# In[18]:


# iterate over both datasets in combine (train_df and test_df)
# and note that we also added the new field to both datasets

for dataset in combine_df:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])


#print("After", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)


# Perform further data wrangling on Title
# Some of the titles are much more rare (Rev, Sir) than others (Mme, Mrs)- which are also redundant.
# This should be fixed to streamline our analysis.

# In[19]:


for dataset in combine_df:
    # these titles are interesting, but very few numbers of them exist, hence 'Rare'
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# Rare titles had an about average rate of survival
# Miss, Mrs, Master (children) had much higher rates of survival
# Mr had the poorest survival
# 
# To simplify this will be converted to ordinal values.

# In[20]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in combine_df:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
   
train_df.head()


# Now that the useful information has been extracted from Name, it may be removed  from both datasets.
# We will also remove PassengerID from the training dataset (?? why not test as well??)

# In[21]:


train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine_df = [train_df, test_df]

train_df.shape, test_df.shape

#train_df.head()
#test_df.head()



# ## Convert categorical features
# Most models require numerical inputs rather than strings. Thus, the categorical features also should be represented numerically, as we did above with Title. Here the 'Sex' feature has been replaced with numerical representation of 1-> female and 0-> male

# In[22]:


for dataset in combine_df:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
train_df.head()


# # Completing a numerical continuous feature
# As seen above, the Age feature contains null values. To complete it, there are a few possible methods.
# 
# 1. A random number between the mean and 1 standard deviation of Age in the dataset. 
# 2. Guess an age based on correlations with other feature. For example, a correlation between Gender and Pclass. Use the median age value for a given Gneder, Pclass. 
# 3. Combine the two methods above. Use random numbers between the mean and the std for each combination of Pclass and Gender.
# 
# Since method 1 and 3 introduce random noise into the dataset, the cleanest way to proceed is to use method 2.

# In[23]:


grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()


# Preallocate an array to store an age guess for each gender, Pclass combination.

# In[24]:


guess_ages = np.zeros((2,3))
guess_ages


# In[25]:


for dataset in combine_df:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
           
            age_guess = guess_df.median()
            #print(age_guess)
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            #print(guess_ages)
            #print('-'*10)
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),
                        'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)   

train_df.head(20) 


# No more null values in age! They have been replaced with the median value for each specific gender, pclass.

# In[26]:


train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


# For simplicity, replace age with an ordinal value based on the age bands above

# In[27]:


for dataset in combine_df:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[  dataset['Age'] > 64, 'Age'] = 4


# Now we see a simplified plot of the one produced above for age and survival. 

# In[28]:



grid = sns.FacetGrid(train_df, col='Survived')
grid.map(plt.hist, 'Age', bins=20)


# Since age is now represented by a number corresponding to the Age Band, we may remove the age band feature.

# In[29]:


train_df = train_df.drop(['AgeBand'], axis=1)
combine_df = [train_df, test_df]
train_df.head()


# ## Create new features by combining existing features
# We can boil down the Parch and Sibsp features into something that captures both. We'll combine these to create a FamilySize feature.
# 

# In[30]:


for dataset in combine_df:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Of the families on board, it seems like medium size families fared the best.
# 
# Going further along this same idea, we will create another feature called isAlone

# In[31]:


for dataset in combine_df:
    dataset['isAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1 #set isAlone to False if family 

train_df[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean()    


# Simplifying further, we can eliminate the Parch, SibSp and FamilySize features and focus on isAlone for further analysis.

# In[32]:


train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

combine_df = [train_df, test_df]

train_df.head()


# Another derived feature which may be useful is a combination of Age and Class.

# In[33]:


for dataset in combine_df:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


# # Complete categorical feature
# 
# The embarked feature has values of S,Q, and C. Two values are missing, so these can jsut be replaced with the most commonly occurring point of embarkation (use the mode).

# In[34]:


freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


# In[35]:


# use freq_port in the the fillna function
for dataset in combine_df:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ## Convert categorical to numeric
# We will also replace the port with numerical values. we will use a dict and with the ports as key map the values to each.

# In[36]:


for dataset in combine_df:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
train_df.head()


# ## Complete and convert a numeric feature
# The fare value is also incomplete, and we will replace it using the most frequently occurring value (the mode).
# NB. in the tutorial mode is descibed but median is coded?
# 

# In[37]:


test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


# In[38]:


train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)
train_df.head()


# Again, we will use pandas qcut function to separate the fare into bands. Note that qcut ensures that each band has approximately the same number of values where cut separates the range of the data by the number of bins specified.

# In[39]:


train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

#train_df['FareBand'].value_counts()


# As we did with Age, we will now replace this with ordinal values.

# In[40]:


for dataset in combine_df:
    dataset.loc[ dataset['Fare'] <=7.91, 'Fare'] = 0
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <=14.454), 'Fare' ] = 1
    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <=31), 'Fare' ] = 2
    dataset.loc[ (dataset['Fare'] > 31), 'Fare' ] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)


train_df = train_df.drop(['FareBand'], axis=1)
    
    


# In[41]:


train_df.head(10)


# In[42]:


test_df.head(10)


# Just before we begin the model fit, a quick plot of the heat map should reaffirm our feature selection.

# In[43]:


colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# # Model, Predict and Solve
# 
# The dataset is now ready to be used as training data for a model and predict a solution. We need to understand what type of problem we are working with and then implement the best predictive models for this particular case.
# 1. Since we already know the results, this is a **supervised learning** problem
# 2. We are trying to **classify** whether someone survived (1 or 0) based on input variables (**regression**).
# 
# The suggested models for this case are:
# 
# - Logistic Regression 
# -  KNN or k-Nearest Neighbors 
# -  Support Vector Machines 
# -  Naive Bayes classifier 
# -  Decision Tree Random Forrest
# -  Perceptron 
# -  Artificial neural network 
# -  RVM or Relevance Vector Machine
# 
# To begin, we will separate the Survived feature from our training data, as this is what we will try to predict and ensure that the training and test data contain the same features.

# In[44]:


X_train = train_df.drop('Survived', axis=1) # independent variables only
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# ## Logistic Regression
# Logistic Regression is used for cases where the dependent variable or outcome is categorical. In this case, this means that it can be represented by a binary outcome such as pass/fail, win/lose, or in this case, survived/not survived. Logistic regression measures the relationship between the dependent variable and one or more independent variables with estimated probabilities. 

# In[45]:


# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# - Above, line 1 we instantiated a Logistic Regression class object
# - line 2, trained the model on the training data
# - line 3, tested it on the test data set (which only has the independent variables in it)
# - line 4, use logreg.score to evaluate how well the model did
# Lets try and understand how it arrived at these values a bit better. 

# In[46]:


coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# We can see that Sex is positively correlated with Survival (remember Female =1 and Male=0).  Higher title values are also positively correlated. Pclass has the largest negative correlation with survival- ie as Pclass increases, survival decreases.
# 

# ## Support Vector Machines
# 
# SVMs are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other (Wikipedia). SVMs are based on the idea of finding a hyperplane that best divides a dataset into two classes. In the simple case, you can think of it as the green balls are on one side of the lawn, and the blue ones are on the other and you draw a lne to separate them. Now, expand to a higher dimensional space and realize that it is very unlikely that a straight line/plane would separate your data.

# In[47]:


svc  = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# ## k-Nearest Neighbours (k-NN)
# This is a non parametric method used for classification and regression. The model would predict the value for the dependent variable that it is closest to its similar neighbours in the feature space. 

# In[48]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# ## Naive Bayes
# Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. Naive Bayes classifiers are highly scalable and well suited to large datasets.  

# In[49]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# ## Perceptron
# Perceptron is a classification algorithm that makes its predictions based on a linear predictor function by combining a set of weights with the feature vector. (WIkipedia)

# In[50]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# ## Linear SVC
# Linear SVC is typically best for text classification problems. It is another implementation of Support Vector Classification for the case of a linear kernel. (more needed here!!)

# In[51]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# # Stochastic Gradient Descent
# 
# In SGD we iteratively update our weight parameters in the direction of the gradient of the loss function until we have reached a minimum. Unlike traditional gradient descent, we do not use the entire dataset to compute the gradient at each iteration. Instead, at each iteration we randomly select a single data point from our dataset and move in the direction of the gradient with respect to that data point (http://alexminnaar.com/deep-learning-basics-neural-networks-backpropagation-and-stochastic-gradient-descent.html)

# In[52]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# ## Decision Tree
# 
# Decision tree learning uses a decision tree to go from observations about an item (represented in the branches) to conclusions about the item's target value (represented in the leaves). (wikipedia). 

# In[53]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# ## Random Forest
# Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees (n_estimators=100) at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. (WIkipedia)

# In[54]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# ## Model Evaluation
# Both decision tree and Random Forest gave the same score, but Random Forest is acknowledged as slightly better as decision trees have a tendency to overfit the training data set. 

# In[55]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'kNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[66]:


# Now to submit prediction
results = Y_pred
results = pd.Series(results,name="Survived")
results=results.astype(int)
test_df = pd.read_csv('../input/test.csv')
passengerid = test_df['PassengerId'].astype(int)

submission = pd.DataFrame({
        "PassengerId": test_df['PassengerId'],
        "Survived": results
    })


submission.to_csv("random_forest.csv",index=False)
from subprocess import check_output
print(check_output(["ls", "."]).decode("utf8"))


# A huge thank you to Manav Sehgal for creating this tutorial. Still lots to learn for me, but this is a start. 
