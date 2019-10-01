#!/usr/bin/env python
# coding: utf-8

# # Sinking of the Titanic
# 
# This is my second attempt at this competition. I am reading and learning a lot and this time around I wanted to focus more on data preprocessing, engineering and better feature selection. As saying goes, 'Model will be as good as data is'. In my last attempt - <a>https://www.kaggle.com/uguess/titanic-ml-from-disaster</a>, I received the score of 0.71 so my goal this time would be to at least achieve 75% accuracy in prediction. Like always, comments and suggestions are greatly appreciated.
# 
# So lets divide the work into following sections:
# 1. Load and view data
# 2. Understand feature-label relations with visualization
# 3. Data preprocessing and feature engineering
# 4. Build various models and compare
# 5. Visualize the best model prediction
# 6. File Submission

# ## 1. Load and view data

# In[1]:


# Import libraries for linear algebra and loading data
import numpy as np
import pandas as pd

# Load training and test data into dataframes
orig_training_set = pd.read_csv('../input/train.csv')
orig_test_set = pd.read_csv('../input/test.csv')


# In[2]:


# View training data
orig_training_set.head(n=10)


# In[3]:


# View test data
orig_test_set.head(n=10)


# In given training data, **Survived** is label (prediction) and all the rest are features (predictors) but all the provided features do not always contribute to help build model that can make better prediction. For example, PassengerId feature has no relation with survival or death of passengers, Ticket feature is not meaningful and will not help towards better prediction. So, lets drop these features out of our training and test data

# In[4]:


# Drop unnecessary columns from training and test set
training_set = orig_training_set.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)
test_set = orig_test_set.drop(['Name', 'PassengerId', 'Ticket'], axis = 1)

print(training_set.columns)
print(test_set.columns)


# In[5]:


# View statistical information about the training data
training_stats = training_set.describe(include='all')
print(training_stats)


# In[6]:


# View statistical information about the test data
test_stats = test_set.describe(include='all')
print(test_stats)


# In[7]:


# Lets focus on count index of these stats
training_stats.loc['count', :] 


# In[8]:


test_stats.loc['count', :]


# The count data from above tells us that Age, Cabin and Embarked data are missing values from training set and Age, Fare and Cabin are missing values from test set. The Cabin feature in both cases are missing many values and also Cabin feature will not help us predict our labels more accurately so lets drop this feature as well.

# In[9]:


# Drop Cabin feature from training and test set
training_set = training_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

print(training_set.columns)
print(test_set.columns)


# In[10]:


# Update variable that carries statistical information
training_stats = training_set.describe(include='all')
test_stats = test_set.describe(include='all')


# ## 2. Understand feature-label relations with visualization
# 
# Now lets do some data visualization to better understand and examine the correlation between various feature and label. 
# 
# ### Pclass Versus Survived

# In[11]:


# Import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

sns.barplot(x=training_set.Pclass, y=training_set.Survived)
plt.ylabel('Passengers Survived(%)')


# We can see from the bar plot that the Survival percentages are more for Pclass of 1 than Pclass of 2 than Pclass of 3. This gives us a good understanding that higher the socio-economic status of the passengers higher the chances for  their survival.
# 
# ### Sex Versus Survived

# In[12]:


sns.barplot(x=training_set.Survived, y=training_set.Sex)


# We can clearly see that male passengers have fewer chances of survival than females.
# 
# ### Age Versus Survived

# In[13]:


sns.kdeplot(
        training_set.loc[training_set['Survived'] == 0, 'Age'].dropna(), 
        color='red', 
        label='Did not survive')
sns.kdeplot(
        training_set.loc[training_set['Survived'] == 1, 'Age'].dropna(), 
        color='green', 
        label='Survived')
plt.xlabel('Age')
plt.ylabel('Passengers Survived(%)')


# Here we see that passengers are mostly comprised of ages between 20 and 40. Also we see that Age feature has some outliers i.e. Age > ~62 years old.
# 
# ### Embarked Versus Survived

# In[14]:


sns.pointplot(training_set.Embarked, training_set.Survived)


# Looks like most of the passengers who survived the disaster embarked from Cherbourg (C).
# 
# ### SibSp and Parch vs Survived

# In[15]:


sns.regplot(x=training_set.SibSp, y=training_set.Survived, color='r')


# In[16]:


sns.regplot(x=training_set.Parch, y=training_set.Survived, color='b')


# We see that relation between SibSp and Survived and between Parch and Survived does not provide us good understanding of data. Survival rate slightly decreases with increase in SibSp whereas it increases with increase in Parch. We will be combining these two features into HasFamily features in the data preprocessing step.
# 
# ### Fare Versus Survived

# In[17]:


sns.swarmplot(x=training_set.Survived, y=training_set.Fare)


# There is not really useful information we can extract from Fare versus Survived but we can get a rough idea that percentage of passengers who survived paid more higher fares than the percentage of passenger who did not survived the disaster.
# 
# So, now that we have better understanding of the data let's move on to the data preprocessing and feature engineering
# 
# ## 2. Data preprocessing and feature engineering

# In[18]:


# lets view the stats on training and test data again to do some analysis
print(training_stats)
print(test_stats)


# As we know from previous analysis that Age and Embarked features are missing values from training set and Age and Fare features are missing values from test set. We will usually look at the distribution of other data for the same feature to assist us fill these missing values. Data Visualization will help us determine what method to use.

# In[19]:


sns.distplot(training_set['Age'].dropna(), bins=20, rug=True, kde=True)


# From the distribution plot above, we see that the mean of Age values will be most appropriate to use to fill missing values.

# In[20]:


training_set['Age'] = training_set.Age.fillna(training_stats.loc['mean', 'Age'])
test_set['Age'] = test_set.Age.fillna(test_stats.loc['mean', 'Age'])

training_stats = training_set.describe(include='all')
test_stats = test_set.describe(include='all')

print(training_stats.loc['count', 'Age'])
print(test_stats.loc['count', 'Age'])


# In[21]:


sns.countplot(x=training_set.Embarked, palette="Greens_d");


# Since majority of the passengers embarked from Southampton, we will be using mode to fill missing values for Embarked feature in training set.

# In[22]:


from statistics import mode
mode_embarked = mode(training_set['Embarked'])
training_set['Embarked'] = training_set['Embarked'].fillna(mode_embarked)

training_stats = training_set.describe(include='all')

print(training_stats.loc['count', 'Embarked'])


# For a missing value for Fare feature in test set, we will just use passenger with similar feature values to fill it. Note that we really don't need to do all this but this since Fare feature is only missing a value and it may not really contribute a whole lot for our model to understand the data but it would be a good practice and we can use similar logic for other predictions.

# In[23]:


# Understand the relation between empty Fare feature value and other features values
empty_fare = test_set[test_set['Fare'].isnull()]
print(empty_fare)


# So, the passenger with missing Fare value is Pclass of 3, with no SibSp and Parch and Embarked from Southampton. We will pick another passenger with same features values and use that fare feature value to fill this one.

# In[24]:


use_fare = test_set[(test_set['Pclass'] == 3) & 
                    (test_set['SibSp'] == 0) & 
                    (test_set['Parch'] == 0) &
                    (test_set['Embarked'] == 'S')]
test_set['Fare'] = test_set['Fare'].fillna(use_fare['Fare'].iloc[0]);
test_stats = test_set.describe(include='all')

print(test_stats.loc['count', 'Fare'])


# So now we have a complete training and test data with no missing values. So let's move on to feature engineering.
# 
# <u>Feature Engineering</u> is the process of using the knowledge of given data to engineer features such that it helps create better features for machine learning algorithms. The better the features are, the better the model will be.
# 
# Let's begin feature engineering with **Age Feature**. Currently Age is a continous feature, we will be dividing these into different age groups hence turning them into categorical feature, namely, Kid/Teenager (less than 20), Young/Adult (20-40), Mature (40-60) and Elderly (>60).
# 
# The way I am going to add these age groups is by creating four different columns instead of one column. The reason behind this is before we train our model using this data, the values have to be encoded and followed by One Hot Encoding. I will write more about this later. 

# In[25]:


# Engineer the Age data, drop the Age feature and add engineered categorical Age feature.
# If a passenger is less than or equal to 20, then Kid/Teenager, if greater than 20 and
# less than or equal to 40 then Young/Adult and so on...
training_set['Kid/Teenager'] = np.where(training_set['Age'] <= 20, 1, 0)
training_set['Young/Adult'] = np.where((training_set['Age'] > 20) & (training_set['Age'] <= 40), 1, 0)
training_set['Mature'] = np.where((training_set['Age'] > 40) & (training_set['Age'] <= 60), 1, 0)
training_set['Elderly'] = np.where(training_set['Age'] > 60, 1, 0)

test_set['Kid/Teenager'] = np.where(test_set['Age'] <= 20, 1, 0)
test_set['Young/Adult'] = np.where((test_set['Age'] > 20) & (test_set['Age'] <= 40), 1, 0)
test_set['Mature'] = np.where((test_set['Age'] > 40) & (test_set['Age'] <= 60), 1, 0)
test_set['Elderly'] = np.where(test_set['Age'] > 60, 1, 0)

# Now we can drop the Age column
training_set = training_set.drop(['Age'], axis=1)
test_set = test_set.drop(['Age'], axis=1)

# Lets view training data now
training_set.head(n=10)


# In[26]:


# Lets view test data now
test_set.head(n=10)


# Now let's engineer **SibSp and Parch feature**. From the plots above, we couldn't extract useful relations between these feature and Survived label. So, let's combine them into one feature - **HasFamily** and see if we are are able to understand the relation. If the passenger has sibling/spouse or parent/child then the HasFamily feature will be true (1) and if not, false (0).

# In[27]:


training_set['HasFamily'] = np.where(training_set['SibSp'] + training_set['Parch'] > 0, 1, 0)
test_set['HasFamily'] = np.where(test_set['SibSp'] + test_set['Parch'] > 0, 1, 0)

# Now we can drop SibSp and Parch columns
training_set = training_set.drop(['SibSp', 'Parch'], axis=1)
test_set = test_set.drop(['SibSp', 'Parch'], axis=1)

# Lets view training data now
training_set.head(n=10)


# In[28]:


# Lets view test data now
test_set.head(n=10)


# Let's take a quick look at the relation between HasFamily and Survived now. 
# 
# ### HasFamily Versus Survived

# In[29]:


# We can compare percentages between passengers, who has family, survived the disaster versus did not
hasfamily = len(training_set[(training_set['HasFamily'] == 1)])
fam_survived = (len(training_set[(training_set['HasFamily'] == 1) & (training_set['Survived'] == 1)]) / hasfamily) * 100
fam_didnotsurvive = (len(training_set[(training_set['HasFamily'] == 1) & (training_set['Survived'] == 0)]) / hasfamily) * 100

print ("{0:.2f}".format(fam_survived) + "% of passenger with family survived")
print ("{0:.2f}".format(fam_didnotsurvive) + "% of passenger with family did not survive")


# In[30]:


# Now lets compare percentages between passengers, who do not have family, survived the disaster versus did not
nofamily = len(training_set[(training_set['HasFamily'] == 0)])
nofam_survived = (len(training_set[(training_set['HasFamily'] == 0) & (training_set['Survived'] == 1)]) / nofamily) * 100
nofam_didnotsurvive = (len(training_set[(training_set['HasFamily'] == 0) & (training_set['Survived'] == 0)]) / nofamily) * 100

print ("{0:.2f}".format(nofam_survived) + "% of passenger with no family survived")
print ("{0:.2f}".format(nofam_didnotsurvive) + "% of passenger with no family did not survive")


# So, there is no real difference in survival chances if you have a family but surprisingly with you have no family then there are almost 70% chances that a passenger did not survive. We will explore both options with HasFamily and without HasFamily to see if we should keep or drop this feature later when building and comparing models.

# The last one to engineer is the **Fare feature**. We will extract Fare-Group feature out of Fare feature by dividing it into four groups that represents range of fare feature values. 

# In[31]:


# Apply pandas qcut to Fare feature
fare_groups = pd.qcut(training_set['Fare'], 4)
fare_groups.unique()


# In[32]:


# Lets assign values to these fare group feature as we did to age feature
#training_set['Fare-Group1'] = np.where(training_set['Fare'] <= 7.91, 1, 0)
#training_set['Fare-Group2'] = np.where((training_set['Fare'] > 7.91) & (training_set['Fare'] <= 14.454), 1, 0)
#training_set['Fare-Group3'] = np.where((training_set['Fare'] > 14.454) & (training_set['Fare'] <= 31), 1, 0)
#training_set['Fare-Group4'] = np.where(training_set['Fare'] > 31, 1, 0)

#test_set['Fare-Group1'] = np.where(test_set['Fare'] <= 7.91, 1, 0)
#test_set['Fare-Group2'] = np.where((test_set['Fare'] > 7.91) & (test_set['Fare'] <= 14.545), 1, 0)
#test_set['Fare-Group3'] = np.where((test_set['Fare'] > 14.454) & (test_set['Fare'] <= 31), 1, 0)
#test_set['Fare-Group4'] = np.where(test_set['Fare'] > 31, 1, 0)

# View the training data
#training_set.head(n=10)


# In[33]:


# View the test data
test_set.head(n=10)


# In[34]:


# We can now drop the Fare feature out of training and test data
#training_set = training_set.drop(['Fare'], axis = 1)
#test_set = test_set.drop(['Fare'], axis = 1)


# Now that feature engineering is complete, let's encode all the feature, apply one hot encoder (if needed), avoid dummy variable trap and prepare data to build models. The term **Encoding** refers to converting the data into integer form such that machine learning models can intepret them. For example, converting Sex feature into 1 (male) and 0 (female), converting Embarked feature into 1 (Q), 2  S) and 3 (C) by applying encoding. 

# In[35]:


# Encode categorical feature - Sex
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
training_set['Sex'] = encoder.fit_transform(training_set['Sex'])

# Lets view the training data now
training_set.head(n=10)


# In[36]:


# encode test data Sex feature
encoder = LabelEncoder()
test_set['Sex'] = encoder.fit_transform(test_set['Sex'])

# Lets view the training data now
test_set.head(n=10)


# In[37]:


# Now lets encode Embarked feature
encoder = LabelEncoder()
training_set['Embarked'] = encoder.fit_transform(training_set['Embarked'])

# Lets view the training data now
training_set.head(n=10)


# In[38]:


# encode test data Embarked feature
encoder = LabelEncoder()
test_set['Embarked'] = encoder.fit_transform(test_set['Embarked'])

# Lets view the training data now
test_set.head(n=10)


# Therefore, we now see that all of the features have been converted into integer form but there is an issue with the Embarked feature. We have more than two different integer values for Embarked feature. So, in order to eliminate issues of model assuming one Embarked feature value being greater than other i.e. assuming one feature value is more important than others, we will be producing new features converting these into binary form. This is known as **One Hot Encoding**. These new features are also called dummy variables.

# In[39]:


# Apply One-Hot encoding to Embarked feature in training data
training_set = pd.get_dummies(data=training_set, prefix=['Embarked'], columns=['Embarked'])
training_set.head(n=10)


# In[40]:


# Apply One-Hot encoding to Embarked feature in test data
test_set = pd.get_dummies(data=test_set, prefix=['Embarked'], columns=['Embarked'])
test_set.head(n=10)


# In[41]:


# Apply One-Hot Encoding to Pclass feature in training data
training_set = pd.get_dummies(data=training_set, prefix=['Pclass'], columns=['Pclass'])
training_set.head(n=10)


# In[42]:


# Apply One-Hot Encoding to Pclass feature in training data
test_set = pd.get_dummies(data=test_set, prefix=['Pclass'], columns=['Pclass'])
test_set.head(n=10)


# Now One Hot Encoding is complete and also the pandas get dummies class took care of dropping the Embarked and Pclass feature that we one-hot encoded. Now lets eliminate dummy variable trap.  Dummy Variable trap refers to avoiding perfect multicollinearity. This can be done by simply dropping a column out of columns that were one hot encoded. For example, in our data, lets frop the Embarked_2 feature and Pclass_3 feature.

# In[43]:


# Drop columns to avoid dummy variable trap in our training set
#training_set = training_set.drop(['Embarked_2', 'Pclass_3', 'Elderly'], axis=1)
#training_set.head(n=10)


# In[44]:


# Drop columns to avoid dummy variable trap in our test set
#test_set = test_set.drop(['Embarked_2', 'Pclass_3', 'Elderly'], axis=1)
#test_set.head(n=10)


# Finally, now we have to apply feature scaling to the data. **Feature scaling** is applied in order to eliminate the possibility of one feature dominating the other feature because of the values they contain. But before we do that let's seperate our data into features and label and apply scaling afterwards.

# In[45]:


# All the columns in our training data expect Survived represent our features
features_train = training_set.loc[:, training_set.columns != 'Survived'].values
label_train = training_set.loc[:, training_set.columns == 'Survived'].values
features_test = test_set.values

# features is now ndarray type (Sparse matrix of shape) since our model (classifiers) take this type while fitting
print(features_train[0])


# In[46]:


# labels
print(label_train[0])


# In[47]:


# Apply feature scaling to our features_train
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)
print(features_train[0])


# ## 4. Build various models and compare
# 
# The data preprocessing and feature engineering is complete and data has been prepared to start building machine learning models and comparing them. From the given data we can tell that this is a **Classification** problem. Classification is an supervised machine learning algorithm where the prediction values (label) are in discrete form. For example, in this data, we are predicting either a passenger with some features survived or did not suvive the titanic disaster. The other type is a Regression problem where the prediction values are in continous form. For example, predicting a measure of rainfall volume based on the activities of clouds for weather.
# 
# We will be utilizing following classification algorithms to build our model:
# 1. K-Nearest Neighbors
# 2. Support Vector Machines
# 3. Naive Bayes
# 4. Decision Trees
# 5. Random Forest

# In[48]:


# First lets prepare few things to avoid redundant coding and for result visualization

# Create a dataframe to store algorithms and their accuracies
algo_accuracy = pd.DataFrame(columns = ['Algorithm', 'Accuracy'])

# Create a method that gets the mean accuracy score obtained by the classifer (model)
# And store the accuracy in algo_accuracy
def get_store_accuracy(classifier, clf_name, X, y):
    accuracy = classifier.score(X, y) * 100
    algo_accuracy.loc[len(algo_accuracy)] = [clf_name, "{0:.2f}".format(accuracy) + ' %']


# In[49]:


# 4. Decision Trees
algo_accuracy = pd.DataFrame(columns = ['Algorithm', 'Accuracy'])
import warnings
warnings.filterwarnings("ignore")
from sklearn import *
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(n_estimators=6000),
    ensemble.BaggingClassifier(n_estimators=300),
    ensemble.ExtraTreesClassifier(n_estimators=300),
    ensemble.GradientBoostingClassifier(n_estimators=3000),
    ensemble.RandomForestClassifier(n_estimators=300),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(multi_class='one_vs_rest'),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(max_iter=5),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(algorithm='brute', n_neighbors=3, weights='uniform'),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),
    
    #Neural Network
    neural_network.MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(15, 5, 2)),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

class MyTree():
    def __init__(self):
        self.trees = MLA
        self.w = [1.0/len(self.trees) for i in self.trees]
        
    def fit(self, f, l):
        scores = []
        
        for tree in self.trees:
            tree.fit(f, l.ravel())
            get_store_accuracy(tree, tree.__class__.__name__, f, l)
            scores.append(tree.score(f, l.ravel()))
        
        scores = [i**20 for i in scores]
        self.w = scores/sum(scores)
    
    def predict(self, f):
        res = np.asarray(self.trees[0].predict(f), dtype=np.float64) * 0

        for i in range(len(self.trees)):
            weight = self.w[i]
            tree = self.trees[i]
            res += weight * np.asarray(tree.predict(f), dtype=np.float64)

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.bar(range(len(res)), res)
        
        return (np.asarray(res > 0.25, dtype=np.int32))
    
    def score(self, f, l):
        res = self.predict(f)
        return sum(res == l.ravel())/len(res)
    

class AdaTree():
    def __init__(self):
        self.trees = MLA
        self.classifier = ensemble.ExtraTreesClassifier(n_estimators=300)
    
    def convert_data(self, f):
        l = np.asarray(self.trees[0].predict(f))
        for tree in self.trees[1:]:
            res = np.asarray(tree.predict(f))
            l = np.vstack((l, res))
        return l.T
        
    def fit(self, x, y):
        for tree in self.trees:
            tree.fit(x, y)
        l = self.convert_data(x)
        self.classifier.fit(l, y)
    
    def predict(self, x):
        l = self.convert_data(x)
        return self.classifier.predict(l)
    
    def score(self, x, y):
        l = self.convert_data(x)
        return self.classifier.score(l, y)

mytree = MyTree()
mytree.fit(features_train, label_train)
get_store_accuracy(mytree, "My Tree", features_train, label_train)

adatree = AdaTree()
adatree.fit(features_train, label_train)
get_store_accuracy(adatree, "Ada Tree", features_train, label_train)

algo_accuracy


# 

# In[50]:


# Use decision tree classifier to predict the test results
#decision_tree_preds = mytree.predict(features_test)
temp = adatree.predict(features_test)
#print (sum (temp != decision_tree_preds))

# Build a submission file
submission = pd.DataFrame({
        "PassengerId": orig_test_set["PassengerId"],
        "Survived": temp
    })
submission.to_csv('sub_preds.csv', index=False)


# In[ ]:




