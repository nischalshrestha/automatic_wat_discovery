#!/usr/bin/env python
# coding: utf-8

# # Titanic: Machine Learning from Disaster

# ## Table of contents
# 
# - Introduction
# - Step 1: Import libraries & data
# - Step 2: Join data files
# - Step 3: Initial inspection
# - Step 4: Visual data exploration
#     - Survived
#     - Pclass  
#     - Age
#     - SibSp & Parch
#     - Fare
# - Step 5: Feature Engineering
# - Step 6: Machine Learning
#     - Initial models
#     - Feature selection
#     - Model tuning
#     - Voting Classifier
# - Step: Final model predictions and submission

# # Introduction
# 
# Welcome Kagglers! In this kernel, I will be working with the very famous Titanic dataset. It provides information on the fate of each passenger on board the vessel when it tragically sank on its maiden voyage in 1912. Inclusive in the file is further passenger information including age, sex, ticket type and cabin. The challenge set by Kaggle using this dataset is to build a predictive model that can determine whether or not an individual would have survived the 1912 tragedy.
# 
# My aim in this kernel is to explain each step taken right from data loading through to final submission, so that this hopefully serves as a particluar useful resource for beginners, either to this challenge or Data Science more generally. I also hope this kernel can demonstrate how with some relatively simple feature engineering steps and modelling trial and error, you cab achieve a high score within Data Science competitions. At the time of submission the obtained score (0.818) pitched me within the top 5% of the leaderboard. It's now currently flitting between the top 5%/6%. 
# 
# Please ask any questions that you have on my code or approach, or suggestions if you think any part could be done better - I am always looking to improve! And if you found this kernel helpful, i'd very much like to hear it :). OK, the last thing to now say is: Enjoy the read!

# ## 1. Import libraries & data
# 
# I will firstly begin by importing every library that will be used at somepoint throughout this project. 

# In[ ]:


# Every library that will be used in this project is imported at the start.

# Data handling and processing
import pandas as pd
import numpy as np

# Data visualisation & images
import matplotlib.pyplot as plt
import seaborn as sns

# Pipeline and machine learning algorithms
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier

# Model fine-tuning and evaluation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn import model_selection

# Hide system warnings
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# The dataset is split into two separate files, a train set and a test set. We will 'train' and optimise our predictive model on the train set, holding back the test set data for final prediction and submission. We want our model to generalise well to new, never before seen data, hence the test set provides this barometer to us. If our model could predict accurately on the train set, but then fails to generalise to new data (e.g. it predicts poorly), this is a sign that it has 'overfit' to the unique characteristics of the train data. We don't the know 'Survived' labels of the test set, and that's what we're here to find out. If our model can't generalise to new data, it won't be very good in predicting these new labels for us. Therefore, the model won't be very useful!

# In[ ]:


# Data downloaded from Kaggle as a .csv file and read into this notebook from my local directory.
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# ## 2. Join data files

# It's helpful to bring the train and test files together for all steps pre-modelling. That way, we can inspect and edit features within one line of code rather than having duplicte every action over both the train and test set individually. The below code completes this combination.

# In[ ]:


# Join all data into one file
ntrain = train.shape[0]
ntest = test.shape[0]

# Creating y_train variable; we'll need this when modelling, but not before
y_train = train['Survived'].values

# Saving the passenger ID's ready for our submission file at the very end
passId = test['PassengerId']

# Create a new all-encompassing dataset
data = pd.concat((train, test))

# Printing overall data shape
print("data size is: {}".format(data.shape))


# ## 3. Initial inspection

# In[ ]:


# Let's see some basic info about the dataset
data.info()


# .info() is a really helpful way to understand more about the data structure, and how much there is of it. Looking at the Titanic dataset, we can still straight away a mixture of numbers (float & int) & words (objects), and some missing values that we'll need to take care of later. Machine Learning does not take kindly to missing values, that is, it will not work. So we'll need to fill them in somehow.
# 
# For reference, here's a description of what each feature contains:
# 
# <b> - PassengerId:</b> The unique identifier in this Data file <br>
# <b> - Survived:</b>    The fate of each passenger (target) <br>
# <b> - Pclass:</b>      The ticket class <br>
# <b> - Name:</b>       The passenger name <br>
# <b> - Sex:</b>         The passenger sex <br>
# <b> - Age:</b>         The passenger age in years   
# <b> - Sibsp:</b>       The number of siblings/spouses also travelling <br>
# <b> - Parch:</b>       The number of parents/children also travelling <br>
# <b> - Ticket:</b>      The passenger ticket number <br>
# <b> - Fare:</b>        The passenger fare <br>
# <b> - Cabin:</b>       The passenger cabin number <br>
# <b> - Embarked:</b>    The passenger's port of Embarkation <br>

# In[ ]:


# Inspecting the first five rows, or 'observations'
data.head()


# Evidence of missing values is straight away evident in the 'cabin' column - NaN (not a number) indicates this. Ticket type also looks like a bit of a mish/mash - initial thoughts are that this might prove a difficult feature to engineer anything meaningful out of. 

# In[ ]:


# Returning descriptive statistics of the train dataset
data.describe()


# Every column that is either an int or a float can be 'described'. Taking the mean value of the 'Survived' column, we can see that <b>38%</b> of passengers (within the train set) survived when the Titanic sank.
# 
# To understand a little more around how much data is actually missing, the final step before some more visual EDA is conducted will be to inspect per feature the quantity of missing values:

# In[ ]:


# Provide NaN count for each feature in the dataset
print(data.isnull().sum())


# So Age and Cabin are the main culprits - we'll come back to these a little later on.

# ## 4. Visual Data Exploration

# Before we can know how much (or little) feature engineering is needed, we need to have a good sense of what we're working with. The simple explorations discussed above are useful in terms of getting a holistic view of the overall dataset. To understand more about specific features, it is considered best practice to visualise it first. Step 4 walks through some simple visualisations, beginning with a correlation matrix

# In[ ]:


# Correlation matrix of all numeric features
sns.heatmap(train.corr(), annot = True)


# Perhaps not the most insightful view at this stage given that some features are pending engineering, however a visible correlation does exist between Survived and Pclass and Fare. Age, SibSp & Parch would also seem like logical predictors and it would be expected that after these variables have been preprocessed their correlation to Survived will increase.
# 
# Let's get an initial sense of these features then, beginning with the target: Survived.

# #### Survived

# In[ ]:


fig = plt.figure(figsize = (10,5))
sns.countplot(x='Survived', data = train)
print(train['Survived'].value_counts())


# A quick inspection into the survived feature reveals that as before seen, 38% of passengers within the training set survived when the Titanic sank. This equates to 342 passengers out of 891 in total.

# #### Pclass

# In[ ]:


# Bar chart of each Pclass type
fig = plt.figure(figsize = (10,10))
ax1 = plt.subplot(2,1,1)
ax1 = sns.countplot(x = 'Pclass', hue = 'Survived', data = train)
ax1.set_title('Ticket Class Survival Rate')
ax1.set_xticklabels(['1 Upper','2 Middle','3 Lower'])
ax1.set_ylim(0,400)
ax1.set_xlabel('Ticket Class')
ax1.set_ylabel('Count')
ax1.legend(['No','Yes'])

# Pointplot Pclass type
ax2 = plt.subplot(2,1,2)
sns.pointplot(x='Pclass', y='Survived', data=train)
ax2.set_xlabel('Ticket Class')
ax2.set_ylabel('Percent Survived')
ax2.set_title('Percentage Survived by Ticket Class')


# Confirmation of what was seen in the correlation matrix that a trend is visible between ticket class and survival chance. The higher class ticket, the more likely one is to have survived. This will become a very handy predictor in the machine learning algorithm.

# #### Age

# In[ ]:


# Bar chart of age mapped against sex. For now, missing values have been dropped and will be dealt with later
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Sex']=='female']
men = train[train['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=20, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=20, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# More can be understood about age when plotted alongside sex. These graphs reveal that overall women were much more likely to survive than men, and this is largely regardless of age. For both sexes, it appears that chances of survival are more likely at a younger age, which is what might have been expected. From the age of 20, it was consistently more likely that men would not have survived, up until their age approached 80. For women, apart from a potentially anomalous finding around the 8-9 bracket, they were always more likely to survive.

# #### SibSp & ParCh

# In[ ]:


# Plotting survival rate vs Siblings or Spouse on board
fig = plt.figure(figsize = (10,12))
ax1 = plt.subplot(2,1,1)
ax1 = sns.countplot(x = 'SibSp', hue = 'Survived', data = train)
ax1.set_title('Survival Rate with Total of Siblings and Spouse on Board')
ax1.set_ylim(0,500)
ax1.set_xlabel('# of Sibling and Spouse')
ax1.set_ylabel('Count')
ax1.legend(['No','Yes'],loc = 1)

# Plotting survival rate vs Parents or Children on board
ax2 = plt.subplot(2,1,2)
ax2 = sns.countplot(x = 'Parch', hue = 'Survived', data = train)
ax2.set_title('Survival Rate with Total Parents and Children on Board')
ax2.set_ylim(0,500)
ax2.set_xlabel('# of Parents and Children')
ax2.set_ylabel('Count')
ax2.legend(['No','Yes'],loc = 1)


# Not surprisingly, the structure of these two graphs appear similar, with a similar density of passengers featured within each count, with also a similar ratio of survived vs not survived. This adds further rationale for these two features to be combined, which will be performed at the Data Preprocessing stage.

# #### Fare

# In[ ]:


# Graph to display fare paid per the three ticket types
fig = plt.figure(figsize = (10,5))
sns.swarmplot(x="Pclass", y="Fare", data=train, hue='Survived')


# Fare has been displayed per ticket type, revealing that those within Pclass 3 paid a similar fare to those in Pclass 2, but their chance of survival appears to be a lot lower. Pclass contains the highest fares, along with the highest rate of survivial, denoted by the higher ratio of orange points.

# ## 5. Feature engineering

# With a clearer understanding about the current data shape, we can now start engineering features so they're ready for modelling. Some steps will be things we need to do, e.g. filling in blanks. Other steps are more choice in terms of making features more useful, thus allowing for stronger model performance. I'll begin with one such feature.

# ### Family survival

# I must reference this [kernel](http://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever), which is credited with introducing this feature:
# 
# The aim of this feature is to group together people (usually families) with similar ticket information, with the logic that groups that are together have similar survival chances. I encourage you to follow the link to learn more about this feature, as the code is quite complex. In my opinion it seemed an interesting and logical feature to create, so i'm adding it in my dataset.

# #### Family information

# In[ ]:


# Extract last name
data['Last_Name'] = data['Name'].apply(lambda x: str.split(x, ",")[0])

# Fill in missing Fare value by overall Fare median
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

# Setting coin flip (e.g. random chance of surviving)
default_survival_chance = 0.5
data['Family_Survival'] = default_survival_chance

# Grouping data by last name and fare - looking for families
for grp, grp_df in data[['Survived','Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',
                           'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):
    
    # If not equal to 1, a family is found 
    # Then work out survival chance depending on whether or not that family member survived
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            smax = grp_df.drop(ind)['Survived'].max()
            smin = grp_df.drop(ind)['Survived'].min()
            passID = row['PassengerId']
            if (smax == 1.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
            elif (smin == 0.0):
                data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0

# Print the headline
print("Number of passengers with family survival information:", 
      data.loc[data['Family_Survival']!=0.5].shape[0])


# #### Group information

# In[ ]:


# If not equal to 1, a group member is found
# Then work out survival chance depending on whether or not that group member survived
for _, grp_df in data.groupby('Ticket'):
    if (len(grp_df) != 1):
        for ind, row in grp_df.iterrows():
            if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):
                smax = grp_df.drop(ind)['Survived'].max()
                smin = grp_df.drop(ind)['Survived'].min()
                passID = row['PassengerId']
                if (smax == 1.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 1
                elif (smin==0.0):
                    data.loc[data['PassengerId'] == passID, 'Family_Survival'] = 0

# Print the headline
print("Number of passenger with family/group survival information: " 
      +str(data[data['Family_Survival']!=0.5].shape[0]))


# For this feature to work, the train/test index had to be kept as is. For the remaining feature engineering steps this isn't essential so I will reset the index so we now have one continual index (1-1309), rather than 1-end of train set followed by 1-end of test set.

# In[ ]:


# Reset index for remaining feature engineering steps
data = data.reset_index(drop=True)
data = data.drop('Survived', axis=1)
data.tail()


# Great, the index runs from 1-1309. Let's progress with engineering features.

# ### Fare

# The single missing fare value was taken care of previously, so let's take a look at fare overall and see whether we could or should take any further action.

# In[ ]:


# Visualising fare data
plt.hist(data['Fare'], bins=40)
plt.xlabel('Fare')
plt.ylabel('Count')
plt.title('Distribution of fares')
plt.show()


# Well, that's a fare from ideal view! There is quite a severe left-side skew which probably won't pair up all that well with Machine Learning algorithms. I think there's two possible approaches here.
# 
# 1. Turn Fare into categorical data, by breaking it down into bins
# 2. Transform the data so it fits in with a normal distribution (e.g. log transformation could work here)
# 
# I played around with both options, in the end deciding on using bins was the most effective means of proceeding. I also achieved a better final score using the bin approach. To create my bins, i'm going to use the clever Pandas qcut tool, which creates equal size bins based on the quantity chosen (in this case, 4).

# In[ ]:


# Turning fare into 6 bins due to heavy skew in data
data['Fare'] = pd.qcut(data['Fare'], 4)

# I will now use Label Encoder to convert the bin ranges into numbers
lbl = LabelEncoder()
data['Fare'] = lbl.fit_transform(data['Fare'])


# In[ ]:


# Visualise new look fare variable
sns.countplot(data['Fare'])
plt.xlabel('Fare Bin')
plt.ylabel('Count')
plt.title('Fare Bins')


# ### Name

# In[ ]:


# Inspecting the first five rows of Name
train['Name'].head()


# The full names as they are will not be helpful to us, although, there's probably something useful within title e.g. categorising males and females, boys and girls. Therefore, i'm going to extract this data and create a new feature for Title, before binning Name.

# In[ ]:


# New function to return name title only
def get_title(name):
    if '.' in name:
        return name.split(',')[1].split('.')[0].strip()
    else:
        return 'Unknown'


# In[ ]:


# Creating two lists of titles, one for each dataset
titles_data = sorted(set([x for x in data['Name'].map(lambda x: get_title(x))]))


# To understand better what we are now working with, the list size and values will be printed below.

# In[ ]:


# Printing list length and items in each list
print(len(titles_data), ':', titles_data)


# 18 unique title values is a lot, and I anticipate that for many only a few observations exist, which isn't helpful. I'm going to keep this simple and band titles in one of four categories: Mr, Mrs, Master & Miss. To help me complete this I will define my own handy function - see below:

# In[ ]:


# New function to classify each title into 1 of 4 overarching titles
def set_title(x):
    title = x['Title']
    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        return 'Mr'
    elif title in ['the Countess', 'Mme', 'Lady','Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title


# In[ ]:


# Applying the get_title function to create the new 'Title' feature
data['Title'] = data['Name'].map(lambda x: get_title(x))
data['Title'] = data.apply(set_title, axis=1)


# In[ ]:


# Printing values of the title column (checking function worked!)
print(data['Title'].value_counts())


# Group sizing looks good enough, i'm happy to continue with this!

# ### Age

# Age has some missing values, as was seen earlier when initially inspecting the data. Let's recap on the current state-of-play:

# In[ ]:


# Returning NaN within Age across Train & Test set
print('Total missing age data: ', pd.isnull(data['Age']).sum())


# There's a reason that I chose to look at Age after Name, that's because I'm going to use the new Title feature to calculate the missing the Age values - a technique called 'imputation'. Imputing can either be completed by the Mean or Median. Let's use the below information to help decide on which might be most accurate.

# In[ ]:


# Check which statistic to use in imputation
print(data['Age'].describe(exclude='NaN'))


# The mean and percentile breakdown indicates multiple features converging around the 30 mark, which perhaps isn't surprising. Based on this it may be better to proceed with imputing with the median (middle) value. What i'm now going to do is group the dataset by the four different titles, and then impute the missing age values with the average age of each title, be that Mr, Mrs, Master or Miss. The below code completes this:

# In[ ]:


# Imputing Age within the train & test set with the Median, grouped by Pclass and title
data['Age'] = data.groupby('Title')['Age'].apply(lambda x: x.fillna(x.median()))


# In[ ]:


# Visualise new look age variable
plt.hist(data['Age'], bins=40)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of ages')
plt.show()


# So now we have a completed view of Age, it makes sense to visualise it. The distribution isn't bad, but I'm not totally satisfied - a slight left skew can still be seen. I'm going to follow the approach used for the Fare feature by turning Age into bins.

# In[ ]:


# Turning data into 5 bins due to heavy skew in data
data['Age'] = pd.qcut(data['Age'], 4)

# Transforming bins to numbers
lbl = LabelEncoder()
data['Age'] = lbl.fit_transform(data['Age'])


# In[ ]:


# Visualise new look fare variable
plt.xticks(rotation='90')
sns.countplot(data['Age'])
plt.xlabel('Age Bin')
plt.ylabel('Count')
plt.title('Age Bins')


# I will stick with these bins for now. Now that I'm finished using the Title feature, I will round up by transferring the titles over to numbers ready for Machine Learning.

# In[ ]:


data['Title'] = data['Title'].replace(['Mr', 'Miss', 'Mrs', 'Master'], [0, 1, 2, 3])


# ### Sex

# A simple step for Sex, just recoding to numbers for Machine Learning.

# In[ ]:


# Recoding sex to numeric values with use of a dictionary for machine learning model compatibility
data['Sex'] = data['Sex'].replace(['male', 'female'], [0, 1])


# ### Embarked

# In[ ]:


data['Embarked'].describe()


# There are two missing values for Embarked - let's replace it with the most frequently occurring value. I'll then convert the letters to numeric values.

# In[ ]:


# Filling in missing embarked values with the mode (S)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Converting to numeric values
data['Embarked'] = data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2])


# ### Cabin

# In[ ]:


# Inspecting head of Cabin column
train.head()


# Upon closer inspection into cabin, we can see that it follows a Letter/Number format. A bit of extra internet research reveals that the letter actually refers to the floor in the titanic where each passenger resided. This  information may be helpful in the prediction, e.g. did those in lower cabins have a smaller/larger chance of survival? Therefore we will begin by extracting the letter only from the Cabin column, and then labelling all NaN's with an 'Unknown' cabin reference.

# In[ ]:


# Labelling all NaN values as 'Unknown'
data['Cabin'].fillna('Unknown',inplace=True)


# In[ ]:


# Extracting the first value in the each row of Cabin
data['Cabin'] = data['Cabin'].map(lambda x: x[0])


# In[ ]:


# Return the counts of each unique value in the Cabin column
data['Cabin'].value_counts()


# As previously seen, there is an overwelhming majority of unknown Cabins in the train dataset. Based on this, the best option here might be to create two groups: known and unknown. This will avoid over-fitting on the sparse data by cabin level, and is what will be completed next with the help of a new function.

# In[ ]:


# New function to classify known cabins as 'Known', otherwise 'Unknown'
def unknown_cabin(cabin):
    if cabin != 'U':
        return 1
    else:
        return 0
    
# Applying new function to Cabin feature
data['Cabin'] = data['Cabin'].apply(lambda x:unknown_cabin(x))


# ### SibSp & Parch

# It would make sense that these two features were combined into one, so that's what we'll do now. Nice and simple!

# In[ ]:


# Creating two features of relatives and not alone
data['Family Size'] = data['SibSp'] + data['Parch']


# ### Final look

# In[ ]:


# Final look at the data
data.head()


# Everything that I want to carry forward to the machine learning stage looks ready. There are some features remaining that, as previously discussed, I don't wish to use. The very last step will be to therefore remove these features.

# In[ ]:


# Dropping what we know need for Machine Learning
data = data.drop(['Name', 'Parch', 'SibSp', 'Ticket', 'Last_Name', 'PassengerId'], axis = 1)


# And that's the lot - onto machine learning!

# ## 6. Machine Learning

# Before we can fit models, a few more steps are needed in order to get the data in the correct shape for modelling. This involves re-splitting the train & test datasets, followed by setting up our X_train & X_test variables. Note that we already have our y_train variable from before. We don't have a y_test variable, this would be the survival stat per users in the test set, and this is what we are looking to predict!
# 
# I am also going to scale the data using the StandardScaler tool. This isn't a requirement for every algorithm, but for those that use what's known as the euclidean distance (or straight line distance) in order to make predictions, having features that all operate on different length scales will unfairly skew the model's interpretation of the data. So, StandardScaler aligns all features onto the same scale and thus avoids us running into this issue.

# In[ ]:


# Return to train/test sets
train = data[:ntrain]
test = data[ntrain:]


# In[ ]:


# Set up feature and target variables in train set, and remove Passenger ID from test set
X_test = test
X_train = train

# Scaling data to support modelling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Now onto the fun stuff, training our cleaned up data. I am going to tackle the modelling in a series of steps which are hopefully easy to follow:
# 
# 1. <b>Train initial models</b>
# 2. <b>Remove features and re-train</b>
# 3. <b>Tuned parameters and re-train</b>
# 4. <b>Build into a Voting Classifier and predict.</b>
# 
# I'll explain more about each step as I go along!

# ### Round 1: Initial models

# The current problem is a classification problem, that is, the outcome can be classified into one class or the other (survived or not). Each of the below algorithms is applicable for such a problem and I'm interested to see how each of them perform on the current dataset, and which comes out on top. I would encourage to check the sci-kit learn documentation for a more detailed overview in regards to how each algorithm specifically works.
# 
# What i'm going to do here is fit each of these 11 models in turn, before returning what's know as their cross-validated score. A model's performance is dependent on the way the data has been split between training and test data. This isn't really that representative of the model's ability to generalise, because there could be unique quirks within the train (or test) set which the model either learns (or does not learn) incorrectly. In order to gain a smooth out representation of the full dataset, K fold cross validation can be applied. It works in the following way: 
# 
# Randomly split your entire dataset into k 'folds'. 
# - For each k folds in your dataset, build your model on k – 1 folds of the data set. Then, test the model to check the effectiveness for kth fold. 
# - Record the error you see on each of the predictions. 
# - Repeat this until each of the k folds has served as the test set. 
# 
# The average of your k recorded errors is called the cross-validation error and will serve as your performance metric for the model. I am going to use 10 folds in this example, which will be plenty.

# In[ ]:


# Initiate 11 classifier models
ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
xgb = XGBClassifier()
gbc = GradientBoostingClassifier()
svc = SVC(probability=True)
ext = ExtraTreesClassifier()
ada = AdaBoostClassifier()
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier()

# Prepare lists
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
scores = []

# Sequentially fit and cross validate all models
for mod in models:
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)
    scores.append(acc.mean())


# The above for loop has packed my 11 model cross validated scores into the list 'score'. I'm now going to unpack this data into a table first, then graph to learn the results.
# 
# Note at this time that our scoring metric is 'accuracy'. This is the competition requirement and is simply a percentage count of the correct classifications provided. For classification problems, however, it's not always best practice to use accuracy as your scoring metric. Consider a dataset where 80% of the target is class A, and the remaining 20% is class B. An accuracy score of 80% on this dataset might on face value sound positive, however, the classifier could have simply predicted every observation as class A. There are 80% in class A in total, thus it is 80% correct. This model clearly lacks any real use, but a mis-leading accuracy score could prevent that being so easily spotted. Metrics such as precision & recall or the roc-auc score can be used as alternative scoring metrics for classification problems. I'll spare the description here, but have a Google if you're interested in learning more.

# In[ ]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],
    'Score': scores})

result_df = results.sort_values(by='Score', ascending=False).reset_index(drop=True)
result_df.head(11)


# In[ ]:


# Plot results
sns.barplot(x='Score', y = 'Model', data = result_df, color = 'c')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.xlim(0.80, 0.86)


# Round 1 complete and it is the Gradient Boosting algorithms that come out top. This isn't a surprise - Gradient boosting is a machine learning technique for regression and classification problems, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. It builds the model in a stage-wise fashion like other boosting methods do, and it generalizes them by allowing optimization of an arbitrary differentiable loss function. They are typically regarded as best in class predictors and they form the basis on many winning competition models.
# 
# I want to now see how heavily each feature was leaned in the modelling process. Let's look at what feature XGBoost found most useful when achieved the top score in round one. To help present this data i'm going to construct my own gragh - code below.

# In[ ]:


# Function for new graph
def importance_plotting(data, x, y, palette, title):
    sns.set(style="whitegrid")
    ft = sns.PairGrid(data, y_vars=y, x_vars=x, size=5, aspect=1.5)
    ft.map(sns.stripplot, orient='h', palette=palette, edgecolor="black", size=15)
    
    for ax, title in zip(ft.axes.flat, titles):
    # Set a different title for each axes
        ax.set(title=title)
    # Make the grid horizontal instead of vertical
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    plt.show()


# In[ ]:


# Building feature importance into a DataFrame
fi = {'Features':train.columns.tolist(), 'Importance':xgb.feature_importances_}
importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)


# In[ ]:


# Creating graph title
titles = ['The most important features in predicting survival on the Titanic: XGBoost']

# Plotting graph
importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)


# Always the most interesting - in my opinion anyway! The imputed age feature went down a treat with XGBoost, followed closely by the new Family Survival feature. I'm pretty happy with that. Sex, Cabin & Embarked were a little more, well, useless shall we say. Scores so low indicates that these features more likely hindered rather than helped in the model prediction. Let's check out a different top performing algorithm.

# In[ ]:


# Building feature importance into a DataFrame
fi = {'Features':train.columns.tolist(), 'Importance':np.transpose(log.coef_[0])}
importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)


# In[ ]:


# Creating graph title
titles = ['The most important features in predicting survival on the Titanic: Logistic Regression']

# Plotting graph
importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)


# Rather than feature importances, Logistic Regression uses coefficients which actually aid better real world interpretation. There are both positive & negative impact features on display, this time with Title being most useful. Sex is also a strong feature for this model, in stark contrast to XGB. Those in the middle (Embarked, Fare, Cabin) all had largely no part to play except for possibly creating unhelpful 'noise' in the dataset. Noise weakens the model's ability to find patterns clearly, and can in some cases lead to lower overall performance. Embarked & cabin are two repeat offenders, and as such go onto my watch list!

# ### Round 2: Feature selection

# We have made good progress so far and have a few particularly strong performing models, however, if we did we'd be failing to harness the true potential of each model's ability to predict on this dataset. The following steps will walk through how we can optimise our 'baseline' predictions to yield a stronger overall score (and ultimately climb further up the leaderboard).
# 
# The first step is feature selection, and by that I mean carry forward only helpful features and remove those that are not contributing towards model performance. This step I anticipate should help to unshackle some currently weak performing models (i'm looking at the KNN model when I say this).
# 
# To get a better gist in terms of what needs the boost, i'm going to get a collective view of all models that will give us a 'feature importance' list. We can then compare these values and draw a cut-off for the features that, overall, provides the least amount of help.

# In[ ]:


# Getting feature importances for the 5 models where we can
gbc_imp = pd.DataFrame({'Feature':train.columns, 'gbc importance':gbc.feature_importances_})
xgb_imp = pd.DataFrame({'Feature':train.columns, 'xgb importance':xgb.feature_importances_})
ran_imp = pd.DataFrame({'Feature':train.columns, 'ran importance':ran.feature_importances_})
ext_imp = pd.DataFrame({'Feature':train.columns, 'ext importance':ext.feature_importances_})
ada_imp = pd.DataFrame({'Feature':train.columns, 'ada importance':ada.feature_importances_})

# Merging results into a single dataframe
importances = gbc_imp.merge(xgb_imp, on='Feature').merge(ran_imp, on='Feature').merge(ext_imp, on='Feature').merge(ada_imp, on='Feature')

# Calculating average importance per feature
importances['Average'] = importances.mean(axis=1)

# Ranking top to bottom
importances = importances.sort_values(by='Average', ascending=False).reset_index(drop=True)

# Display
importances


# The league table is out and we have a winner in Title. Maybe this isn't so surprising given within title information on both age and gender is available. I find it interesting to see that where some models lean very heavily on a certain feature, others gain barely any value from it. The variance in Sex and Age is particularly striking. Let's get this data int a final graph before making a decision on which to cull.

# In[ ]:


# Building feature importance into a DataFrame
fi = {'Features':importances['Feature'], 'Importance':importances['Average']}
importance = pd.DataFrame(fi, index=None).sort_values('Importance', ascending=False)


# In[ ]:


# Creating graph title
titles = ['The most important features in predicting survival on the Titanic: 5 model average']

# Plotting graph
importance_plotting(importance, 'Importance', 'Features', 'Reds_r', titles)


# It looks clear now that Embarked & Cabin really aren't helping us out, and therefore I am going to get rid of them. If I was looking to prioritise 1 or 2 models rather than 11, I would also consider removal of Sex, Fare and potentially Pclass too (depending on which were the models I was considering). However we can see from the above table that for some algorithms, each feature has an important part to play, and I want to prioritise strong performance across the board in preparation for the Voting Classifier (more on that later). So for now it's just the two for the chop.

# In[ ]:


# Drop redundant features
train = train.drop(['Embarked', 'Cabin'], axis=1)
test = test.drop(['Embarked', 'Cabin'], axis=1)

# Re-build model variables
X_train = train
X_test = test

# Transform
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# We're now into Round 2, i will complete the same steps as above and then stack our new results against the previous results, hopefully showing the increments gained from feature selection.

# ### Model re-training

# In[ ]:


# Initiate models
ran = RandomForestClassifier(random_state=1)
knn = KNeighborsClassifier()
log = LogisticRegression()
xgb = XGBClassifier(random_state=1)
gbc = GradientBoostingClassifier(random_state=1)
svc = SVC(probability=True)
ext = ExtraTreesClassifier(random_state=1)
ada = AdaBoostClassifier(random_state=1)
gnb = GaussianNB()
gpc = GaussianProcessClassifier()
bag = BaggingClassifier(random_state=1)

# Lists
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
scores_v2 = []

# Fit & cross validate
for mod in models:
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)
    scores_v2.append(acc.mean())


# In[ ]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],
    'Original Score': scores,
    'Score with feature selection': scores_v2})

result_df = results.sort_values(by='Score with feature selection', ascending=False).reset_index(drop=True)
result_df.head(11)


# In[ ]:


# Plot results
sns.barplot(x='Score with feature selection', y = 'Model', data = result_df, color = 'c')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.xlim(0.80, 0.86)


# Great stuff, some big shifters on display. Gaussian Process now heads the pack over Gradient Boosting and the KNN model is now also mixing it up at the top, whereas prior to feature selection it was languishing down at the bottom. In all, 9/11 models have improved in score, with XGBoost remaining static and Adaboost taking a slight knock. That's a very favourable from something as simple as removing two features. We can now progress to round 3 - model tuning.

# ### Round 3: Model (hyper-parameter) tuning

# Within most machine learning algorithm exist a number of parameters that together can be fine tuned to produce the most accurate prediction on a given dataset. This is essentially what Hyperparamter tuning is - finding the best parameters for your model. The best way to achieve this, while computationally expensive, is to use a GridSearchCV. 
# 
# GridSearchCV is an exhaustive search over specified parameter values for an estimator, in order to find the values for optimum model performance. Each model has its own parameter's, so the Grid Search needs to be specific for each model. Below, I will complete a GridSearchCV for each model, specifying search ranges for each model's most important parameters, in order to find the one's that yield the highest accuracy score for the Titanic train set. 
# 
# FYI, if you're forking this kernel, it will take a while to run. I've kept my GridSearch relatively light touch in all, but expect to wait around 30 minutes from start to finish.
# 
# More information on each model's parameter's can be found online. If you're also wondering how I arrived at suitable ranges for each parameter, I used a combination of Google, past experience and trial and error. That's it :).

# #### SVC

# In[ ]:


# Parameter's to search
Cs = [0.001, 0.01, 0.1, 1, 5, 10, 15, 20, 50, 100]
gammas = [0.001, 0.01, 0.1, 1]

# Setting up parameter grid
hyperparams = {'C': Cs, 'gamma' : gammas}

# Run GridSearch CV
gd=GridSearchCV(estimator = SVC(probability=True), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Gradient Boosting Classifier

# Note: There are many more parameter's that could, and possibly should be tested here, but in the interest i've limited the tuning to establishing the appropriate learning_rate vs n_estimators trade off. The higher one value, the lower the other.

# In[ ]:


# Parameter's to search
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
n_estimators = [100, 250, 500, 750, 1000, 1250, 1500]

# Setting up parameter grid
hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}

# Run GridSearch CV
gd=GridSearchCV(estimator = GradientBoostingClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Logistic Regression

# In[ ]:


# Parameter's to search
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 10)

# Setting up parameter grid
hyperparams = {'penalty': penalty, 'C': C}

# Run GridSearch CV
gd=GridSearchCV(estimator = LogisticRegression(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### XGBoost
# ##### Step 1

# Gradient boosting algorithms are best tuned sequentially - it's less expensive and can yield better results. I'll demonstrate the approach for XGBoost. I could have done the same for the Gradient Boosting classifier, but it's a slower algorithm to run and I am obviously too impatient! Each time I find the appropriate parameter value, i'll specify this within my model in the subsequent testing step.

# In[ ]:


# Parameter's to search
learning_rate = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
n_estimators = [10, 25, 50, 75, 100, 250, 500, 750, 1000]

# Setting up parameter grid
hyperparams = {'learning_rate': learning_rate, 'n_estimators': n_estimators}

# Run GridSearch CV
gd=GridSearchCV(estimator = XGBClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# ##### Step 2

# In[ ]:


max_depth = [3, 4, 5, 6, 7, 8, 9, 10]
min_child_weight = [1, 2, 3, 4, 5, 6]

hyperparams = {'max_depth': max_depth, 'min_child_weight': min_child_weight}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# ##### Step 3

# In[ ]:


gamma = [i*0.1 for i in range(0,5)]

hyperparams = {'gamma': gamma}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 
                                          min_child_weight=1), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# ##### Step 4

# In[ ]:


subsample = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
colsample_bytree = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    
hyperparams = {'subsample': subsample, 'colsample_bytree': colsample_bytree}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 
                                          min_child_weight=1, gamma=0), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# ##### Step 5

# In[ ]:


reg_alpha = [1e-5, 1e-2, 0.1, 1, 100]
    
hyperparams = {'reg_alpha': reg_alpha}

gd=GridSearchCV(estimator = XGBClassifier(learning_rate=0.0001, n_estimators=10, max_depth=3, 
                                          min_child_weight=1, gamma=0, subsample=0.6, colsample_bytree=0.9),
                                         param_grid = hyperparams, verbose=True, cv=5, scoring = "accuracy")

gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Gaussian Process

# In[ ]:


# Parameter's to search
n_restarts_optimizer = [0, 1, 2, 3]
max_iter_predict = [1, 2, 5, 10, 20, 35, 50, 100]
warm_start = [True, False]

# Setting up parameter grid
hyperparams = {'n_restarts_optimizer': n_restarts_optimizer, 'max_iter_predict': max_iter_predict, 'warm_start': warm_start}

# Run GridSearch CV
gd=GridSearchCV(estimator = GaussianProcessClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Adaboost

# In[ ]:


# Parameter's to search
n_estimators = [10, 25, 50, 75, 100, 125, 150, 200]
learning_rate = [0.001, 0.01, 0.1, 0.5, 1, 1.5, 2]

# Setting up parameter grid
hyperparams = {'n_estimators': n_estimators, 'learning_rate': learning_rate}

# Run GridSearch CV
gd=GridSearchCV(estimator = AdaBoostClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### K Nearest Neighbours

# In[ ]:


# Parameter's to search
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
algorithm = ['auto']
weights = ['uniform', 'distance']
leaf_size = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]

# Setting up parameter grid
hyperparams = {'algorithm': algorithm, 'weights': weights, 'leaf_size': leaf_size, 
               'n_neighbors': n_neighbors}

# Run GridSearch CV
gd=GridSearchCV(estimator = KNeighborsClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Random Forest

# In[ ]:


# Parameter's to search
n_estimators = [10, 25, 50, 75, 100]
max_depth = [3, None]
max_features = [1, 3, 5, 7]
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [2, 4, 6, 8, 10]

# Setting up parameter grid
hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

# Run GridSearch CV
gd=GridSearchCV(estimator = RandomForestClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Extra Trees

# In[ ]:


# Parameter's to search
n_estimators = [10, 25, 50, 75, 100]
max_depth = [3, None]
max_features = [1, 3, 5, 7]
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [2, 4, 6, 8, 10]

# Setting up parameter grid
hyperparams = {'n_estimators': n_estimators, 'max_depth': max_depth, 'max_features': max_features,
               'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

# Run GridSearch CV
gd=GridSearchCV(estimator = ExtraTreesClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Bagging Classifier

# In[ ]:


# Parameter's to search
n_estimators = [10, 15, 20, 25, 50, 75, 100, 150]
max_samples = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 50]
max_features = [1, 3, 5, 7]

# Setting up parameter grid
hyperparams = {'n_estimators': n_estimators, 'max_samples': max_samples, 'max_features': max_features}

# Run GridSearch CV
gd=GridSearchCV(estimator = BaggingClassifier(), param_grid = hyperparams, 
                verbose=True, cv=5, scoring = "accuracy")

# Fitting model and return results
gd.fit(X_train, y_train)
print(gd.best_score_)
print(gd.best_estimator_)


# #### Gaussian Naive Bayes

# Gaussian Naive Bayes doesn't have parameters to tune, so we're stuck with the current score. This algorithm is known to be designed to work best on text data (e.g. once passed into a matrix), so perhaps in comparison to the other algorithms, it's less of a surprise to see it performing less favourably on the Titanic dataset.

# ### Model re-training

# We're into round three now, and hopefully optimising more of our models and really squeezing every last drop of predictive out of the dataset. I will run the same procedure as before and stack the latest results against the previous two. The only change now is that I can specify exact parameters as per the above GridSearchCV results to yield even better performance scores.

# In[ ]:


# Initiate tuned models
ran = RandomForestClassifier(n_estimators=25,
                             max_depth=3, 
                             max_features=3,
                             min_samples_leaf=2, 
                             min_samples_split=8,  
                             random_state=1)

knn = KNeighborsClassifier(algorithm='auto', 
                           leaf_size=1, 
                           n_neighbors=5, 
                           weights='uniform')

log = LogisticRegression(C=2.7825594022071245,
                         penalty='l2')

xgb = XGBClassifier(learning_rate=0.0001, 
                    n_estimators=10,
                    random_state=1)

gbc = GradientBoostingClassifier(learning_rate=0.0005,
                                 n_estimators=1250,
                                 random_state=1)

svc = SVC(probability=True)

ext = ExtraTreesClassifier(max_depth=None, 
                           max_features=3,
                           min_samples_leaf=2, 
                           min_samples_split=8,
                           n_estimators=10,
                           random_state=1)

ada = AdaBoostClassifier(learning_rate=0.1, 
                         n_estimators=50,
                         random_state=1)

gpc = GaussianProcessClassifier()

bag = BaggingClassifier(random_state=1)

# Lists
models = [ran, knn, log, xgb, gbc, svc, ext, ada, gnb, gpc, bag]         
scores_v3 = []

# Fit & cross-validate
for mod in models:
    mod.fit(X_train, y_train)
    acc = cross_val_score(mod, X_train, y_train, scoring = "accuracy", cv = 10)
    scores_v3.append(acc.mean())


# In[ ]:


# Creating a table of results, ranked highest to lowest
results = pd.DataFrame({
    'Model': ['Random Forest', 'K Nearest Neighbour', 'Logistic Regression', 'XGBoost', 'Gradient Boosting', 'SVC', 'Extra Trees', 'AdaBoost', 'Gaussian Naive Bayes', 'Gaussian Process', 'Bagging Classifier'],
    'Original Score': scores,
    'Score with feature selection': scores_v2,
    'Score with tuned parameters': scores_v3})

result_df = results.sort_values(by='Score with tuned parameters', ascending=False).reset_index(drop=True)
result_df.head(11)


# In[ ]:


# Plot results
sns.barplot(x='Score with tuned parameters', y = 'Model', data = result_df, color = 'c')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
plt.xlim(0.82, 0.86)


# Round 3 results in and Gaussian Process remains our strongest model. I actually omitted parameters for a few models where I saw the score had dropped subsequent to specifying them - I presume this could be because I did not test the full set of parameters in those cases. So the Gaussian Process score remains as before but for the Extra Trees Classifier, it has shot right up into 2nd place after a 2% jump in accuracy. Aside from the Gaussian Naive Bayes, each model now predicts with over 84% accuracy which I am really pleased to see. A strong set of final models which I can now carry forward into the final step 4 - Voting Classifier.

# ### Round 4: Voting Classifier

# Voting is one of the simplest ways of combining the predictions from multiple machine learning algorithms.
# 
# It works by first creating two or more standalone models from your training dataset. A Voting Classifier can then be used to wrap your models and average the predictions of the sub-models when asked to make predictions for new data.
# 
# There are two types of Voting Classifier, hard or soft. I saw a very simple explanation on a Q&A forum which i'll share with you now to explain the difference in approach:
# 
# - Suppose you have probabilities: 0.45 0.45 0.90
# - The hard voting would give you a score of 1/3 (1 vote in favour and 2 against), so it would classify as a "negative".
# - Soft voting would give you the average of the probabilities, which is 0.6, and would be a "positive".
# 
# The code below completes both a hard and soft voting classifier on all 11 models, minus the Gaussian Naive Bayes which, due to it's poorer score, I will omit from this final step. I will then compare the scores of either classifier, before selecting the best to proceed with making final predictions.

# In[ ]:


#Hard Vote or majority rules w/Tuned Hyperparameters
grid_hard = VotingClassifier(estimators = [('Random Forest', ran), 
                                           ('Logistic Regression', log),
                                           ('XGBoost', xgb),
                                           ('Gradient Boosting', gbc),
                                           ('Extra Trees', ext),
                                           ('AdaBoost', ada),
                                           ('Gaussian Process', gpc),
                                           ('SVC', svc),
                                           ('K Nearest Neighbour', knn),
                                           ('Bagging Classifier', bag)], voting = 'hard')

grid_hard_cv = model_selection.cross_validate(grid_hard, X_train, y_train, cv = 10)
grid_hard.fit(X_train, y_train)

print("Hard voting on train set score mean: {:.2f}". format(grid_hard_cv['train_score'].mean()*100)) 
print("Hard voting on test set score mean: {:.2f}". format(grid_hard_cv['test_score'].mean()*100))


# In[ ]:


grid_soft = VotingClassifier(estimators = [('Random Forest', ran), 
                                           ('Logistic Regression', log),
                                           ('XGBoost', xgb),
                                           ('Gradient Boosting', gbc),
                                           ('Extra Trees', ext),
                                           ('AdaBoost', ada),
                                           ('Gaussian Process', gpc),
                                           ('SVC', svc),
                                           ('K Nearest Neighbour', knn),
                                           ('Bagging Classifier', bag)], voting = 'soft')

grid_soft_cv = model_selection.cross_validate(grid_soft, X_train, y_train, cv = 10)
grid_soft.fit(X_train, y_train)

print("Soft voting on train set score mean: {:.2f}". format(grid_soft_cv['train_score'].mean()*100)) 
print("Soft voting on test set score mean: {:.2f}". format(grid_soft_cv['test_score'].mean()*100))


# Well it was pretty close, but for the test set the soft voting classifier came out just on top, so I will proceed to use this as my final model for prediction and submission. I'll complete that final step below before wrapping up:

# ## 7. Final model prediction & submission

# In[ ]:


# Final predictions
predictions = grid_soft.predict(X_test)

submission = pd.concat([pd.DataFrame(passId), pd.DataFrame(predictions)], axis = 'columns')

submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission.csv', header = True, index = False)


# And there we go! In this kernel I have taken several steps to analyse, clean and engineer features before applying and fine-tuning a selection of classification models for a final test set score of ~86%. I have hopefully been clear with my code and sufficient in my explanation so that you've been able to read along and enjoy the ride with me. I do intend to return to this kernel to add in more explanation where I can do and, because I am a sucker for a competition, see if I can edge myself up the leaderboard any further. Some thoughts on how I could do this include:
# 
# - Exploring new features in the data
# - Using a predictive model to impute age, rather than a simple groupby operation
# - Investing more time in parameter tuning and model optimisation
# - Prioritising fewer models and thus cutting more redundant features
# - Exploring deep learning application
# 
# As mentioned at the beginning, I would love to hear your feedback on my work, including what you found helpful or what perhaps needs clearer explanation. I'll do my best to help! Thanks for reading, and assuming that you are also participating in this competition - good luck!
