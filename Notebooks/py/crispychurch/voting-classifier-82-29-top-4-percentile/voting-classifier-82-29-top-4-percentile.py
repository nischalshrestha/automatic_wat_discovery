#!/usr/bin/env python
# coding: utf-8

# # **Introduction**
# 
# This notebook is a basic-to-intermediate level walkthrough of the Titanic Classification Challenge, and how I approached it. It makes use of some basic data exploration techniques, and makes use of an ensemble-model for classification. 
# I managed to get a public LB score of about 82% after more than 30 tries, more than half of which were for different models, like Random Forests, XGBoost etc. My score drastically improved after I plugged in this Voting classifier, so, robustness of model rewards a lot here.
# This was my first real and independently done Machine Learning problem and I learnt a lot, and hopefully, this helps you. Also, this is **my first Kaggle kernel**, so please feel free to point out corrections and to suggest better approaches.
# 
# ## **Before Starting out**
# 
# The Titanic dataset is a delightful challenge, having a pretty compact set of features and not many rows, which is perfect for us beginners. Its design and inherent density of interweaving contexts, as I found out, constantly reward the analyst for their time. But, also, as a warning sign, the data is rather volatile, probably because its dependent quantity depends on a lot of human decisions and chance.
# I would advise a read-through of [what exactly took place](https://en.wikipedia.org/wiki/RMS_Titanic#Maiden_voyage) on that night of April, 1912. Apart from that, the [general structure](https://en.wikipedia.org/wiki/RMS_Titanic#Dimensions_and_layout) and build of the ship, may help a bit. And lastly, a very basic idea of the socio-economic conditions at the time may help you understand the results better, even if it doesn't help you improve the model.
# 
# Ok, lets go.
# 

# In[ ]:


# Importing required libaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))


# ## **Data Exploration**

# In[ ]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/gender_submission.csv')

train_data.head()


# Looking at this gives a general idea about what's on our hands. PassengerId is probably not going to be important, going by logic. There are already three NaNs out of five, in Cabin column. So, we better check for missing values as well. Pclass is the estimated socio-economic "class" of the passenger. And we all know that class *was* a deciding factor, in the rescue. And of course, Sex, Age, and family information will definitely play a role.
# 
# Apart from the usual, does a person's name have a say in saving their life? Based on what is known to us already, and after some research, it -seems like it might. On further observation, we notice that there are uncommon titles that passengers possess. Although this might directly correlate to their Class, we'll extract it anyway, and see if it helps.
# 
# Ticket, Fare and Cabin are different from the rest and may or may not correlate highly with Survived. We might be able to extract something from Ticket, but I can't see anything of interest in it right now.

# ## **Checking for missing data**

# In[ ]:


train_data.isnull().sum()


# In[ ]:


test_data.isnull().sum()


# Ok. So there's good news and bad news. The good news is that there are only 4 columns in both train_data and test_data that have any data missing. The bad news, however, is that 2 of the columns have serious amounts of data missing. Handling Embarked and Fare should be easy. As far as Age is concerned, we should be able to impute the missing rows and use them. But, for Cabin, unfortunately, there are just too many empty rows and its too diffcult to predict its string values with any confidence. Still, lets give it a chance without spending too much time on it and decide later.

# Let me just fill in missing values for Embarked and Fare.
# I'll fill the Embarked ones with S, because Southampton is the most common source.
# I'll fill Fare with the mean, because its just one row, and we might end up not using the feature.

# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())


# In[ ]:


train_data['Embarked'].isnull().sum() + test_data['Fare'].isnull().sum()


# ## **Visualising the data**

# In[ ]:


sns.barplot(x=train_data['Pclass'], y=train_data['Survived'])


# So, the target variable seems to be significantly dependent on Pclass as we'd expected.

# In[ ]:


sns.barplot(x=train_data['Sex'], y=train_data['Survived'])


# Wow. This implies a pretty heavy correlation. Survival is extremely skewed towards females.

# In[ ]:


sns.pointplot(x=train_data['Age'], y=train_data['Survived'])


# This got pretty messy. But, should we be surprised? Well..no. Age, intuitively, is crucial to survival, but the extensive range of ages (~1 to ~80) adds a LOT of points, and hence a lot of scope for other statistically important features to affect proceedings. Perhaps we should divide this range into 4-5 bins, according to trends. 

# In[ ]:


sns.barplot(x=train_data['SibSp'], y=train_data['Survived'])


# In[ ]:


sns.barplot(x=train_data['Parch'], y=train_data['Survived'])


# There is a high variance in both SibSp and Parch. But, they are not without their uses. We could combine both of them into one variable that handles all the family information.
# Lets engineer some features on our own first and then get back to visualising them.

# ## **Feature Engineering**

# **Title**

# In[ ]:


train_data = train_data.join(train_data['Name'].str.split(',', 1, expand=True).rename(columns={0:'LastName', 1:'FName'}))
train_data = train_data.join(train_data['FName'].str.split('.', 1, expand=True).rename(columns={0:'Title', 1:'FirstName'}))
train_data['Title'] = train_data['Title'].str.strip()
train_data.drop(['Name', 'FName'], axis=1, inplace=True)
train_data.head()


# In[ ]:


test_data = test_data.join(test_data['Name'].str.split(',', 1, expand=True).rename(columns={0:'LastName', 1:'FName'}))
test_data = test_data.join(test_data['FName'].str.split('.', 1, expand=True).rename(columns={0:'Title', 1:'FirstName'}))
test_data['Title'] = test_data['Title'].str.strip()
test_data.drop(['Name', 'FName'], axis=1, inplace=True)
test_data.head()


# So, now we've carved out First Name, Last Name and Title from the Name column. Out of these, Title remains the most useful, because of the aristocratic titles, which might have a say. We *might* be able to work with Last Name and do something, but First Name is out of the picture definitely.

# **Family Survival Ratio and Friends Survival Ratio**

# In[ ]:


# Initialising with -1s
family_ratio_train = [-1]*len(train_data)
family_ratio_test = [-1]*len(test_data)

friends_ratio_train = [-1]*len(train_data)
friends_ratio_test = [-1]*len(test_data)

for i in range(len(train_data)):
    #print('i = '+str(i))
    family_survive_list = []
    friends_survive_list = []
    for j in range(len(train_data)):
        if ((train_data['LastName'][i] == train_data['LastName'][j]) & (train_data['Fare'][i] == train_data['Fare'][j])):
            family_survive_list.append(train_data['Survived'][j])
        elif (train_data['Ticket'][i] == train_data['Ticket'][j]):
            friends_survive_list.append(train_data['Survived'][j])
    if len(family_survive_list) > 1:
        family_ratio_train[i] = np.mean(family_survive_list)
    else:
        family_ratio_train[i] = -1
    if len(friends_survive_list) > 1:
        friends_ratio_train[i] = np.mean(friends_survive_list)
    else:
        friends_ratio_train[i] = -1

i = 0
j = 0

for i in range(len(test_data)):
    #print('i = '+str(i))
    family_survive_list = []
    friends_survive_list = []
    for j in range(len(train_data)):
        if ((test_data['LastName'][i] == train_data['LastName'][j]) & (test_data['Fare'][i] == train_data['Fare'][j])):
            family_survive_list.append(train_data['Survived'][j])
        elif (test_data['Ticket'][i] == train_data['Ticket'][j]):
            friends_survive_list.append(train_data['Survived'][j])
    if len(family_survive_list) > 1:
        family_ratio_test[i] = np.mean(family_survive_list)
    else:
        family_ratio_test[i] = -1
    if len(friends_survive_list) > 1:
        friends_ratio_test[i] = np.mean(friends_survive_list)
    else:
        friends_ratio_test[i] = -1

train_data['Family Survival Ratio'] = family_ratio_train
test_data['Family Survival Ratio'] = family_ratio_test

train_data['Friends Survival Ratio'] = friends_ratio_train
test_data['Friends Survival Ratio'] = friends_ratio_test


# These features are derived from an ingenious feature I found in [S.Xu's kernel ](https://www.kaggle.com/shunjiangxu/blood-is-thicker-than-water-friendship-forever). I got the idea of such a pattern when I went through his very well-written notebook, and decided to make my own implementation of it.
# 
# Now, put yourself in the situation. You're on a ship that's about to go down. How will you proceed? If you're travelling alone, you would just get the hell out of there as soon as possible. But, if you're travelling with family, you would make sure that none of your family membres get left behind. And this holds true for extended family relatives and friends as well. This human tendency is what I've tried to  capture in these two features.
# 
# Family Survival Ratio holds the** ratio of members in a person's family that survived**, and Friends Survival Ratio does the same for friends.
# Families are derived by comparing Last Name AND Fare, because there are passengers who are not related but have the same Last Name. Friends are derived from the Ticket column IF their Last Names are different.
# 
# Also, this code section is quite slow to execute because of the double loop, but I'll try and find a better alternative.

# In[ ]:


# Creating the FamilySize variable as suggested before
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1


# In[ ]:


# Filling the empty Cabin rows with U, for unknown.
train_data['Cabin'] = train_data['Cabin'].fillna('U')
test_data['Cabin'] = test_data['Cabin'].fillna('U')
# Choosing only the first character of the Cabin string, to denote Deck.
train_data['Cabin'] = train_data['Cabin'].str[0]
test_data['Cabin'] = test_data['Cabin'].str[0]


# In[ ]:


# Rearranging columns and creating a joint variable so that fewer lines of code are required
train_data = train_data[['Pclass','Title','Sex','Age','FamilySize','Family Survival Ratio',
                         'Friends Survival Ratio','Fare','Cabin','Embarked','Survived']]

test_data = test_data[['Pclass','Title','Sex','Age','FamilySize','Family Survival Ratio',
                       'Friends Survival Ratio','Fare','Cabin','Embarked']]

total_data = [train_data, test_data]


# In[ ]:


# Lets take a look at train_data first
train_data.head()


# Ok. We're good to go. We see that there are a sizeable amount of categorical variables. To handle them, we will create dummy variables.

# **Categorical Variables**

# Now, we'll handle the categorical variables. Some of the variables are naturally categorical, like Title, Sex, Cabin. And then, we will bin some of the continuous ones, thus turning them into categorical variables, like Fare and Age. The reason why we do that is that the target variable is not always proportional to the continuous variables. And in some truly statistical models like logistic regression (for classification), and linear regression (for regression), the model might misconceive some proportional relationship between, say, Age and Survived. But, we know that the relationship between Age and Survived is not monotonous at all. So, I think it is a good practice to bin important, non-monotonous, continous variables, regardless of the model used.

# In[ ]:


for dataset in total_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Sir', 'Don','Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Capt', 'Col', 'Major'], 'Occ')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Occ": 5, "Rare": 6}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    dataset['Sex'] = dataset['Sex'].map( {"female": 1, "male": 2} )
    
    cabin_mapping = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8,"U":9}
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    
    dataset['Embarked'] = dataset['Embarked'].map( {'C': 1, 'Q': 2, 'S': 3} ).astype(int)
    
    dataset.loc[ dataset['Fare'] <= 10, 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10) & (dataset['Fare'] <= 40), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 80), 'Fare'] = 3
    dataset.loc[ dataset['Fare'] > 80, 'Fare'] = 4


# First up, Title. Aristocratic Last Names, which would have a say in matters of life or death, have been put into a category called "Rare". Apart from that, all the occupational titles, like Dr., Rev., Capt., etc. have been put in a separate category called Occ. Then, some standard titles and their equivalents have been put together.
# The next 4 bunches of lines are nothing but mapping each category to a number, because most models don't work with strings.
# And at the end, Fare has been divided into 4 bins, according to its distribution. Much of the bin edges have been decided by trial-and-error so go ahead and try values of your own.
# 

# In[ ]:


train_data.head()


# In[ ]:


# Just some steps for a smoother implementation
y = train_data['Survived'].values

train_data = pd.DataFrame(train_data, columns = ['Pclass','Title','Sex','Age','FamilySize',
                                                 'Family Survival Ratio','Friends Survival Ratio',
                                                 'Fare','Embarked'])    

test_data = pd.DataFrame(test_data, columns = ['Pclass','Title','Sex','Age','FamilySize',
                                                 'Family Survival Ratio','Friends Survival Ratio',
                                                 'Fare','Embarked'])    
X_total = pd.concat([train_data, test_data])


# In[ ]:


# Filling in missing values for Age
from fancyimpute import KNN
X_select = X_total[['Title', 'Age', 'FamilySize']]
X_complete = KNN(k=50).complete(X_select)

X_complete[:,1] = np.trunc(X_complete[:,1])

X_total['Age'] = X_complete[:,1]


# I have used fancyimpute for filling in the missing values. Again, a personal choice, nothing else. If you find an estimator with better performance, do let me know. Next we'll bin Age, just like we did with Fare.

# In[ ]:


# Categorising age
complete_data = pd.DataFrame(X_total,columns=X_total.columns)

X_total = complete_data.values

def categ_age(x):
    if x<=8:
        return 1
    elif x<=25:
        return 2
    elif x<=40:
        return 3
    elif x<=50:
        return 3
    else:
        return 4

for i in range(len(X_total)):
    X_total[i,3] = categ_age(X_total[i,3])


# **Scaling feature values**

# In[ ]:


# Scaling feature values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_total)

X = X_scaled[:len(train_data),:]
X_final = X_scaled[len(train_data):,:]


# ## **Model Implementation**

# Suppose there's  a competition to count the number of lets say, jawbreakers in a jar. If you know where this is going, you are welcome to skip this section, but if you don't, read on. So, people come to take a look at the jar, lean in for a closer look and make some calculations. Then, they go to the organisers and give their estimate. Now, because we know statistics enough to know that "more makes merry", we'll bring our favourite chair and sit right next to the organisers and note down all the entries. After all of them are done, we can just take an average of all the values and give that as our entry. This is somewhat similar to what a voting classifier does. It asks all of the models what they think about the problem, and at the end takes an average, or in this case, picks the majority vote. This helps in balancing any errors that a particular model might face in front of a particular row in the dataset, since all the models are inherently and conceptually different to one another. This method of majority vote is referred to as "hard voting".
# 
# But, what we are going to use is a bit different, soft voting. Going back to the same problem, what if you know all the people who were participating in the competition, or even better, what if you asked them how they felt about their answer? You could ask them to rate their confidence in their guess on a scale of 1-10. This might help you in establishing a better average of values. You could, very easily, assign weights to their answers and then, take an average. This is known as "soft voting". And that is what we are going to use in our model.

# In[ ]:


# Fitting the Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[ ]:


estimators_list = []

dt = DecisionTreeClassifier()
estimators_list.append(('Decision Tree', dt))

rf = RandomForestClassifier(n_estimators=3001)
estimators_list.append(('Random Forest', rf))

ada = AdaBoostClassifier(n_estimators=1000,learning_rate=0.00001)
estimators_list.append(('AdaBoost', ada))

grad_boost = GradientBoostingClassifier(max_features='sqrt',n_estimators=1000,learning_rate=0.00001)
estimators_list.append(('Gradient Boosting', grad_boost))

knn = KNeighborsClassifier(n_neighbors=19)
estimators_list.append(('KNN', knn))

logreg = LogisticRegression()
estimators_list.append(('Logistic Regression', logreg))

svc = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
estimators_list.append(('SVC', svc))

ensemble = VotingClassifier(estimators_list, voting='soft')


# In this ensemble model, I have made use of 7 very different machine learning models. All the individual parameters have been tuned accordingly. I have set the voting to "soft", because it improved the accuracy. This is a basic implementation. You could try it with many different models and see if it helps.
# 
# Now, to fit the model and predict the results.

# In[ ]:


# Fitting the model on the train data
ensemble.fit(X,y)


# In[ ]:


#  Predicting for the test data
y_final_pred = ensemble.predict(X_final)


# In[ ]:


# Submission file
submission['Survived'] = y_final_pred
submission.to_csv('voting_submission.csv', index=False)


# ## **What now?**

# So, there it is. There are a lot of other things as well which I didn't include. I spent most of time visualising and exploring the data and trying to make new features, but I've only included the significantly successful ones. I tuned the parameters for different component-models as well, but I haven't included the particular code section for it. Also, I spent some time trying to figure out what component-models to use in this Voting Classifier, primarily by hit-and-trial. The ones included above are the ones that gave the best results. Also, strangely, I was getting a better result with sklearn's Gradient Boosting Classifier than XGBoost, so I replaced it.
# 
# As with everything I would say there is a scope for improvement. For me, most certainly, its about improving my Python skills. For the model, parameter tuning could help. I am also thinking about a new feature which groups according to ethnicities because I want to see if that was a major factor in saving lives or not. Also, extracting some useful information out of Ticket and Cabin would be great.
# 
# Well, thanks for reading and I'll see you around!

# ## **Credits**

# Here are some kernels which helped me get an idea on how to begin working on such a project:
# 
# https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic
# 
# https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python
# 
# https://www.kaggle.com/erikbruin/titanic-2nd-degree-families-and-majority-voting
# 
# https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83
# 
