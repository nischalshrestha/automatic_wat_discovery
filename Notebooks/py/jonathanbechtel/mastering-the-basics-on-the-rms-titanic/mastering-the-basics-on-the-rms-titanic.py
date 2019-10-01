#!/usr/bin/env python
# coding: utf-8

# The Titanic dataset is a great way to hone your chops on the basics of Data Science.  Being my first foray as a Kaggler, I figured I'd use this exercise as a way to carefully walk myself through the basics of data mining with an emphasis on the following details:
# 
#  - Careful data handling and categorization that removes bias
#  - Thoughtful feature selection that adds extra value to the model
#  - Simple but deliberate Cross Validation that prevents overfitting
#  - Astute analysis post-hoc to determine the best roads for future improvements
# 
# So, with that said, here's how the journey went.
# 
# **Step 1:  Load the Data.**
#  
# The one detail to note is that I set the 'PassengerId' column as the index, which makes future munging more convenient since it's what's used to identify who survives and who doesn't.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm

#Read files into the program
test = pd.read_csv("../input/test.csv", index_col='PassengerId')
train = pd.read_csv("../input/train.csv", index_col='PassengerId')


# If we take a look at our training set we get the following:

# In[ ]:


train.head(5)


# We're going to do three things to start off:
# 
#  1. Store the 'Survived' column as its own separate series and delete it from the 'Train' dataset.
#  2.  Concatenate the training and testing set to fill in and parse all the data at once.
#  3.  Drop two columns:  'Embarked' and 'Ticket.'
# 
# Embarked demarcates what port a passenger was picked up at, and because the Titanic stopped only at major cities and it denotes details of an event that happened *before* the ship actually sank I'm going to assume it has no predictive value.  
# 
# Logic for dropping the ticket number is similar.......if there's causal information to be inferred from it I'm not sure what it is, so at the risk of carrying variables that just add noise we're going to ax them:

# In[ ]:


y = train['Survived']
del train['Survived']

train = pd.concat([train, test])

#Drop unnecessary columns
train = train.drop(train.columns[[6,9]], axis=1)  


# So now we get:

# In[ ]:


train.head(5)


# **Categorical Data:  Encoding and Feature Generation**
# 
# All  the operations in Sci-kit learn operate on integers so any data that exists as text needs to be classified into digits.  
# 
# Ie, a category like 'Sex' should be  either 0 or 1.  Something with three options (like Pclass) should be 0,1,2 and so on.  
# 
# The easiest way to do this (IMO) is with the [LabelEncoder()][1] method in SciKitLearn.  It categorizes all the different data in a Series and translates it into sequential digits.   You can do this in one fell swoop with the method fit_transform().
# 
# Let's do that for PClass and Sex:
# 
#   [1]: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

# In[ ]:


train['Sex'] = LabelEncoder().fit_transform(train.Sex)
train['Pclass'] = LabelEncoder().fit_transform(train.Pclass)


# For the 'Cabin' feature we're going to first do a little bit of data transformation.  
# 
# Might the deck someone stays on have some predictive value for what kind of passenger they were?  Perhaps the 'C' deck was for passengers of a lower financial ilk, while the 'F' deck was the equivalent of the Penthouse, only reserved for the cream of the crop.
# 
# To see we're going to extract the first letter of each passenger's cabin (if it exists) using the 'lambda x' feature in Python, and then encode it.  We change the np.nan values to 'X' so all data is the same type, allowing it to be labeled.

# In[ ]:


train['Cabin'] = train.Cabin.apply(lambda x: x[0] if pd.notnull(x) else 'X')
train['Cabin'] = LabelEncoder().fit_transform(train.Cabin)


# Looking at these three categories we now have:

# In[ ]:


train[['Pclass', 'Sex', 'Cabin']][0:3]


# grreeeaaaat.  
# 
# **Missing Data**
# 
# Now let's look to see if we have any missing data:

# In[ ]:


train.info()


# Clearly there's an important amount of missing data in the 'Age' category.  To fill it we're going to use the median age of that passengers Class and Sex, which will be accessed via the groupby method in Pandas:

# In[ ]:


train.groupby(['Pclass', 'Sex'])['Age'].median()


# And then set using Lambda x

# In[ ]:


train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.replace(np.nan, x.median()))
train.iloc[1043, 6] = 7.90


# The missing cell for 'Fare' was achieved in a similar fashion.  
# 
# **Feature Generation:  Family Size and Greeting**
# 
# We're going to create a new feature called 'Family Size' that's the sum of the 'Parch' and 'SibSp' features.  The idea being that maybe large families were at increased risk of not getting on a boat together,  or that maybe people with children were given preference over singles.  
# 
# This can be accomplished in one line:

# In[ ]:


train['Family_Size'] = train.SibSp + train.Parch


# Next we're going to use string processing to extract the greeting that was used for each passenger on the name of their ticket.  
# 
# Conveniently each person's name is used with a greeting that begins with a capital letter and ends with a period like so:

# In[ ]:


train['Name'].iloc[0]


# So we'll extract each of these labels using a for loop by using the Python method split() to break up each name into an array of words, and then evaluate them using the isupper() and endswith() methods in a for loop:

# In[ ]:


#Used to create new pd Series from Name data that extracts the greeting used for their name to be used as a separate variable
def greeting_search(words):
    for word in words.split():
        if word[0].isupper() and word.endswith('.'):
            return word


# We'll then apply this function to the 'Name' column:

# In[ ]:


train['Greeting'] = train.Name.apply(greeting_search)


# Which yields the following greetings:

# In[ ]:


train['Greeting'].value_counts()


# Looks interesting to me.  
# 
# **Useless Trivia:**  I had no idea what 'Jonkheer' meant before doing this exercise, but it turns out it's a Dutch word for Royalty, increasing my naive hope that I might be onto something here.
# 
# However, it's not a good idea to have large amounts of teensy-weensy variable because it can create outliers in your data that'll skew your results.  
# 
# So we'll take all the greetings that occur 8 or less times and classify them under the moniker 'Rare', encode it, and then delete the Series called 'Name' since we don't need it anymore.

# In[ ]:


train['Greeting'] = train.groupby('Greeting')['Greeting'].transform(lambda x: 'Rare' if x.count() < 9 else x)
del train['Name']


# Then tranform the data and drop the 'Name' series since it's no longer needed.

# In[ ]:


train['Greeting'] = LabelEncoder().fit_transform(train.Greeting)


# **Categorical Coding**
# 
# Continuous order has a precise hierarchy to it.  Someone who paid $50 for a ticket definitely paid more than someone who paid $30.  
# 
# However, someone who's greeted as 'Master' doesn't have more of a greeting than someone who's approached as 'Miss', but if one as coded as a 6 and the other a 1, SKLearn will think the one is 6 times as large as the other.  
# 
# But they're actuallly separate yes/no categorizations packed on top of one another.  
# 
# So what we want to do is re-code a Series into a package of yes/no decisions demarcated as 0 or 1 depending on which option they were.
# 
# Ie, Someone's Passenger class should be denoted as [0, 0, 1], [1, 0, 0], or [0, 1, 0] depending on which of the three classes they are.  
# 
# Pandas has a useful tool to do this called pd.get_dummies, which takes a series of encoded and then unpacks it into the appropriate number of yes/no columns.
# 
# For example, we can take the 'Pclass' series and use pd.get_dummies like this:

# In[ ]:


#Categorical coding for data with more than two labels
Pclass = pd.get_dummies(train['Pclass'], prefix='Passenger Class', drop_first=True)


# And have it turned into this:

# In[ ]:


Pclass.head(5)


# **Important:**  You might notice there's an option called 'drop_first' which is set to 'True.'
# 
# That means the first variable in the series is excluded, which is important for avoiding something called collinearity, which you can read more about [here][1].
# 
# To be honest, probably not that important for this dataset, but a useful habit to keep in mind, especially if you work with Time Series data.
# 
# We can transform our other categorical variables in the same way:
# 
#   [1]: https://en.wikipedia.org/wiki/Multicollinearity

# In[ ]:


Greetings = pd.get_dummies(train['Greeting'], prefix='Greeting', drop_first=True)
Cabins = pd.get_dummies(train['Cabin'], prefix='Cabin', drop_first=True)


# **Scaling Data**
# 
# Continuous data needs to be scaled so the values that get fed into it have similar properties.  In particular, the L2 penalization assumes all data follows the standard normal distribution.  
# 
# Some algorithms like the Support Vector Machine are sensitive to scale, and not scaling it can significantly skew its results.  Not scaling isn't always going to be a cardinal sin, but it's good practice to do it to help the consistency of your results and make your data more robust to overfitting and false signals.  
# 
# There are a number of ways to scale data, but the most popular method is probably taking each value and subtracting the mean from it and then dividing by the standard deviation, like so:

# In[ ]:


#Scale Continuous Data
train['SibSp_scaled'] = (train.SibSp - train.SibSp.mean())/train.SibSp.std()
train['Parch_scaled'] = (train.Parch - train.Parch.mean())/train.Parch.std()
train['Family_scaled'] = (train.Family_Size - train.Family_Size.mean())/train.Family_Size.std()
train['Age_scaled'] = (train.Age - train.Age.mean())/train.Age.std()
train['Fare_scaled'] = (train.Fare - train.Fare.mean())/train.Fare.std()


# **Final Processing**
# 
# Here's what we'll do for the final data processing steps:
# 
#  1. Drop the columns that have been transformed into something else.
#  2.  Concatenate the dataframes that were created with pd.get_dummies()
#  3.  split the data back into its training and test sets now that everything's been munged, scaled, filled and transformed.

# In[ ]:


#Drop unmodified data since it's no longer needed
train = train.drop(train.columns[[0,2,3,4,5,6,7,8]], axis=1)

#Concat modified data to be used for analysis, set to X and y values
data = pd.concat([train, Greetings, Pclass, Cabins], axis=1)

#Split the data back into its original training and test sets
test = data.iloc[891:]
X = data[:891]


# **Cross Validation**
# 
# We need to cross-validate to avoid overfitting.  
# 
# Lots of people use the K-fold technique, but the only method I've ever used myself is creating a separate Cross Validation set from the original training data, so that's what we'll do here using train_test_split() from SKLearn

# In[ ]:


#Create cross - validation set 
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.6)


# Then we'll initialize the Logistic Regression Algorithm:

# In[ ]:


clf = LogisticRegression()


# Next we're going to iterate through multiple values of C on the Cross Validation set to find which one creates the smallest amount of error.

# In[ ]:


def find_C(X, y):
    Cs = np.logspace(-4, 4, 10)
    score = []  
    for C in Cs:
        clf.C = C
        clf.fit(X_train, y_train)
        score.append(clf.score(X, y))
  
    plt.figure()
    plt.semilogx(Cs, score, marker='x')
    plt.xlabel('Value of C')
    plt.ylabel('Accuracy on Cross Validation Set')
    plt.title('What\'s the Best Value of C?')
    plt.show()
    clf.C = Cs[score.index(max(score))]
    print("Ideal value of C is %g" % (Cs[score.index(max(score))]))
    print('Accuracy: %g' % (max(score)))


# Here's a brief overview of what went on here:
# 
#  - We created an array of C values that range from 10^-4 to 10^4.  
# For each value of C we:
#  - fit the data to X_train and y_train
#  -  score it on X_val and y_val
#  - append the score into the list 'score', and then plot the values in 'score' to their corresponding C values
#   - set the value of C in Logistic Regression to the value that had the highest accuracy on the cross validation data.  
# 
# Running it gives us this:

# In[ ]:


find_C(X_val, y_val)


# As you can see, the choice of C can have a pretty large impact on the results!  

# **Final Answers**
# 
# We'll make our predictions on the test data and write them to a .csv file here:

# In[ ]:


answer = pd.DataFrame(clf.predict(test), index=test.index, columns=['Survived'])
answer.to_csv('answer.csv')


# **Analyzing the Results**
# 
# Now that our analysis is done it might be helpful to do some post-hoc analysis to see what was going on underneath the hood.  
# 
# An additional benefit of feaure scaling is that it makes it much easier to compare coefficients of different variables.  
# 
# Let's store and look at them now:

# In[ ]:


coef = pd.DataFrame({'Variable': data.columns, 'Coefficient': clf.coef_[0]})
coef


# As expected, sex and passenger class had large effects on final outcome.  It's also useful to note that the greetings had a very large effect as well, with everything else being more modest.
# 
# **Precision and Recall**
# 
# Classifications algorithms can have a high level of accuracy without necessarily being well-tuned to the problem at hand.  
# 
# To take a closer look at how our algorithm is handling the data, let's evaluate it using two measures known as Precision and Recall.  
# 
# Simply put, they measure how accurately your algorithm accepts correct answers and rejects false ones.  [You can read more about them here][1].
# 
# Once you have these measures you can evaluate them using an F1-Score, which is:
# 
# 2 * Precision * Recall / (Precision + Recall)
# 
# SK learn has its own method for doing it, which you can see [here][2] , but somestimes it's useful to roll things out yourself to make sure you understand what's going on.  
# 
# So here's a simple function that calculates all three measures:
# 
#   [1]: https://en.wikipedia.org/wiki/Precision_and_recall
#   [2]: http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

# In[ ]:


results = y_val.tolist()
predict = clf.predict(X_val)

def precision_recall(predictions, results):
    
    tp, fp, fn, tn, i = 0.0, 0.0, 0.0, 0.0, 0
    
    while i < len(results):
        
            if predictions[i] == 1 and results[i] == 1:
                tp = tp + 1
            elif predictions[i] == 1 and results[i] == 0:
                fp = fp + 1
            elif predictions[i] == 0 and results[i] == 0:
                tn = tn + 1
            else: 
                fn = fn + 1
            i = i + 1
    
    precision = tp / (tp + fp)
    recall = tn / (tn + fn)
    f1 = 2*precision*recall / (precision + recall)
    print("Precision: %g, Recall: %g, f1: %g" % (precision, recall, f1))


# If we run it, we get:	

# In[ ]:


precision_recall(predict, results)


# So the algorithm is a little better at throwing out the baddies than recognizing the goodies, but is reasonably sound at both.
# 
# Okay, so that about wraps it up.  Critiques, questions, and improvements are all welcome and appreciated!  
# 
# Here are some future steps you could take with this data:
# 
#  - Different algorithms that might fit the data more tightly (SVM, Random Forest, etc)
#  - Detection of outliers that might be skewing data (I have a feeling this might be one of the most useful    
#     steps with such a limited data set).
#  - Playing around with the addition or subtraction of features to see if they improve performance.  (**hint:** I actually improved my score by removing one of the initial categories entirely, so there is some noise in this script).
# 
# Hope this helped, enjoy!
