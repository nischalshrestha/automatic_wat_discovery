#!/usr/bin/env python
# coding: utf-8

# ## Welcome Kaggler!
# 
# With this interactive diving course I invite you to learn some machine learning basics. This course is designed as a series of kernels that guides you through different topics and hopefully you can discover some hidden treasures that push you forward on your data science road. You don't have to pick the courses in sequence if you are only interested in some topics that are covered. If you are new I would recommend you to take them step by step one after another. ;-)
# 
# Just fork the kernels and have fun! :-)
# 
# * **Prepare to start**: Within this kernel we will prepare our data such that we can use it to proceed. Don't except nice feature selection or extraction techniques here because we will stay as simple as possible. Without a clear motivation we won't change any features. Consequently we are only going to explore how to deal with missing values and how to turn objects to numerical values. In the end we will store our prepared data as output such that we can continue working with it in the next kernel.
# * [MyClassifier](https://www.kaggle.com/allunia/titanic-dive-through-myclassifier): Are you ready to code your own classifier? Within this kernel you will build logistic regression from scratch. By implementing the model ourselves we can understand the assumptions behind it. This knowledge will help us to make better decisions in the next kernel where we will use this model and build some diagnosis tools to improve its performance.
# * [The feature cave](https://www.kaggle.com/allunia/titanic-dive-through-feature-cave): By using our own logistic regression model we will explore how we can improve by adding a bias term and why we should encode categorical features. 
# * [Feature scaling and outliers](https://www.kaggle.com/allunia/titanic-dive-through-feature-scaling-and-outliers): Why is it important to scale features and to detect outliers? By analyzing the model structure we will discover how our gradients and our model performance are influenced by these topics. 
# 

# ## Step into the water ...
# 
# Before we start we need to set up our equipment and load some packages:

# In[ ]:


# data analysis tools
import numpy as np 
import pandas as pd 

# data visualization
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns

# Let's have a look at our input files:
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# And of course we have to read in the data:

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
submission = pd.read_csv("../input/gender_submission.csv")


# Ok, now we are ready to take a step:

# In[ ]:


train.head()


# By using the head method of pandas we can gain a first impression of our data sets. In the case of the train data we can see that it includes the quantity we want to predict: the Survived label. This target is missing in the test set. Now its your turn: Use the head function to peek at the test and submission data:

# In[ ]:


# Your turn: Peek at the test data ;-)


# In[ ]:


# Your turn: Peek at the submission data ;-)


# ## Feature drops and types
# 
# Beside the differences between the data sets we can find that some features are not numerical for example the *Name*. As our models are based on mathematical equations we should turn them into numerical values as well. The name feature is complicated as it includes the first and last name and titles as well. In some cases we have is further information given like for passenger with id 2 in the train set:

# In[ ]:


train[train.PassengerId == 2].Name.values


# It seems to be that the women is named by her husband and her own maiden name is added in brackets. We will keep this in mind but exclude the name feature as a first attempt. Of course there could be useful information we will lose this way, but we should stay as simple as possible on our first diving course. We will use pandas drop method to do this. By using inplace=True we make sure that this operations works without a new assignment to the train set. And by using axis=1 we force it to work on a column label.
# 
# So instead of train = train.drop("Name"), we will use:

# In[ ]:


train.drop("Name", axis=1, inplace=True)


# Has it worked as expected?

# In[ ]:


train.head()


# Yes! Let's have a look at the other features: The *Ticket* feature is complicated as well so let's drop it:   

# In[ ]:


train.drop("Ticket", axis=1, inplace=True)


# Now its your turn: Drop the name and ticket feature from the test set. Hint: You can use a list of features to drop multiple features at one. In our case this would be ["Name", "Ticket"]. 

# In[ ]:


# Your turn: Drop the name and ticket feature from the test set.


# Back to the feature types. We can have a closer look by using pandas info method:

# In[ ]:


train.info()


# We habe to put some work into the sex, embarked and cabin feature. We could do so by mapping them to numerical values using dictionaries and pandas apply method. Using a lambda expression ([link to funny explanation video](http://https://www.youtube.com/watch?v=25ovCm9jKfA)) we can apply this map to each passenger:

# In[ ]:


sex_map = {"male": 0, "female": 1}
train.Sex = train.Sex.apply(lambda l: sex_map[l])
train.head()


# Ok, everything worked fine. Now it's again your turn: 

# In[ ]:


# Apply the sex_map to the test set.


# Let's continue with the embarked feature. But first of all, we should find out, how many different values the embarked feature can take:

# In[ ]:


train.Embarked.unique()


# Oh! We have encountered missing values that are represented by nan values. Before we proceed with our mapping it might by better to explore the missing values of all features of our data. 
# 
# ## Exploring missing values

# In[ ]:


train.isnull().sum()


# The isnull method returns true for each feature entry if it contains a nan value. By summing up all true values (true yields 1 and false yields 0) we can obtain the total number of nan values per feature. Instead of absolute values it might be more advantageous to look at the relative frequency. Let's visualize it:

# In[ ]:


nans_in_train = train.isnull().sum() / train.shape[0] * 100
nans_in_train = nans_in_train[nans_in_train > 0]

plt.figure(figsize=(10,4))
sns.barplot(x=nans_in_train.index.values, y=nans_in_train.values, palette="Set1")
plt.ylabel("Percentage of nans")
plt.ylim([0, 100])
plt.title("Missing values in train")


# Puuhh! Almost 77 % of the cabins are unknown in the train data!!! What about the test set?

# In[ ]:


# Your task: Compute the relative frequencies of missing values in the test set


# Reconstructing the cabin values for each passenger seems to be a horrendous and faulty task. Let's drop it!

# In[ ]:


# Your job: Drop the cabin feature from the train and test set.


# Ok, we are left with the embarked, age and fare feature. Let's try to replace the nan values with meaningful alternatives:

# ## Replacing missing values
# 
# Let's start with the embarkation! To whom does the missing values belong to?

# ### Missing embarkation

# In[ ]:


train[train.Embarked.isnull()]


# Ok, two women who travelled with the first class and payed the same ticket fare. Unfortunately we have already lost information by dropping features. But maybe it would be helpful to see if both had the same cabin or shared the same name. Let's restore the old information by reading in the original train data:

# In[ ]:


original_train = pd.read_csv("../input/train.csv")
original_train[train.Embarked.isnull()]


# Ahh! The two womed travelled together in cabin B28 and had the same ticket 113572. Let's have a look if we could find some people that share one or both of these features:

# In[ ]:


original_train[(original_train.Cabin == "B28") | (original_train.Ticket == "113572")]


# Unfortunately no!  Let's again stay simple and have a look at the absolute frequencies of embarkation. We will replace the nan value by the most frequent one:

# In[ ]:


plt.figure(figsize=(10,4))
sns.countplot(train.Embarked)


# Southampton is the most frequent port of embarktion. Hence we will use "S" to replace the nans. For this purpose let's use pandas fillna method again using inplace=True:

# In[ ]:


train.Embarked.fillna("S", inplace=True)


# Let's validate: Are there nans left for embarkation in the train set?

# In[ ]:


train.Embarked.isnull().value_counts()


# Of course there are many other ways to replace this missing nan values of both ladies. Perhaps passengers of different ticket classes embarked in different ports. Try to find it out if you like and build a replacement of your choice. ;-) 
# 
# Afterwards it's still your turn: 

# In[ ]:


# Replace the embarkation object values with numerical ones. 
# You can use a dictionary as we have done for the sex feature. 


# ### Missing fare
# 
# Let's continue with the missing fare in the test set:

# In[ ]:


test[test.Fare.isnull()]


# Let's again follow the simplest strategy and implace the fare by a common fare value. But which one to chose... the mean, the median or the mode? Maybe the distributions ot the fare in the train and test set can give us some hints:

# In[ ]:


plt.figure(figsize=(20,5))
train_fares = sns.distplot(train.Fare, kde=False, label="Train", norm_hist=True)
test_fares = sns.distplot(test.Fare.dropna(), kde=False, label="Test", norm_hist=True)
plt.title("Normed histogram of ticket fares")
train_fares.legend()
test_fares.legend()


# We have used pandas dropna to exclude the nan value to make our plot work. We obtained further insights:
# 
# * First of all: The distributions of the train and test sets are very similar. We don't have to worry that an implacement we gained from the train set might not work for the test set.
# * The distributions are right skewed and exhibit ourliers. Both are strong hints that tell us: "Do not use the mean!" Why? The mean is highly shifted towards higher values and does not reflect what is most common. Here one example:

# In[ ]:


example = np.array([1,2,3,1,2,1,3,99,1,4,5,2])
print(np.mean(example))
print(np.std(example))
print(np.median(example))


# You can see that the median would be a better choice. Let's use it!

# In[ ]:


print(train.Fare.median())
print(test.Fare.median())


# Again, we can see that the fare of train and test ist very similar. Now your task:

# In[ ]:


# Replace the nan value in the test set with the fare median 


# What do you think, is this a good choice? Why not? What if the fare depends somehow on the ticket class or the sex or the age? Even though Mr. Storey is only one person in the test set and we don't expect that our implacement has hugh impact on our performance, let's have some fun and make some nice plots to gain even more insights:

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(15,8))
sns.violinplot(x=train.Pclass, y=train.Fare, ax=ax[0,0])
sns.violinplot(x=train.Sex, y=train.Fare, ax=ax[0,1])
sns.violinplot(x=train.Embarked, y=train.Fare, ax=ax[1,0])
ax[1,1].scatter(train.Age.values, train.Fare.values)
ax[1,1].set_xlabel("Age")
ax[1,1].set_ylabel("Fare")


# Obviously the ticket fare depends has a strong relationship with the ticket class. That does make sense. Now it's your turn:

# In[ ]:


# plot the normalized fare histograms for train and test conditioned on the 3rd ticket class. 
# Then compute some statistics you like and make a new decision for the nan replacement of Mr Storey.
# Replace the our previous fare value with yours.


# ### Missing ages
# 
# We have already found that the age of 20 % of our passengers in the train and test data is unknown. We have to deal with this problem. A common way to start with would be to replace the nans with statistics of the age distribution again. Let' take this simple way but we should keep in mind that in contrast do the embarkation and the fare, our replacements can have a hugh impact on our model performance. We should try to impute nans with some more sophisticated methods to improve our model performance later on. 
# 

# In[ ]:


# Your task: Plot the ages distributions of both test and train set. 
# In addition use data.Age.describe() to obtain some statistical values. 
# Use the median for your replacements.
# Validate that there are no nans left by using data.Age.isnull().sum()


# ## Store your diving equipment
# 
# Now we are ready to take the next step of our diving course. If you like, store your own data and import it as external data source for the next diving course:

# In[ ]:


# train.to_csv("your_prepared_train.csv")
# test.to_csv("your_prepared_test.csv")

