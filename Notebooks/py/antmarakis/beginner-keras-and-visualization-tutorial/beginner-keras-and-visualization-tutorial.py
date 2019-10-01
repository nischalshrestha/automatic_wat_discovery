#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# In this tutorial we will use the popular Deep Learning library, `Keras`, and the visualization libraries `Matplotlib` and `Seaborn` to build a classifying simple model. The libraries `Numpy` and `Pandas` will help us along the way.
# 
# ## Data
# 
# First, we will load our dataset using `pandas`.

# In[ ]:


import pandas as pd

train_x = pd.read_csv("../input/train.csv", sep=',')
test_x = pd.read_csv("../input/test.csv", sep=',')


# Before we do anything else, let's see what our data looks like:

# In[ ]:


train_x.head(5)


# Looks good.
# 
# We will now start thinking of which of these features we will use in our model. Also, since we can see some missing values, we will need to tidy our data up too. First, let's see how many rows have missing values:

# In[ ]:


train_x.isnull().sum()


# Indeed there are some values missing from 'Age', 'Cabin' and a couple of rows in 'Embarked'. We will take care of them after we pick our features. For now, let's visualize some data!

# ## Visualization
# 
# To visualize our data, we will use `matplotlib` and `seaborn`.
# 
# Intuitively, it makes sense that the class of the passenger ('Pclass') would play a big role in the passenger's survival. The same probably holds true for the sex of the passenger. If I remember correctly, in the movie *Titanic* we see that the ladies were embarking the lifeboats first.
# 
# Let's see if these hypotheses are correct:

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x="Pclass", y="Survived", data=train_x);


# From the above data we can see that more than 60% of 1st class passengers survived, while less that 30% of 3rd class did. Let's keep this in mind, 'Pclass' so far seems like a good estimator.
# 
# Now to check the sex:

# In[ ]:


sns.barplot(x="Sex", y="Survived", data=train_x);


# *Titanic* stands true, it seems. Ladies indeed got in the lifeboats first, with about 70% of them surviving, while men only had a 1 in 5 chance of survival.
# 
# Let's now take a look at features like the number of siblings and spouses ('SibSp') and the type of the relationship ('Parch')'. These do not seem to be that relevant, but better to be safe than sorry.

# In[ ]:


sns.barplot(x="SibSp", y="Survived", data=train_x);


# In[ ]:


sns.barplot(x="Parch", y="Survived", data=train_x);


# Something does seem to be going on with 'SibSp', but the differences are small so to keep our feature set small we will skip it. For 'Parch', there doesn't seem to be a connection.
# 
# The 'Ticket' feature seems to be comprised of random alphanumericals, so we will skip it too.
# 
# Next we will take a look at 'Age'. Intuitively it makes sense to use this as a feature, since it is likely that they let elders reach the lifeboats alongside women and children.

# In[ ]:


sns.barplot(x="Age", y="Survived", data=train_x);


# Unfortunately, it is a bit hard to visualize ages the way we did the rest of the features, since we have a lot of different samples. Here we can see that ages on both ends of the spectrum seem to fare better, but we need to get a closer look. We will 'bin-ify' the ages, grouping them to bins according to their value. So, ages closer together will appear as one and it will be easier to visualize.
# 
# The function we will use will round the ages within a factor. To make our lives easier, we will use `numpy`.
# 
# NOTE: We pass as argument not the array itself, but its (deep) copy because we don't want the values of the dataset itself to be changed.

# In[ ]:


import numpy as np

def make_bins(d, col, factor=2):
    rounding = lambda x: np.around(x / factor)
    d[col] = d[col].apply(rounding)
    return d

t = make_bins(train_x.copy(True), 'Age', 7.5)
sns.barplot(x="Age", y="Survived", data=t);


# In the above case we round the ages up to a factor of `7.5`, and aside from the elders making it through, we don't get much information. Maybe we round too much? Let's try a smaller value:

# In[ ]:


t = make_bins(train_x.copy(True), 'Age', 5)
sns.barplot(x="Age", y="Survived", data=t);


# There doesn't seem to be much correlation to survival rate, except on the opposite ends of the spectrum. Maybe age didn't play much of a role after all.
# 
# Note that instead of creating the function `make_bins`, we could have used one of `seaborn`'s functions:

# In[ ]:


g = sns.FacetGrid(train_x, col='Survived')
g.map(plt.hist, 'Age', bins=20);


# For 'Fare', we will use the same, `make_bins`, method:

# In[ ]:


t = make_bins(train_x, 'Fare', 10)
sns.barplot(x="Fare", y="Survived", data=t);


# Even though small prices don't have much chance of surviving, the top prices seem to correlate to survivability at random. Still, we need to investigate this further in case there is some value to 'Fare'. If we listen to our instincts, it seems reasonable that a high fare means better passenger class. Maybe this is how 'Fare' ties in to survivability.

# In[ ]:


sns.barplot(x="Pclass", y="Fare", data=t);


# Indeed, the 1st class is more expensive, which makes sense. So, even though it seems that 'Fare' is not such a terrible indicator, after we snooped around we revealed that 'Fare' was 'Pclass' in disguise. Since we want to reduce dimensionality as much as possible, we will only use 'Pclass' and drop 'Fare'.
# 
# We have one more feature remaining, 'Embarked':

# In[ ]:


sns.barplot(x="Embarked", y="Survived", data=train_x);


# It does seem that those that embarked from 'C' have a higher chance of survival, so maybe we need to look further into this. Maybe the different embarkation points have different prices and that is what makes 'C' stand out.

# In[ ]:


sns.barplot(x="Embarked", y="Fare", data=train_x);


# As we can see, the port correlates to the fare, which is 'Pclass' in disguise. Thus, we can safely drop this feature as well to reduce computational strain.
# 
# ### Adding a feature of our own
# 
# Before we continues, let's consider another route. What if we were to create a feature of our own using data from above? In a lot of cases, this can be useful.
# 
# Remember that there is still a feature we haven't used: 'Name'. It doesn't really make much sense to use it, since a name cannot possibly be a factor in one's survival. But let's take a closer look, there is something interesting hiding between the lines...

# In[ ]:


train_x.head(3)


# Notice how the 'Name' column holds not only the actual name, but the *title* of the person too ('Mr.', 'Mrs.', etc.). That sounds kind of useful. Surely a lord of sorts would get in the lifeboats earlier, right? Let's see if that hypothesis is true.
# 
# First we will need to extract that information. Thankfully, `pandas` allows us to quickly parse a dataset using regular expressions.

# In[ ]:


train_x['Title'] = train_x.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


# We extracted all the titles from the 'Name' column, using a regular expression that matches parts of the field that start with a space and end with a period.
# 
# Below we can see how many times each title appears.

# In[ ]:


train_x['Title'].value_counts()


# The top four values appear a lot, while the rest don't appear that often. We need to clean this up, since that many different values may cause some trouble. We will merge the values that appear the least amount of times into a new title, called 'Rare'.
# 
# We will also merge 'Ms' and 'Mlle' to 'Miss', since we know that they all mean the same thing, and 'Mme' to 'Mrs' for the same reason.

# In[ ]:


train_x['Title'] = train_x['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don',                                             'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],                                             'Rare')
train_x['Title'] = train_x['Title'].replace('Mlle', 'Miss')
train_x['Title'] = train_x['Title'].replace('Ms', 'Miss')
train_x['Title'] = train_x['Title'].replace('Mme', 'Mrs')


# Let's now see if 'Title' is a good indicator:

# In[ ]:


sns.barplot(x="Title", y="Survived", data=train_x);


# It seems that we do get a bit more information. Whereas men generally didn't survive, those that had the title 'Master' did at almost 60%. There were also men that survived at a higher rate with a 'Rare' title. Thus, this seems like a good feature to add to our model.
# 
# Now we just need to convert these values to numerical, so that we can use them to fit our model. The little snippet of code that does this is the following:

# In[ ]:


_, train_x['Title'] = np.unique(train_x['Title'], return_inverse=True)


# ### Features to Pick
# 
# From the above results, we see that 'Pclass' and 'Sex' are good indicators, while 'Age' is so-so. In our model, we will pick the good indicators plus the feature we just created, 'Title'. If the results are not that good, we may consider adding 'Age' in the mix and seeing what comes out (note: I *did* use 'Age', and the results did not improve; in some models they even got worse).
# 
# Onward to cleaning our data!

# ## Cleaning Up
# 
# Now that we picked the features we want to use, we need to clean up our dataset.
# 
# First, we will drop the features we will not need:

# In[ ]:


train_x.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name',        'Cabin', 'PassengerId', 'Fare', 'Age'], inplace=True, axis=1)


# Next we will drop the null values:

# In[ ]:


train_x.dropna(inplace=True)


# Finally, we also need to convert 'Sex' to numerical values:

# In[ ]:


_, train_x['Sex'] = np.unique(train_x['Sex'], return_inverse=True)


# With that out of the way, we need to separate our dataset into features and target (the target in this case is 'Survived'). That way we can input the features into the model and compare the result to the corresponding target. Also, the target list should be a 1D array, which is the form Keras requires.

# In[ ]:


train_y = np.ravel(train_x.Survived) # Make 1D
train_x.drop(['Survived'], inplace=True, axis=1)


# And voila! Our dataset is ready. Let's build our Keras model.

# ## Keras Model
# 
# The model we will built, as well as most Keras models, will be `Sequential`.
# 
# We know that since the problem is a binary classification problem, we will need a sigmoid outer layer. Before that, we will add two fully-connected (`Dense`) layers that use `ReLU` as their activation function.

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Next we will compile our network. We have two target classes, so we will use the binary cross-entropy loss function. the `adam` optimizer is a good default gradient-descent option, and the metric will be the humble accuracy.

# In[ ]:


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# Only thing that remains is to fit our model with the given data:

# In[ ]:


model.fit(train_x, train_y, epochs=50, batch_size=1, verbose=1)


# We seem to have about 80% accuracy on our training data. Quite probably this will drop a bit in the test data, but it seems good enough for our little model.
# 
# Let's now prepare the testing dataset, similarly to how we prepared the training one.
# 
# *NOTE: If you want to save the weights, you can execute this line: `model.save_weights('weights.h5')`.*

# In[ ]:


to_test = test_x.copy(True)

# Add Title
to_test['Title'] = to_test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
to_test['Title'] = to_test['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don',                                                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],                                                'Rare')
to_test['Title'] = to_test['Title'].replace('Mlle', 'Miss')
to_test['Title'] = to_test['Title'].replace('Ms', 'Miss')
to_test['Title'] = to_test['Title'].replace('Mme', 'Mrs')

_, to_test['Title'] = np.unique(to_test['Title'], return_inverse=True)

# Clean Data
to_test = to_test.drop(['SibSp', 'Parch', 'Ticket', 'Embarked', 'Name', 'Cabin',                        'PassengerId', 'Fare', 'Age'], axis=1)
_, to_test['Sex'] = np.unique(test_x['Sex'], return_inverse=True)


# Note that since we are going to need the id of the passengers for our submission, we need to copy `test_x` to a temporary dataset. If we made the changes to `test_x` directly, we would lose `PassengerId`, since we drop it.
# 
# Finally it is time to test our model!

# In[ ]:


predictions = model.predict_classes(to_test).flatten()
predictions[:5]


# To make our lives easier, we flattened the array. We also show the first five values. Only thing we need to do is write this to a `.csv` and submit it:

# In[ ]:


submission = pd.DataFrame({
    "PassengerId": test_x["PassengerId"],
    "Survived": predictions
})
submission.to_csv('submission.csv', index=False)


# ## Conclusion
# 
# And that is all.
# 
# We visualized our data, found the best features to use, created a feature of our own, build our model and submitted our results. Even though the model is quite simple, it gave about 75% accuracy on the test. Not bad for the little buddy.
# 
# You might be thinking that we could increase the results by adding more layers, neurons and epochs. Maybe, but this was my initial approach and I can tell you it didn't work that well. That is because the dataset is quite small and my bigger model overfit. So, I went back to a simpler model and I found that the results were better.
# 
# So, bigger is not always better when it comes to Deep Learning models.
