#!/usr/bin/env python
# coding: utf-8

# **TITANIC: A Data Starter Kit**
# 
# Data Science generally flows between a few different steps. Using the Titanic survival data, we'll be taking a look at a few components, start to finish. Wherever possible, I'll give links to places to look further for more info.  NOTE: In code, messages after a # symbol are comments. They don't execute, but provide helpful tips and tricks!

# **Step 0: Set up**
# 
# This notebook is *(Well, obviously)* hosted on Kaggle. For the most part, if you find a dataset or competition on Kaggle, you can get your entire set up done online! Simply go to the competition/dataset you want to use, then go to the Kernals tab. Another option for you to use, If you find someone else's Kernel that you really like and want to use as a base point, then you'll need to "Fork" it. Think like you're making a fork in the road in terms of the code base. Basically, go to that code's page, and hit the "Fork" button. That'll create a copy of that code for you to edit. 
# 
# From there, hit "New Kernal" and select notebook. Then, get coding! Otherwise, if you want to use something that *isn't* on Kaggle, you're going to have to use a local set up. I personally use Jupyter Notebooks for everything local, and you can find more on that here: [https://www.youtube.com/watch?v=HW29067qVWk](http://). 
# 
# This will work whichever route you choose. Personally, I recommend starting with the Kaggle  verson until you need to break away from it, but that's just me!

# **Step 1: Importing our Libraries**
# 
# We don't have time to reinvent the wheel, we've got data to use. So, we import Libraries. Specific libraries can be imported later as needed, but to start, these are the really important ones. 

# In[ ]:


import numpy as np # Numpy handles all things Linear Algebra
import pandas as pd  # Pandas handles DataFrames, the things that we use to structure our data

#Then, to help visualize our data, we use matplotlib and seaborn. 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# **Step 2: Get in the Data**
# 
# Next, now that our enviornment is set up, we need to read in the data. Pandas, imported above, makes this process really simple. Just use pd.read_csv(DIRECTORY)! In this case, we can use the "input" folder in the working directory.  

# In[ ]:


data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')


# **Step 3: Getting our Feet Wet**
# 
# We have a functioning enviornment. We have data. Now, before we can procede, we need to know what we're working with! There are a few commands that work on DataFrames, which generally help give you a feel of the structure of the data:
# * .head(n)  will print the first n rows of the data
# * .tail(n) will print the last n rows of the data
# * .sample(n) will print n random rows of the data
# * .describe() will give a 5 number summary of the data

# In[ ]:


data_train.describe()


# In[ ]:


data_train.sample(5)


# **Step 4: Deeper exploration**
# 
# Once we have an idea of what the data is, what shape it takes, what the columns are, etc. we can dive into a deeper look. With this deeper look, we can hope to reveal a bunch of intuitions - maybe there are some missing values, and maybe they mean something? Maybe some variables are more important than others? Maybe we can come up with another variable that might help? Whatever the case may be, graphs help shed a lot of light on the subject.
# 
# As for how to go about that, there are actually a lot of ways. The seaborn library, imported above, is powerful. However, pandas has a very, very simple graph library that can get us up and running. To use it, simply use the command data_frame_name["ColumnName"].plot(kind= "hist")  *Note*: A full list of graph types is available here: [https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.plot.html](http://)
# 
# You can also set things like the title, axes labels, whatever you want! Again, a full list of the parameters available is available in the documentation above. [If you would like to learn more about visualization, I recommend you work on your Pandas DataFrame skills, then seaborn. This video talks about some DataFrame commands that pick out things from your DataFrame: [https://www.youtube.com/watch?v=xvpNA7bC8cs]. Specifically, he talks about iloc, ix, and loc, and how you use them. They're annoyingly similar, and it'll take some practice and reference to figure out when to use which but... Once you get there, it gives you a LOT of flexibility. If you have any other pandas questions, he's the guy to answe rthem. Once you have a finner tuned control of the data sets, I recommend giving this a go for seaborn: Note, you may not be able to actually *download* the data without setting up a whole desktop enviornment, but the lessons generalize well to what we're doing here: (http://). https://elitedatascience.com/python-seaborn-tutorial)

# In[ ]:


data_train["Age"].plot(kind="hist", title="Ages of Titanic Passengers")


# *Technique:* Sometimes, we want to compare subcategories of things. For example, let's say we wanted to compare the ages of those who survived vs. those who didn't survive. In this case, we can use the groupby command. The groupby command clusters a particular column into groups, each of which is then processed on it's own. For example, let's take the data, and split it in two: those who survived, and those who didn't. Let's take the ages of everyone in each of those two groups, and plot it on a histogram.

# In[ ]:


data_train.groupby("Survived")["Age"].plot(kind="hist", title="Ages of Titanic Survivers vs. Casulities", legend=True)


# In[ ]:


data_train.groupby("Survived").describe()


# **Step 5: Preparing the Data for the Model**
# 
# So, we have the data. We understand the data. Now, we need to get the data ready for the model. This can take a bit of creativity --- More often than not, these models all require numerical data. How do we handle categorical data? Plus, how do we handle missing values? Outliers? Do we need to normalize anything? Are there useless variables, or can we add anything? 
# 
# This all takes some careful thought, and ultimately makes or breaks a good model. Heck, that last question opens the door to the entire field of feature engineering. (Really recommend reading up on that more here: [https://elitedatascience.com/feature-engineering)] - Or, if you want the really long version: [http://adataanalyst.com/machine-learning/comprehensive-guide-feature-engineering/](http://))
# 
# In this case, we're just going to use the pandas "Drop" command to get rid of a few useless columns, and use a cool technique to represent categorical data.
# 
# *Technique:* When we want to convert categorical data into numerical data, we can use a process called one-hot encoding. Essentially, we take a column like "Embarked", and we split it into three or so columns. If the person embarked from location A, they get a "1" in the "Embarked_A" column. If they embarked from location B, they get a "1" in the "Embarked_B" column, etc. We can use the pandas "get_dummies(x)" command for this.
# 
# Lastly, we're going to want to handle missing values. Some techniques involve regressing an accurate value, others involve filling in the mean. In this case, we're just going to replace the missing values with 0 for the sake of simplicity. This can be done with the pandas fillna function.

# In[ ]:


#This is a function. It takes in a DataFrame X, cleans it up, and returns the result. We can reuse this wherever we may need, whenever we may need. 

def cleanInput(x):
    #In this case, passenger ID, the ticket number, and the cabin number seem to be fairly bad indicators. Let's just drop them all together. 
    x = x.drop("PassengerId", axis=1) #axis=1 tells the command to drop a column, not a row.
    x = x.drop("Ticket", axis=1) 
    x = x.drop("Cabin", axis=1) 
    x = x.drop("Name", axis=1)
    
    #One-hot encode the given variable
    x = pd.get_dummies(x)
    
    #Lastly, replace all 0's. 
    x = x.fillna(0)
    return x


# In[ ]:


#Now that we declared our function above, we can pass in the training data to it, get the result, and finally that result.
#Best of all, because that code is in a function, we can call it over and over again.
cleanInput(data_train).head()


# **Step 6: Getting our Training and Testing Data**
# 
# We have our data. It's clean, and we have a general understanding of it. Now, we just need to get some predictions! This is when we break out the ScikitLearn library. They make it really simple to get thorough this process painlessly. 
# 
# First, in order to tell if we're doing good, we're going to take our training data, and split it even further. We want to have a set of data to compare to after we build this model, and the train_test_split command will help us get there. Basically, it randomly selects 'num_test'% of the data to be test data. Then, it removes those rows, and adds them to the X_test and y_test, where X_test is the inputs we want to use to test the model, and y_test is the answer key. We can use X_train and y_train to actually build the model. *Note: the reason we do this split, is simply because we don't want the model 'memorizing' the input. We want to be sure that this model generalizes to other sitations, not specifically built into the training data.*

# In[ ]:


#Seperate our x variables from our y variables
from sklearn.model_selection import train_test_split

x_all = cleanInput(data_train.drop(['Survived'], axis=1))
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=num_test, random_state=23)


# In[ ]:


X_train.sample(4)


# In[ ]:


y_train.sample(4)


# **Step 7: Machine Learning**
# 
# Now, we can see where the SkLearn library really shines. It's a ton of different  models, all hooked up to the same framework.
# 
# Some of the possibilities are....:
# * Random Forests [https://en.wikipedia.org/wiki/Random_forest](http://) ----- [http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](http://)
# * K-Nearest Neighbors [https://medium.com/@adi.bronshtein/a-quick-introduction-to-k-nearest-neighbors-algorithm-62214cea29c7](http://) -----  [http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](http://)
# * Support Vector Machines [https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72](http://)  ----- [http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC](http://)
# 
# But, a good idea for the possibilities can be found right on the SKLearn website: [http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html](http://)
# 
# XKCD:[ https://xkcd.com/1838/](http://)

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

model = RandomForestClassifier()
model = LinearSVC() #SVC = Support Vector Classifier
#You could use the above to keep subbing data in and out. Or, you could use the below to 
model = KNeighborsClassifier(n_neighbors = 4)

# Fit the best algorithm to the data. 
model.fit(X_train, y_train)


# Next, we compare our model against the test data.... (We're going for a '1' here)

# In[ ]:


from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))


# **Step 8: Make our Final Predictions!**
# 
# At long last. Now, we just have to take the data, make our predictions on the given Test data, and output it. 

# In[ ]:


#Predict on the final test variables
ids = data_test["PassengerId"]
predictions = model.predict(cleanInput(data_test))
submission = pd.DataFrame({"PassengerId": ids,"Survived": predictions})


# In[ ]:


#Ok, everything seems to be in order... Note: The competition runners will specify the format. In this case, they provided a sample submission
#here (it's the gendered_submission. For the sake of demonstration, they just said every female survived, every male didn't.
#https://www.kaggle.com/c/titanic/data)
print(submission.sample(10))


# **Step 9: Get the output data**
# 
# Now, ,finally, we can collect our results in a Comma-Seperated-Values file. If you'd like to actually submit this file, you need to hit Commit & Run in the top corner. Then, go back, and make this notebook public on the Titanic page itself. That should open up an "output" tab. Took me some time to figure that step out too, and this is the post that got me there: [https://www.kaggle.com/szamil/where-is-my-output-file](http://)
# 
# The best part is that you can submit right from the output menu, which is neato. No downloading required. 

# In[ ]:


submission.to_csv('titanic.csv', index=False)


# **What now?**
# 
# Now, you can tweek the model and try to bump up your score on that leaderboard! Maybe instead of replacing values with 0, you replace it with the mean of that column. Maybe, instead of using 5 neighbors for that KNN back there, you use 3. Maybe, you come up with some other killer variable, like the origin of that person's last name. The world is your oyster!
# 
# I talked about a few sources throughout that were helpful to me. Here are a few other things to check out: First of all, I mentioned him before, but THIS GUY IS REALLY GOOD TO WATCH (on 1.25 speed): [https://www.youtube.com/watch?v=yzIMircGU5I&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y](http://). Second, it can't hurt to learn more vanilla python in case you ever need to break it out: [https://www.youtube.com/watch?v=N4mEzFDjqtA](http://) If you're an experienced programmer, feel free to cruise through that. Otherwise, take some time to stop and play around with things! NOTE: You can probably stop at about the 32 minute mark for the sake of this. You won't be needing classes or objects, we have enough libraries for that!
# 
# Other than that, really it's just look into what interests you. If you're into visualization, try to mess around with Matplotlib and seaborn. If you're into the models, try to bounce around the Wikipedia page, get the lay of the land. If you're into some of the more mathematically involved stuff, consider giving Principle Component Analysis a go. Basically, it re-expresses data along it's most significant eigenvectors, which can reduce a 9142 dimmensional matrix to a 10 dimmensional one or something like that. Again, SKlearn makes this super simple, but knowing the math behind it has it's benefits.  [https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60](http://).  
# 
# Pick a road to start on, maybe find another dataset somewhere, and go with it. I promise, you'll find plenty of paths on your own. GG.

# In[ ]:




