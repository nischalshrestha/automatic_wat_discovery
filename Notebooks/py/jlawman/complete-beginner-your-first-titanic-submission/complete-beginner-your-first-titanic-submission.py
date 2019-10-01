#!/usr/bin/env python
# coding: utf-8

# This tutorial walks you through submitting a ".csv" file of predictions to Kaggle for the first time.<br><br>
# 
# ### Scoring and challenges:<br>
# If you simply run the code below, your score will be fairly poor. I have intentionally left lots of room for improvement regarding the model used (currently a simple decision tree classifier). <br><br> The idea of this tutorial is to get you started and have you make the decisions of how to improve your score. At the bottom of the tutorial are challenges which, if you follow them, will significantly improve your score.
# 
# 
# 
# ### Steps to complete this tutorial on your own computer:
# The kernel below can be run in the browser. But if you would like to run the code locally on your own computer, you can follow the steps below.
# 
# 1. Create a Kaggle account (https://www.kaggle.com/).
# 2. Download Titanic dataset (https://www.kaggle.com/c/titanic/data).<br>
#     a. Download 'train.csv' and 'test.csv'.<br>
#     b. Place both files in a folder named 'input'.<br>
#     c. Place that folder in the same directory as your notebook.
# 3. Install [Jupyter Notebooks](https://jupyter.org/) (follow my [installation tutorial](http://joshlawman.com/getting-set-up-in-jupyter-notebooks-using-anaconda-to-install-the-jupyter-pandas-sklearn-etc/) if you are confused)
# 4. Download this kernel as a [notebook](https://github.com/jlawman/Meetup/blob/master/11.7%20Meetup%20-%20Decision%20Trees/Submit%20your%20first%20Kaggle%20prediction%20-%20Titanic%20Dataset.ipynb) with empty cells from my GitHub. If you are new to GitHub go [the repository folder](https://github.com/jlawman/Meetup), click "Clone or Download", then unzip the file and pull out the notebook you want.
# 5. Run every cell in the notebook (except the optional visualization cells).
# 6. Submit CSV containing the predictions.
# 7. Try to improve the prediction by using the challenge prompts which are suitable to your level.

# ## 1. Process the data
# 
# ### Load data

# In[ ]:


#Load data
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Look at the first 3 rows of our training data
train.head(3)


# Our data has the following columns:
# - PassengerId - Each passenger's id
# - Survived - Whether the passenger survived or not (1 - yes, 0 - no)
# - Pclass - The passenger class: (1st class - 1, 2nd class - 2, third class - 3)
# - Sex - Each passenger's sex
# - Age - Each passenger's age

# ### Prepare the data to be read by our algorithm

# In[ ]:


#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(3)


# Let's look at the first 3 corresponding target variables. This is the measure of whether the passenger survived or not (i.e. the first passenger (22 year-old male) did not survive, but the second passenger (38 year-old female did survive).
# <br><br>
# Our classifier will use this to know what the output should be for each of the training instances.

# In[ ]:


#Display first 3 target variables
train[target].head(3).values


# # 2. Create and fit the decision tree
# 
# This tree is definitely going to overfit our data. When you get to the challenge stage, you can return here and tune hyperparameters in this cell. For example, you could reduce the maximum depth of the tree to 3 by setting max_depth=3 with the following command:
# >clf = DecisionTreeClassifier(max_depth=3)
# 
# To change multiple hyperparameters, seperate out the parameters with a comma. For example, to change the learning rate and minimum samples per leaf and the maximum depth fill in the parentheses with the following:
# >clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=2)
# 
# The other parameters are listed below.
# You can also access the list of parameters by reading the [documentation](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier) for decision tree classifiers. Another way to access the parameters is to place your cursor in between the parentheses and then press shift-tab.
# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
clf = DecisionTreeClassifier()  

#Fit our classifier using the training features and the training target values
clf.fit(train[features],train[target]) 


# ### Visualize default tree (optional)
# This is not a necessary step, but it shows you how complex the tree is when you don't restrict it. To complete this visualization section you must be going through the code on your computer.

# In[ ]:


#Create decision tree ".dot" file

#Remove each '#' below to uncomment the two lines and export the file.
#from sklearn.tree import export_graphviz
#export_graphviz(clf,out_file='titanic_tree.dot',feature_names=features,rounded=True,filled=True,class_names=['Survived','Did not Survive'])


# Note, if you want to generate a new tree png, you need to open terminal (or command prompt) after running the cell above. Navigate to the directory where you have this notebook and the type the following command.
# >dot -Tpng titanic_tree.dot -o titanic_tree.png<br><br>

# In[ ]:


#Display decision tree

#Blue on a node or leaf means the tree thinks the person did not survive
#Orange on a node or leaf means that tree thinks that the person did survive

#In Chrome, to zoom in press control +. To zoom out, press control -. If you are on a Mac, use Command.

#Remove each '#' below to run the two lines below.
#from IPython.core.display import Image, display
#display(Image('titanic_tree.png', width=1900, unconfined=True))


# # 3. Make Predictions

# In[ ]:


#Make predictions using the features from the test data set
predictions = clf.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions


# # 4. Create csv to upload to Kaggle

# In[ ]:


#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()


# In[ ]:


#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)


# # 5. Submit file to Kaggle
# 
# Go to the [submission section](https://www.kaggle.com/c/titanic/submit) of the Titanic competition. Drag your file from the directory which contains your code and make your submission.<br><br> Congratulations - you're on the leaderboard!****

# # Challenges
# 
# The default decision tree gives a score of .70813 placing you at rank 8,070 out of 8,767. Can you improve it?
# 
# ### Level 1: First time on Kaggle
# Level 1a: Can you try to give the tree a max depth to improve your score?
# 
# Level 1b:  Can you import a different tree models such as the Random Forest Classifier to see how it affects your score? Use the following code line to create it. Compare this model to a decision tree with depth 3.
# > from sklearn.ensemble import RandomForestClassifier<br>
# > clf = RandomForestClassifier() ****
# 
# 
# ### Level 2: Submitted to Kaggle before
# Level 2a: Can you include other features that were dropped to improve your score? Don't forget to deal with any missing data.
# <br><br>
# Level 2b: Can you visualize your data using matplotlib or seaborn to glean other insights of how to improve your predictions?
# 
# ### Level 3: Some familiarity with scikit-learn
# Level 3a: Can you use GridSearchCV from sklearn.model_selection on the Random Forest Classifier to tune the hyperparameters and improve your score?
# <br><br>
# Level 3b: Can you train a list of models and then evaluate each one using sklearn.metrics train_test_split function to see which give you the best score?
# <br><br>
# Level 3c: Can you take the list from challenge 3b and then have the best models in the list vote on how each prediction should be made? 
