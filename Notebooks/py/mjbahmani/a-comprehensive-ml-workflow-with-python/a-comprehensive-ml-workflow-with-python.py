#!/usr/bin/env python
# coding: utf-8

# ### <div style="text-align: center">A Comprehensive Machine Learning Workflow with Python </div>
# 
# <div style="text-align: center">There are plenty of <b>courses and tutorials</b> that can help you learn machine learning from scratch but here in <b>Kaggle</b>, I want to solve <font color="red"><b>Titanic competition</b></font>  a popular machine learning Dataset as a comprehensive workflow with python packages. 
# After reading, you can use this workflow to solve other real problems and use it as a template to deal with <b>machine learning</b> problems.</div>
# <div style="text-align:center">last update: <b>11/13/2018</b></div>
# 
# 
# 
# >###### you may  be interested have a look at it: [**10-steps-to-become-a-data-scientist**](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# ---------------------------------------------------------------------
# you can Fork and Run this kernel on <font color="red">Github</font>:
# 
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated**
#  
#  -----------

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
#      1. [Courses](#1)
#      1. [Kaggle kernels](#1)
#      1. [Ebooks](#1)
#      1. [CheatSheet](#1)
# 1. [Machine learning](#2)
#     1. [Machine learning workflow](#2)
#     1. [Real world Application Vs Competitions](#2)
# 1. [Problem Definition](#3)
#     1. [Problem feature](#4)
#         1. [Why am I  using Titanic dataset](#4)
#     1. [Aim](#5)
#     1. [Variables](#6)
#         1. [Types of Features](#6)
#             1. [Categorical](#6)
#             1. [Ordinal](#6)
#             1. [Continous](#6)
# 1. [ Inputs & Outputs](#7)
#     1. [Inputs ](#8)
#     1. [Outputs](#9)
# 1. [Installation](#10)
#     1. [ jupyter notebook](#11)
#         1. [What browsers are supported?](#11)
#     1. [ kaggle kernel](#12)
#     
#     1. [Colab notebook](#13)
#     1. [install python & packages](#14)
#     1. [Loading Packages](#15)
# 1. [Exploratory data analysis](#16)
#     1. [Data Collection](#17)
#     1. [Visualization](#18)
#         1. [Scatter plot](#19)
#         1. [Box](#20)
#         1. [Histogram](#21)
#         1. [Multivariate Plots](#22)
#         1. [Violinplots](#23)
#         1. [Pair plot](#24)
#         1. [Kde plot](#25)
#         1. [Joint plot](#26)
#         1. [Andrews curves](#27)
#         1. [Heatmap](#28)
#         1. [Radviz](#29)
#         1. [Data Preprocessing](#30)
#         1. [Data Cleaning](#31)
# 1. [Model Deployment](#32)
#     1. [ KNN](#33)
#     1. [Radius Neighbors Classifier](#34)
#     1. [Logistic Regression](#35)
#     1. [Passive Aggressive Classifier](#36)
#     1. [Naive Bayes](#37)
#     1. [MultinomialNB](#38)
#     1. [BernoulliNB](#39)
#     1. [SVM](#40)
#     1. [Nu-Support Vector Classification](#41)
#     1. [Linear Support Vector Classification](#42)
#     1. [Decision Tree](#43)
#     1. [ExtraTreeClassifier](#44)
#     1. [Neural network](#45)
#         1. [What is a Perceptron?](#45)
#     1. [RandomForest](#46)
#     1. [Bagging classifier ](#47)
#     1. [AdaBoost classifier](#48)
#     1. [Gradient Boosting Classifier](#49)
#     1. [Linear Discriminant Analysis](#50)
#     1. [Quadratic Discriminant Analysis](#51)
#     1. [Kmeans](#52)
#     1. [Backpropagation](#53)
# 1. [Conclusion](#54)
# 1. [References](#55)

#  <a id="1"></a> <br>
# ## 1- Introduction
# This is a **comprehensive ML techniques with python** , that I have spent for more than two months to complete it.
# 
# it is clear that everyone in this community is familiar with Titanic dataset but if you need to review your information about the dataset please visit this [link](https://www.kaggle.com/c/titanic/data).
# 
# I have tried to help **beginners**  in Kaggle how to face machine learning problems. and I think it is a great opportunity for who want to learn machine learning workflow with python completely.
# I have covered most of the methods that are implemented for **Titanic** until **2018**, you can start to learn and review your knowledge about ML with a perfect dataset and try to learn and memorize the workflow for your journey in Data science world.
#  <a id="1"></a> <br>
# ## 1-1 Courses
# There are a lot of online courses that can help you develop your knowledge, here I have just  listed some of them:
# 
# 1. [Machine Learning Certification by Stanford University (Coursera)](https://www.coursera.org/learn/machine-learning/)
# 
# 2. [Machine Learning A-Z™: Hands-On Python & R In Data Science (Udemy)](https://www.udemy.com/machinelearning/)
# 
# 3. [Deep Learning Certification by Andrew Ng from deeplearning.ai (Coursera)](https://www.coursera.org/specializations/deep-learning)
# 
# 4. [Python for Data Science and Machine Learning Bootcamp (Udemy)](Python for Data Science and Machine Learning Bootcamp (Udemy))
# 
# 5. [Mathematics for Machine Learning by Imperial College London](https://www.coursera.org/specializations/mathematics-machine-learning)
# 
# 6. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 
# 7. [Complete Guide to TensorFlow for Deep Learning Tutorial with Python](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/)
# 
# 8. [Data Science and Machine Learning Tutorial with Python – Hands On](https://www.udemy.com/data-science-and-machine-learning-with-python-hands-on/)
# 
# 9. [Machine Learning Certification by University of Washington](https://www.coursera.org/specializations/machine-learning)
# 
# 10. [Data Science and Machine Learning Bootcamp with R](https://www.udemy.com/data-science-and-machine-learning-bootcamp-with-r/)
# 11. [Creative Applications of Deep Learning with TensorFlow](https://www.class-central.com/course/kadenze-creative-applications-of-deep-learning-with-tensorflow-6679)
# 12. [Neural Networks for Machine Learning](https://www.class-central.com/mooc/398/coursera-neural-networks-for-machine-learning)
# 13. [Practical Deep Learning For Coders, Part 1](https://www.class-central.com/mooc/7887/practical-deep-learning-for-coders-part-1)
# 14. [Machine Learning](https://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/index.html)
#  <a id="1"></a> <br>
# ## 1-2 Kaggle kernels
# I want to thanks **Kaggle team**  and  all of the **kernel's authors**  who develop this huge resources for Data scientists. I have learned from The work of others and I have just listed some more important kernels that inspired my work and I've used them in this kernel:
# 
# 1. [https://www.kaggle.com/ash316/eda-to-prediction-dietanic](https://www.kaggle.com/ash316/eda-to-prediction-dietanic)
# 
# 2. [https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
# 
# 3. [https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# 4. [https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# 
# 5. [https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
#  <a id="1"></a> <br>
# ## 1-3 Ebooks
# So you love reading , here is **10 free machine learning books**
# 1. [Probability and Statistics for Programmers](http://www.greenteapress.com/thinkstats/)
# 2. [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)
# 2. [An Introduction to Statistical Learning](http://www-bcf.usc.edu/~gareth/ISL/)
# 2. [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/index.html)
# 2. [A Programmer’s Guide to Data Mining](http://guidetodatamining.com/)
# 2. [Mining of Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/book.pdf)
# 2. [A Brief Introduction to Neural Networks](http://www.dkriesel.com/_media/science/neuronalenetze-en-zeta2-2col-dkrieselcom.pdf)
# 2. [Deep Learning](http://www.deeplearningbook.org/)
# 2. [Natural Language Processing with Python](https://www.researchgate.net/publication/220691633_Natural_Language_Processing_with_Python)
# 2. [Machine Learning Yearning](http://www.mlyearning.org/)
#  <a id="1"></a> <br>
# ## 1-4 Cheat Sheets
# Data Science is an ever-growing field, there are numerous tools & techniques to remember. It is not possible for anyone to remember all the functions, operations and formulas of each concept. That’s why we have cheat sheets. But there are a plethora of cheat sheets available out there, choosing the right cheat sheet is a tough task. So, I decided to write this article.
# 
# Here I have selected the cheat sheets on the following criteria: comprehensiveness, clarity, and content [26]:
# 1. [Quick Guide to learn Python for Data Science ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Data-Science-in-Python.pdf)
# 1. [Python for Data Science Cheat sheet ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/beginners_python_cheat_sheet.pdf)
# 1. [Python For Data Science Cheat Sheet NumPy](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Numpy_Python_Cheat_Sheet.pdf)
# 1. [Exploratory Data Analysis in Python]()
# 1. [Data Exploration using Pandas in Python](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Data-Exploration-in-Python.pdf)
# 1. [Data Visualisation in Python](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/data-visualisation-infographics1.jpg)
# 1. [Python For Data Science Cheat Sheet Bokeh](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Python_Bokeh_Cheat_Sheet.pdf)
# 1. [Cheat Sheet: Scikit Learn ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/Scikit-Learn-Infographic.pdf)
# 1. [MLalgorithms CheatSheet](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/MLalgorithms-.pdf)
# 1. [Probability Basics  Cheat Sheet ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/blob/master/cheatsheets/probability_cheatsheet.pdf)
# 
# I am open to getting your feedback for improving this **kernel**
# ###### [Go to top](#top)
# 

# <a id="2"></a> <br>
# ## 2- Machine Learning
# Machine Learning is a field of study that gives computers the ability to learn without being explicitly programmed.
# 
# **Arthur	Samuel, 1959**

#  <a id="2"></a> <br>
# ## 2-1 Machine Learning Workflow
# 
# If you have already read some [machine learning books](https://towardsdatascience.com/list-of-free-must-read-machine-learning-books-89576749d2ff). You have noticed that there are different ways to stream data into machine learning.
# 
# most of these books share the following steps (checklist):
# *   Define the Problem(Look at the big picture)
# *   Specify Inputs & Outputs
# *   Data Collection
# *   Exploratory data analysis
# *   Data Preprocessing
# *   Model Design, Training, and Offline Evaluation
# *   Model Deployment, Online Evaluation, and Monitoring
# *   Model Maintenance, Diagnosis, and Retraining
# 
# **You can see my workflow in the below image** :
#  <img src="http://s9.picofile.com/file/8338227634/workflow.png" />
# 
# **you should	feel free	to	adapt 	this	checklist 	to	your needs**

#  <a id="2"></a> <br>
# ## 2-1 Real world Application Vs Competitions
# Just a simple comparison between real-world apps with competitions:
# <img src="http://s9.picofile.com/file/8339956300/reallife.png" height="600" width="500" />
# ###### [Go to top](#top)

# <a id="3"></a> <br>
# ## 3- Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# 
# Problem Definition has four steps that have illustrated in the picture below:
# <img src="http://s8.picofile.com/file/8338227734/ProblemDefination.png">
# <a id="4"></a> <br>
# ### 3-1 Problem Feature
# The sinking of the Titanic is one of the most infamous shipwrecks in history. **On April 15, 1912**, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing **1502 out of 2224** passengers and crew. That's why the name DieTanic. This is a very unforgetable disaster that no one in the world can forget.
# 
# It took about $7.5 million to build the Titanic and it sunk under the ocean due to collision. The Titanic Dataset is a very good dataset for begineers to start a journey in data science and participate in competitions in Kaggle.
# 
# we will use the classic titanic data set. This dataset contains information about **11 different variables**:
# <img src="http://s9.picofile.com/file/8340453092/Titanic_feature.png" height="500" width="500">
# 
# * Survival
# * Pclass
# * Name
# * Sex
# * Age
# * SibSp
# * Parch
# * Ticket
# * Fare
# * Cabin
# * Embarked
# 
# <a id="4"></a> <br>
# ### 3-3-1 Why am I  using Titanic dataset
# 
# 1- This is a good project because it is so well understood.
# 
# 2- Attributes are numeric and categorical so you have to figure out how to load and handle data.
# 
# 3- It is a ML problem, allowing you to practice with perhaps an easier type of supervised learning algorithm.
# 
# 4- we can define problem as clustering(unsupervised algorithm) project too.
# 
# 5- because we love   **Kaggle** :-) .
# 
# <a id="5"></a> <br>
# ### 3-2 Aim
# It is your job to predict if a passenger survived the sinking of the Titanic or not.  For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
# 
# <a id="6"></a> <br>
# ### 3-3 Variables
# 
# 1.  **Age** ==>> Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
# 
# 2. **Sibsp** ==>> The dataset defines family relations in this way...
# 
#     a. Sibling = brother, sister, stepbrother, stepsister
# 
#     b. Spouse = husband, wife (mistresses and fiancés were ignored)
# 
# 3. **Parch** ==>> The dataset defines family relations in this way...
# 
#     a. Parent = mother, father
# 
#     b. Child = daughter, son, stepdaughter, stepson
# 
#     c. Some children travelled only with a nanny, therefore parch=0 for them.
# 
# 4. **Pclass** ==>> A proxy for socio-economic status (SES)
#     * 1st = Upper
#     * 2nd = Middle
#     * 3rd = Lower
# 5. **Embarked** ==>> nominal datatype 
# 6. **Name** ==>> nominal datatype . It could be used in feature engineering to derive the gender from title
# 7. **Sex** ==>>  nominal datatype 
# 8. **Ticket** ==>> that have no impact on the outcome variable. Thus, they will be excluded from analysis
# 9. **Cabin** ==>>  is a nominal datatype that can be used in feature engineering
# 11. **Fare** ==>>  Indicating the fare
# 12. **PassengerID ** ==>> have no impact on the outcome variable. Thus, it will be excluded from analysis
# 11. **Survival** is ==>> **[dependent variable](http://www.dailysmarty.com/posts/difference-between-independent-and-dependent-variables-in-machine-learning)** , 0 or 1
# 
# <a id="4"></a> <br>
# ### 3-3-1  Types of Features
# <a id="4"></a> <br>
# ### 3-3-1-1 Categorical
# 
# A categorical variable is one that has two or more categories and each value in that feature can be categorised by them. for example, gender is a categorical variable having two categories (male and female). Now we cannot sort or give any ordering to such variables. They are also known as Nominal Variables.
# 
# **Categorical Features in the dataset: Sex,Embarked.**
# <a id="4"></a> <br>
# ### 3-3-1-2 Ordinal
# An ordinal variable is similar to categorical values, but the difference between them is that we can have relative ordering or sorting between the values. For eg: If we have a feature like Height with values Tall, Medium, Short, then Height is a ordinal variable. Here we can have a relative sort in the variable.
# 
# **Ordinal Features in the dataset: PClass**
# 
# ### 3-3-1-3 Continous:
# A feature is said to be continous if it can take values between any two points or between the minimum or maximum values in the features column.
# 
# **Continous Features in the dataset: Age**
# 
# **<< Note >>**
# > You must answer the following question:
# How does your company expact to use and benfit from your model.
# ###### [Go to top](#top)

# <a id="7"></a> <br>
# ## 4- Inputs & Outputs
# <a id="8"></a> <br>
# ### 4-1 Inputs
# **Titanic** is a very popular **Machine Learning**   problem then I decided to apply it on  plenty of machine learning methods.
# The titanic data set is a **multivariate data set** .
# <a id="9"></a> <br>
# ### 4-2 Outputs
# Your score is the percentage of passengers you correctly predict. This is known simply as "accuracy”.
# 
# 
# The Outputs should have exactly **2 columns**:
# 
# PassengerId (sorted in any order)
# Survived (contains your binary predictions: 1 for survived, 0 for deceased)
# 

# <a id="10"></a> <br>
# ## 5-Installation
# #### Windows:
# * Anaconda (from https://www.continuum.io) is a free Python distribution for SciPy stack. It is also available for Linux and Mac.
# * Canopy (https://www.enthought.com/products/canopy/) is available as free as well as commercial distribution with full SciPy stack for Windows, Linux and Mac.
# * Python (x,y) is a free Python distribution with SciPy stack and Spyder IDE for Windows OS. (Downloadable from http://python-xy.github.io/)
# 
# #### Linux:
# Package managers of respective Linux distributions are used to install one or more packages in SciPy stack.
# 
# For Ubuntu Users:
# sudo apt-get install python-numpy python-scipy python-matplotlibipythonipythonnotebook
# python-pandas python-sympy python-nose

# <a id="11"></a> <br>
# ## 5-1 Jupyter notebook
# I strongly recommend installing **Python** and **Jupyter** using the **[Anaconda Distribution](https://www.anaconda.com/download/)**, which includes Python, the Jupyter Notebook, and other commonly used packages for scientific computing and data science.
# 
# 1. First, download Anaconda. We recommend downloading Anaconda’s latest Python 3 version.
# 
# 2. Second, install the version of Anaconda which you downloaded, following the instructions on the download page.
# 
# 3. Congratulations, you have installed Jupyter Notebook! To run the notebook, run the following command at the Terminal (Mac/Linux) or Command Prompt (Windows):

# > jupyter notebook
# > 

# <a id="12"></a> <br>
# ## 5-2 Kaggle Kernel
# Kaggle kernel is an environment just like you use jupyter notebook, it's an **extension** of the where in you are able to carry out all the functions of jupyter notebooks plus it has some added tools like forking et al.

# <a id="13"></a> <br>
# ## 5-3 Colab notebook
# **Colaboratory** is a research tool for machine learning education and research. It’s a Jupyter notebook environment that requires no setup to use.
# <a id="13"></a> <br>
# ### 5-3-1 What browsers are supported?
# Colaboratory works with most major browsers, and is most thoroughly tested with desktop versions of Chrome and Firefox.
# <a id="13"></a> <br>
# ### 5-3-2 Is it free to use?
# Yes. Colaboratory is a research project that is free to use.
# <a id="13"></a> <br>
# ### 5-3-3 What is the difference between Jupyter and Colaboratory?
# Jupyter is the open source project on which Colaboratory is based. Colaboratory allows you to use and share Jupyter notebooks with others without having to download, install, or run anything on your own computer other than a browser.
# ###### [Go to top](#top)

# <a id="15"></a> <br>
# ## 5-5 Loading Packages
# In this kernel we are using the following packages:

#  <img src="http://s8.picofile.com/file/8338227868/packages.png">
# 

# ### 5-5-1 Import

# In[ ]:


from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from pandas import get_dummies
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import numpy
import json
import sys
import csv
import os


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))



# ### 5-5-2 Setup
# 
# A few tiny adjustments for better **code readability**

# In[ ]:


sns.set(style='white', context='notebook', palette='deep')
pylab.rcParams['figure.figsize'] = 12,8
warnings.filterwarnings('ignore')
mpl.style.use('ggplot')
sns.set_style('white')
get_ipython().magic(u'matplotlib inline')


# <a id="16"></a> <br>
# ## 6- Exploratory Data Analysis(EDA)
#  In this section, you'll learn how to use graphical and numerical techniques to begin uncovering the structure of your data. 
#  
# * Which variables suggest interesting relationships?
# * Which observations are unusual?
# * Analysis of the features!
# 
# By the end of the section, you'll be able to answer these questions and more, while generating graphics that are both insightful and beautiful.  then We will review analytical and statistical operations:
# 
# *   5-1 Data Collection
# *   5-2 Visualization
# *   5-3 Data Preprocessing
# *   5-4 Data Cleaning
# <img src="http://s9.picofile.com/file/8338476134/EDA.png">
# 
#  

# <a id="17"></a> <br>
# ## 6-1 Data Collection
# **Data collection** is the process of gathering and measuring data, information or any variables of interest in a standardized and established manner that enables the collector to answer or test hypothesis and evaluate outcomes of the particular collection.[techopedia]
# I start Collection Data by the training and testing datasets into Pandas DataFrames
# ###### [Go to top](#top)

# In[ ]:


# import train and test to play with it
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **<< Note 1 >>**
# 
# * Each **row** is an observation (also known as : sample, example, instance, record)
# * Each **column** is a feature (also known as: Predictor, attribute, Independent Variable, input, regressor, Covariate)

# After loading the data via **pandas**, we should checkout what the content is, description and via the following:

# In[ ]:


type(train)


# In[ ]:


type(test)


# <a id="18"></a> <br>
# ## 6-2 Visualization
# **Data visualization**  is the presentation of data in a pictorial or graphical format. It enables decision makers to see analytics presented visually, so they can grasp difficult concepts or identify new patterns.
# 
# With interactive visualization, you can take the concept a step further by using technology to drill down into charts and graphs for more detail, interactively changing what data you see and how it’s processed.[SAS]
# 
#  In this section I show you  **11 plots** with **matplotlib** and **seaborn** that is listed in the blew picture:
#  <img src="http://s8.picofile.com/file/8338475500/visualization.jpg" />
# 
# ###### [Go to top](#top)

# <a id="19"></a> <br>
# ### 6-2-1 Scatter plot
# 
# Scatter plot Purpose To identify the type of relationship (if any) between two quantitative variables
# 
# 
# 

# In[ ]:


# Modify the graph above by assigning each species an individual color.
g = sns.FacetGrid(train, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend();


# <a id="20"></a> <br>
# ### 6-2-2 Box
# In descriptive statistics, a **box plot** or boxplot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles, hence the terms box-and-whisker plot and box-and-whisker diagram.[wikipedia]

# In[ ]:


train.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
plt.figure()
#This gives us a much clearer idea of the distribution of the input attributes:



# In[ ]:


# To plot the species data using a box plot:

sns.boxplot(x="Fare", y="Age", data=test )
plt.show()


# In[ ]:


# Use Seaborn's striplot to add data points on top of the box plot 
# Insert jitter=True so that the data points remain scattered and not piled into a verticle line.
# Assign ax to each axis, so that each plot is ontop of the previous axis. 

ax= sns.boxplot(x="Fare", y="Age", data=train)
ax= sns.stripplot(x="Fare", y="Age", data=train, jitter=True, edgecolor="gray")
plt.show()


# In[ ]:


# Tweek the plot above to change fill and border color color using ax.artists.
# Assing ax.artists a variable name, and insert the box number into the corresponding brackets

ax= sns.boxplot(x="Fare", y="Age", data=train)
ax= sns.stripplot(x="Fare", y="Age", data=train, jitter=True, edgecolor="gray")

boxtwo = ax.artists[2]
boxtwo.set_facecolor('red')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[1]
boxthree.set_facecolor('yellow')
boxthree.set_edgecolor('black')

plt.show()


# <a id="21"></a> <br>
# ### 6-2-3 Histogram
# We can also create a **histogram** of each input variable to get an idea of the distribution.
# 
# 

# In[ ]:


# histograms
train.hist(figsize=(15,20))
plt.figure()


# It looks like perhaps two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.
# 
# 

# In[ ]:


train["Age"].hist();


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(20,10))
train[train['Survived']==0].Age.plot.hist(ax=ax[0],bins=20,edgecolor='black',color='red')
ax[0].set_title('Survived= 0')
x1=list(range(0,85,5))
ax[0].set_xticks(x1)
train[train['Survived']==1].Age.plot.hist(ax=ax[1],color='green',bins=20,edgecolor='black')
ax[1].set_title('Survived= 1')
x2=list(range(0,85,5))
ax[1].set_xticks(x2)
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Survived',data=train,ax=ax[1])
ax[1].set_title('Survived')
plt.show()


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Survived vs Sex')
sns.countplot('Sex',hue='Survived',data=train,ax=ax[1])
ax[1].set_title('Sex:Survived vs Dead')
plt.show()


# <a id="22"></a> <br>
# ### 6-2-4 Multivariate Plots
# Now we can look at the interactions between the variables.
# 
# First, let’s look at scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables.

# In[ ]:



# scatter plot matrix
pd.plotting.scatter_matrix(train,figsize=(10,10))
plt.figure()


# Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.

# <a id="23"></a> <br>
# ### 6-2-5 violinplots

# In[ ]:


# violinplots on petal-length for each species
sns.violinplot(data=train,x="Fare", y="Age")


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()


# <a id="24"></a> <br>
# ### 6-2-6 pairplot

# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(train, hue="Age")


# From the plot, we can see that the species setosa is separataed from the other two across all feature combinations
# 
# We can also replace the histograms shown in the diagonal of the pairplot by kde.

# In[ ]:


# updating the diagonal elements in a pairplot to show a kde
sns.pairplot(train, hue="Age",diag_kind="kde")


# <a id="25"></a> <br>
# ###  6-2-7 kdeplot

# In[ ]:


# seaborn's kdeplot, plots univariate or bivariate density estimates.
#Size can be changed by tweeking the value used
sns.FacetGrid(train, hue="Survived", size=5).map(sns.kdeplot, "Fare").add_legend()
plt.show()


# <a id="26"></a> <br>
# ### 6-2-8 jointplot

# In[ ]:


# Use seaborn's jointplot to make a hexagonal bin plot
#Set desired size and ratio and choose a color.
sns.jointplot(x="Age", y="Survived", data=train, size=10,ratio=10, kind='hex',color='green')
plt.show()


# <a id="27"></a> <br>
# ###  6-2-9 andrews_curves

# In[ ]:


# we will use seaborn jointplot shows bivariate scatterplots and univariate histograms with Kernel density 
# estimation in the same figure
sns.jointplot(x="Age", y="Fare", data=train, size=6, kind='kde', color='#800000', space=0)


# <a id="28"></a> <br>
# ### 6-2-10 Heatmap

# In[ ]:


plt.figure(figsize=(7,4)) 
sns.heatmap(train.corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()


# In[ ]:


sns.heatmap(train.corr(),annot=False,cmap='RdYlGn',linewidths=0.2)  
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.show()


# ###  6-2-11 Bar Plot

# In[ ]:


train['Pclass'].value_counts().plot(kind="bar");


# ### 6-2-12 Factorplot

# In[ ]:


sns.factorplot('Pclass','Survived',hue='Sex',data=train)
plt.show()


# ### 6-2-13 distplot

# In[ ]:


f,ax=plt.subplots(1,3,figsize=(20,8))
sns.distplot(train[train['Pclass']==1].Fare,ax=ax[0])
ax[0].set_title('Fares in Pclass 1')
sns.distplot(train[train['Pclass']==2].Fare,ax=ax[1])
ax[1].set_title('Fares in Pclass 2')
sns.distplot(train[train['Pclass']==3].Fare,ax=ax[2])
ax[2].set_title('Fares in Pclass 3')
plt.show()


# **<< Note >>**
# 
# **Yellowbrick** is a suite of visual diagnostic tools called “Visualizers” that extend the Scikit-Learn API to allow human steering of the model selection process. In a nutshell, Yellowbrick combines scikit-learn with matplotlib in the best tradition of the scikit-learn documentation, but to produce visualizations for your models! 

# ### 6-2-12 Conclusion
# we have used Python to apply data visualization tools to the Iris dataset. Color and size changes were made to the data points in scatterplots. I changed the border and fill color of the boxplot and violin, respectively.

# <a id="30"></a> <br>
# ## 6-3 Data Preprocessing
# **Data preprocessing** refers to the transformations applied to our data before feeding it to the algorithm.
#  
# Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
# there are plenty of steps for data preprocessing and we just listed some of them :
# * removing Target column (id)
# * Sampling (without replacement)
# * Making part of iris unbalanced and balancing (with undersampling and SMOTE)
# * Introducing missing values and treating them (replacing by average values)
# * Noise filtering
# * Data discretization
# * Normalization and standardization
# * PCA analysis
# * Feature selection (filter, embedded, wrapper)
# 
# ###### [Go to top](#top)

# ## 6-3-1 Features
# Features:
# * numeric
# * categorical
# * ordinal
# * datetime
# * coordinates
# 
# find the type of features in titanic dataset
# <img src="http://s9.picofile.com/file/8339959442/titanic.png" height="700" width="600" />

# ### 6-3-2 Explorer Dataset
# 1- Dimensions of the dataset.
# 
# 2- Peek at the data itself.
# 
# 3- Statistical summary of all attributes.
# 
# 4- Breakdown of the data by the class variable.[7]
# 
# Don’t worry, each look at the data is **one command**. These are useful commands that you can use again and again on future projects.
# 
# ###### [Go to top](#top)

# In[ ]:


# shape
print(train.shape)


# In[ ]:


#columns*rows
train.size


# how many NA elements in every column
# 

# In[ ]:


train.isnull().sum()


# In[ ]:


# remove rows that have NA's
#train = train.dropna()


# 
# We can get a quick idea of how many instances (rows) and how many attributes (columns) the data contains with the shape property.
# 
# You should see 150 instances and 5 attributes:

# for getting some information about the dataset you can use **info()** command

# In[ ]:


print(train.info())


# you see number of unique item for Species with command below:

# In[ ]:


train['Age'].unique()


# In[ ]:


train["Pclass"].value_counts()


# to check the first 5 rows of the data set, we can use head(5).

# In[ ]:


train.head(5) 


# to check out last 5 row of the data set, we use tail() function

# In[ ]:


train.tail() 


# to pop up 5 random rows from the data set, we can use **sample(5)**  function

# In[ ]:


train.sample(5) 


# to give a statistical summary about the dataset, we can use **describe()

# In[ ]:


train.describe() 


# to check out how many null info are on the dataset, we can use **isnull().sum()

# In[ ]:


train.isnull().sum()


# In[ ]:


train.groupby('Pclass').count()


# to print dataset **columns**, we can use columns atribute

# In[ ]:


train.columns


# **<< Note 2 >>**
# in pandas's data frame you can perform some query such as "where"

# In[ ]:


train.where(train ['Age']==30)


# as you can see in the below in python, it is so easy perform some query on the dataframe:

# In[ ]:


train[train['Age']>7.2]


# In[ ]:


# Seperating the data into dependent and independent variables
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values


# **<< Note >>**
# >**Preprocessing and generation pipelines depend on a model type**

# <a id="31"></a> <br>
# ## 6-4 Data Cleaning
# When dealing with real-world data, dirty data is the norm rather than the exception. We continuously need to predict correct values, impute missing ones, and find links between various data artefacts such as schemas and records. We need to stop treating data cleaning as a piecemeal exercise (resolving different types of errors in isolation), and instead leverage all signals and resources (such as constraints, available statistics, and dictionaries) to accurately predict corrective actions.
# 
# The primary goal of data cleaning is to detect and remove errors and **anomalies** to increase the value of data in analytics and decision making. While it has been the focus of many researchers for several years, individual problems have been addressed separately. These include missing value imputation, outliers detection, transformations, integrity constraints violations detection and repair, consistent query answering, deduplication, and many other related problems such as profiling and constraints mining.[8]
# 
# ###### [Go to top](#top)

# In[ ]:


cols = train.columns
features = cols[0:12]
labels = cols[4]
print(features)
print(labels)


# <a id="32"></a> <br>
# ## 7- Model Deployment
# In this section have been applied plenty of  ** learning algorithms** that play an important rule in your experiences and improve your knowledge in case of ML technique.
# 
# > **<< Note 3 >>** : The results shown here may be slightly different for your analysis because, for example, the neural network algorithms use random number generators for fixing the initial value of the weights (starting points) of the neural networks, which often result in obtaining slightly different (local minima) solutions each time you run the analysis. Also note that changing the seed for the random number generator used to create the train, test, and validation samples can change your results.

# ## 7-1 Families of ML algorithms
# There are several categories for machine learning algorithms, below are some of these categories:
# * Linear
#     * Linear Regression
#     * Logistic Regression
#     * Support Vector Machines
# * Tree-Based
#     * Decision Tree
#     * Random Forest
#     * GBDT
# * KNN
# * Neural Networks
# 
# -----------------------------
# And if we  want to categorize ML algorithms with the type of learning, there are below type:
# * Classification
# 
#     * k-Nearest 	Neighbors
#     * LinearRegression
#     * SVM
#     * DT 
#     * NN
#     
# * clustering
# 
#     * K-means
#     * HCA
#     * Expectation Maximization
#     
# * Visualization 	and	dimensionality 	reduction:
# 
#     * Principal 	Component 	Analysis(PCA)
#     * Kernel PCA
#     * Locally -Linear	Embedding 	(LLE)
#     * t-distributed	Stochastic	Neighbor	Embedding 	(t-SNE)
#     
# * Association 	rule	learning
# 
#     * Apriori
#     * Eclat
# * Semisupervised learning
# * Reinforcement Learning
#     * Q-learning
# * Batch learning & Online learning
# * Ensemble  Learning
# 
# **<< Note >>**
# > Here is no method which outperforms all others for all tasks
# 
# ###### [Go to top](#top)

# <a id="33"></a> <br>
# ## 7-2 Prepare Features & Targets
# First of all seperating the data into dependent(Feature) and independent(Target) variables.
# 
# **<< Note 4 >>**
# * X==>>Feature
# * y==>>Target

# In[ ]:



X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ## 7-3 Accuracy and precision
# * **precision** : 
# 
# In pattern recognition, information retrieval and binary classification, precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances, 
# * **recall** : 
# 
# recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. 
# * **F-score** :
# 
# the F1 score is a measure of a test's accuracy. It considers both the precision p and the recall r of the test to compute the score: p is the number of correct positive results divided by the number of all positive results returned by the classifier, and r is the number of correct positive results divided by the number of all relevant samples (all samples that should have been identified as positive). The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
# 
# **What is the difference between accuracy and precision?**
# "Accuracy" and "precision" are general terms throughout science. A good way to internalize the difference are the common "bullseye diagrams". In machine learning/statistics as a whole, accuracy vs. precision is analogous to bias vs. variance.

# <a id="33"></a> <br>
# ## 7-4 K-Nearest Neighbours
# In **Machine Learning**, the **k-nearest neighbors algorithm** (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:
# 
# In k-NN classification, the output is a class membership. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
# In k-NN regression, the output is the property value for the object. This value is the average of the values of its k nearest neighbors.
# k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until classification. The k-NN algorithm is among the simplest of all machine learning algorithms.
# 
# ###### [Go to top](#top)

# -----------------
# <a id="54"></a> <br>
# # 8- Conclusion

# this kernel is not completed yet , I have tried to cover all the parts related to the process of **Machine Learning** with a variety of Python packages and I know that there are still some problems then I hope to get your feedback to improve it.
# 

# you can Fork and Run this kernel on Github:
# > ###### [ GitHub](https://github.com/mjbahmani/Machine-Learning-Workflow-with-Python)
# 
# --------------------------------------
# 
#  **I hope you find this kernel helpful and some <font color="red"><b>UPVOTES</b></font> would be very much appreciated** 

# <a id="55"></a> <br>
# 
# -----------
# 
# # 9- References
# 1. [https://skymind.ai/wiki/machine-learning-workflow](https://skymind.ai/wiki/machine-learning-workflow)
# 
# 1. [Problem-define](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 
# 1. [Sklearn](http://scikit-learn.org/)
# 
# 1. [machine-learning-in-python-step-by-step](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
# 
# 1. [Data Cleaning](http://wp.sigmod.org/?p=2288)
# 
# 1. [competitive data science](https://www.coursera.org/learn/competitive-data-science/)
# 
# 1. [Machine Learning Certification by Stanford University (Coursera)](https://www.coursera.org/learn/machine-learning/)
# 
# 1. [Machine Learning A-Z™: Hands-On Python & R In Data Science (Udemy)](https://www.udemy.com/machinelearning/)
# 
# 1. [Deep Learning Certification by Andrew Ng from deeplearning.ai (Coursera)](https://www.coursera.org/specializations/deep-learning)
# 
# 1. [Python for Data Science and Machine Learning Bootcamp (Udemy)](Python for Data Science and Machine Learning Bootcamp (Udemy))
# 
# 1. [Mathematics for Machine Learning by Imperial College London](https://www.coursera.org/specializations/mathematics-machine-learning)
# 
# 1. [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/deeplearning/)
# 
# 1. [Complete Guide to TensorFlow for Deep Learning Tutorial with Python](https://www.udemy.com/complete-guide-to-tensorflow-for-deep-learning-with-python/)
# 
# 1. [Data Science and Machine Learning Tutorial with Python – Hands On](https://www.udemy.com/data-science-and-machine-learning-with-python-hands-on/)
# 
# 1. [Machine Learning Certification by University of Washington](https://www.coursera.org/specializations/machine-learning)
# 
# 1. [Data Science and Machine Learning Bootcamp with R](https://www.udemy.com/data-science-and-machine-learning-bootcamp-with-r/)
# 
# 1. [Creative Applications of Deep Learning with TensorFlow](https://www.class-central.com/course/kadenze-creative-applications-of-deep-learning-with-tensorflow-6679)
# 
# 1. [Neural Networks for Machine Learning](https://www.class-central.com/mooc/398/coursera-neural-networks-for-machine-learning)
# 
# 1. [Practical Deep Learning For Coders, Part 1](https://www.class-central.com/mooc/7887/practical-deep-learning-for-coders-part-1)
# 
# 1. [Machine Learning](https://www.cs.ox.ac.uk/teaching/courses/2014-2015/ml/index.html)
# 
# 1. [https://www.kaggle.com/ash316/eda-to-prediction-dietanic](https://www.kaggle.com/ash316/eda-to-prediction-dietanic)
# 
# 1. [https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic](https://www.kaggle.com/mrisdal/exploring-survival-on-the-titanic)
# 
# 1. [https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling](https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling)
# 
# 1. [https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy](https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy)
# 
# 1. [https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
# 
# 1. [Top 28 Cheat Sheets for Machine Learning](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# -------------
# 
# ###### [Go to top](#top)

# #### The kernel is not complete and will be updated soon  !!!
