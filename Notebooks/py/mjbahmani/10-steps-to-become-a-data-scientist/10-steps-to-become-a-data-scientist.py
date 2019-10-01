#!/usr/bin/env python
# coding: utf-8

# 
# <img src="http://s9.picofile.com/file/8338833934/DS.png"/>

# 
# 
# ---------------------------------------------------------------------
# Fork and Run this kernel on GitHub:
# > #### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# -------------------------------------------------------------------------------------------------------------
#  <b>I hope you find this kernel helpful and some <font color="red"> UPVOTES</font> would be very much appreciated<b/>
#  
#  -----------
# 

#  <a id="top"></a> <br>
# **Notebook Content**
# 
#  [Introduction](#Introduction)
# 1. [Python](#Python)
# 1. [Python Packages](#Python Packages)
# 1. [Mathematics and Linear Algebra](#Mathematics and Linear Algebra)
# 1. [Programming & Analysis Tools](#Programming & Analysis Tools)
# 1. [Big Data](#Big Data)
# 1. [Data visualization](#Data visualization)
# 1. [Data Cleaning](#Data Cleaning)
# 1. [How to solve Problem?](#How to solve Problem?)
# 1. [Machine Learning](#Machine Learning)
# 1. [Deep Learning](#Deep Learning)

#  ## <div align="center">  10 Steps to Become a Data Scientist</div>
#  <div align="center">**quite practical and far from any theoretical concepts**</div>
# <div style="text-align:center">last update: <b>11/20/2018</b></div>

#  <a id="Introduction"></a> <br>
# # Introduction
# If you Read and Follow **Job Ads** to hire a machine learning expert or a data scientist, you find that some skills you should have to get the job. In this Kernel, I want to review **10 skills** that are essentials to get the job. In fact, this kernel is a reference for **10 other kernels**, which you can learn with them,  all of the skills that you need. 
# 
# **Ready to learn**! you will learn 10 skills as data scientist: [Machine Learning](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python), [Deep Learning](https://www.kaggle.com/mjbahmani/top-5-deep-learning-frameworks-tutorial), [Data Cleaning](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora), [EDA](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2), [Learn Python](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1), [Learn python packages](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2) such as Numpy, Pandas, Seaborn, Matplotlib, Plotly, Tensorfolw, Theano...., [Linear Algebra](https://www.kaggle.com/mjbahmani/linear-algebra-for-data-scientists), [Big Data](https://www.kaggle.com/mjbahmani/a-data-science-framework-for-quora), Analysis Tools and solve real problem for instance predict house prices.
# ###### [go to top](#top)

#  <a id="1"></a> <br>
# # 1-Python
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [numpy-pandas-matplotlib-seaborn-scikit-learn](https://www.kaggle.com/mjbahmani/numpy-pandas-matplotlib-seaborn-scikit-learn)
# # 1-1 Why you should use python?
# 
# As **machine learning engineer** I would like to compare 4 machine learning programming languages(tools). Let's take this a bit deeper. Since most of us are concerned with ML and analysis being a big part of why we are using these programs. I want to list a few advantages and disadvantages of each for who want to start learning them as a data scientist.
# ## 1-1-1 R
# R is a language and environment for statistical computing and graphics. It is a GNU project which is similar to the S language and environment which was developed at Bell Laboratories (formerly AT&amp;T, now Lucent Technologies) by **John Chambers** and colleagues. **R** can be considered as a different implementation of S. There are some important differences, but much code written for S runs unaltered under R.
# 
# ### 1-1-1-1 Advantages of R 
# 
# * End To End development to execution (some brokers packages allows execution, IB)
# * Rapid development speed (60% fewer lines vs python, ~500% less than C)
# * A large number of Open Source Packages
# * Mature quantitative trading packages( quantstrat, quantmod, performanceanalyitics, xts)
# * Largest Community
# * Can integrate into C++/C with rcpp
# 
# ### 1-1-1-2 Disadvantages of R 
# 
# * Slow vs Python especially in iterative loops and non vectorized functions
# * Worse plotting than python and difficult to implement interactive charts
# * Limited capabilities in creating stand-alone applications
# 
# ## 1-1-2 Python
# 
# Python is an interpreted high-level programming language for general-purpose programming. Created by Guido van Rossum and first released in 1991, Python has a design philosophy that emphasizes code readability, notably using significant whitespace. It provides constructs that enable clear programming on both small and large scales.
# 
# ### 1-1-2-1Advantages
# 
# * End To End development to execution (some brokers packages allows execution, IB)
# * Open source packages( Pandas, Numpy, scipy) 
# * Trading Packages(zipline, pybacktest, pyalgotrade)
# * best for general programming and application development
# * can be a "glue" language to connect R, C++, and others (python)
# * Fastest general speed especially in iterative loops
# 
# ### 1-1-2-2 Disadvantages
# 
# * immature packages especially trading packages
# * some packages are not compatible with others or contain overlap
# * smaller community than R in finance
# * More code required for same operations vs R or Matlab
# * Silent errors that can take a very long time to track down (even with visual debuggers / IDE)
# 
# ## 1-1-3 MATLAB
# 
# **MATLAB (matrix laboratory)** is a multi-paradigm numerical computing environment. A proprietary programming language developed by MathWorks, MATLAB allows matrix manipulations, plotting of functions and data, implementation of algorithms, a creation of user interfaces, and interfacing with programs written in other languages, including C, C++, C#, Java, Fortran, and Python.
# Although MATLAB is intended primarily for numerical computing, an optional toolbox uses the MuPAD symbolic engine, allowing access to symbolic computing abilities. An additional package, Simulink, adds graphical multi-domain simulation and model-based design for dynamic and embedded systems.
# 
# ### 1-1-3-1 Advantages
# 
# 1. Fastest mathematical and computational platform especially vectorized operations/ linear matrix algebra 
# 1. Commercial level packages for all fields of mathematics and trading
# 1. Very short scripts considering the high integration of all packages
# 1. Best visualization of plots and interactive charts
# 1. Well tested and supported due to it being a commercial product
# 1. Easy to manage multithreaded support and garbage collection
# 1. Best debugger
# 
# ### 1-1-3-2 Disadvantages
# 
# 1. Can not execute - must be translated into another language
# 1. Expensive ~1000 per license and 50+ per additional individual package
# 1. Can not integrate well with other languages
# 1. Hard to detect biases in trading systems (it was built for math and engineering simulations) so extensive testing may be required. EG. look ahead bias
# 1. Worst performance for iterative loops
# 1. Can not develop stand-alone applications at all.
# 
# ## 1-1-4 Octave
# 
# Octave is sort of the GNU answer to the commercial language MATLAB. That is, it is a scripting matrix language, and has a syntax that is about 95% compatible with MATLAB. It's a language designed by engineers, and thus is heavily loaded with routines commonly used by engineers. It has many of the same time series analysis routines, statistics routines, file commands, and plotting commands of the MATLAB language.
# 
# ### 1-1-4-1 Advantages
# 
# 1. First of all, there is no robust Octave compiler available and this is not really necessary either since the software can be installed free of charge.
# 1. Looking at the language element the two packages are identical except for some particularities like nested functions. Octave is under constant active development and every deviation from the Matlab syntax is treated as a bug or at least an issue to be resolved.
# 1. There are also plenty of toolboxes available for octave and as long as a program does not require graphical output there is a good chance that it runs under Octave just like under Matlab without considerable modification.
# 1. Graphics capabilities are clearly an advantage of Matlab. The latest versions include a GUI designer on top of excellent visualization features.
# 1. Octave uses either GNU Plot or JHandles as graphics packages, where the latter is somehow closer to what Matlab provides. However, there are no Octave equivalents to a GUI designer and the visualization mechanisms are somehow limited and not Matlab compatible.
# 1. The same holds for an integrated development environment. There is a project called QTOctave but it is still at an early stage.
# 1. Looking at the collaborate efforts taking place around the Octave community it is likely that this software will soon provide better and possibly even compatible graphics and GUI capabilities and it is well worth a look before buying Matlab.
# 
# ### 1-1-4-2 Disadvantages
# 
# 1. it just a free open source of MATLAB and don't bring us anything new
# 
# ## 1-2 Conclusion
# 
# We can now see a number of comparisons already made by other sources.
# 
# <img src='https://media.licdn.com/dms/image/C4E12AQHC8vSsbqji1A/article-inline_image-shrink_1500_2232/0?e=1543449600&amp;v=beta&amp;t=lUVejbr2Lwdz9hZuYmVY3upQB2B4ZIjJsP6eiwvrW0A'>
# <img src='https://media.licdn.com/dms/image/C4E12AQEH61x6adp36A/article-inline_image-shrink_1000_1488/0?e=1543449600&amp;v=beta&amp;t=EJdx7dx7UMFnOpc5QndIulg9GI2Fd1NyAouEM6s945Q'>
# 
# 
# 
# To sum up, there are several tools for data scientist and machine learning engineer in the below chart you can see which one is more popular than others.
# <img src='https://media.licdn.com/dms/image/C4D12AQGPCHd41RDuzg/article-inline_image-shrink_1000_1488/0?e=1543449600&amp;v=beta&amp;t=aksgcN2r_TRkBKgaxYbLh-rZHsMa8xqXiBm-oravz-k'>
# [reference](https://www.linkedin.com/pulse/r-vs-python-matlab-octave-mohamadjavad-mj-bahmani/)
# 
#  
#   
#   [Download paper](https://github.com/mjbahmani/Machine-Learning-Workflow-with-Python/blob/master/Ebooks/R%20vs%20Python%20vs%20MATLAB%20%20vs%20Octave.pdf)
#   ###### [go to top](#top)

# <a id="11"></a> <br>
# # 2-Python Packages
# 1. Numpy
# 1. Pandas
# 1. Matplotlib
# 1. Seaborn
# 1. TensorFlow
# 1. NLTK
# 1. Sklearn
# 
# <img src="http://s8.picofile.com/file/8338227868/packages.png">
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# 
# 
# 1. [The data scientist's toolbox tutorial 1](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-1)
# 
# 1. [The data scientist's toolbox tutorial 2](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# ###### [go to top](#top)

# In[ ]:


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


# <a id="Mathematics and Linear Algebra"></a> <br>
# ##  3- Mathematics and Linear Algebra
# Linear algebra is the branch of mathematics that deals with vector spaces. good understanding of Linear Algebra is intrinsic to analyze Machine Learning algorithms, especially for Deep Learning where so much happens behind the curtain.you have my word that I will try to keep mathematical formulas & derivations out of this completely mathematical topic and I try to cover all of subject that you need as data scientist.
# 
# <img src=" https://s3.amazonaws.com/www.mathnasium.com/upload/824/images/algebra.jpg " height="300" width="300">
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [Linear Algebra for Data Scientists](https://www.kaggle.com/mjbahmani/linear-algebra-in-60-minutes)
# ###### [go to top](#top)

# <a id="Programming & Analysis Tools"></a> <br>
# ## 4- Programming & Analysis Tools
# 
# * **RapidMiner**:
# 
# RapidMiner (RM) was originally started in 2006 as an open-source stand-alone software named Rapid-I. Over the years, they have given it the name of RapidMiner and also attained ~35Mn USD in funding. The tool is open-source for old version (below v6) but the latest versions come in a 14-day trial period and licensed after that.
# 
# RM covers the entire life-cycle of prediction modeling, starting from data preparation to model building and finally validation and deployment. The GUI is based on a block-diagram approach, something very similar to Matlab Simulink. There are predefined blocks which act as plug and play devices. You just have to connect them in the right manner and a large variety of algorithms can be run without a single line of code. On top of this, they allow custom R and Python scripts to be integrated into the system.
# 
# There current product offerings include the following:
# 
# 1. RapidMiner Studio: A stand-alone software which can be used for data preparation, visualization and statistical modeling
# 1. RapidMiner Server: It is an enterprise-grade environment with central repositories which allow easy team work, project management and model deployment
# 1. RapidMiner Radoop: Implements big-data analytics capabilities centered around Hadoop
# 1. RapidMiner Cloud: A cloud-based repository which allows easy sharing of information among various devices
# RM is currently being used in various industries including automotive, banking, insurance, life Sciences, manufacturing, oil and gas, retail, telecommunication and utilities.
# 
# * **DataRobot**:
# 
# DataRobot (DR) is a highly automated machine learning platform built by all time best Kagglers including Jeremy Achin, Thoman DeGodoy and Owen Zhang. Their platform claims to have obviated the need for data scientists. This is evident from a phrase from their website – “Data science requires math and stats aptitude, programming skills, and business knowledge. With DataRobot, you bring the business knowledge and data, and our cutting-edge automation takes care of the rest.”
# 
# DR proclaims to have the following benefits:
# 
# 1. Model Optimization
# Platform automatically detects the best data pre-processing and feature engineering by employing text mining, variable type detection, encoding, imputation, scaling, transformation, etc.
# Hyper-parameters are automatically chosen depending on the error-metric and the validation set score
# 1. Parallel Processing
# Computation is divided over thousands of multi-core servers
# Uses distributed algorithms to scale to large data sets
# 1. Deployment
# Easy deployment facilities with just a few clicks (no need to write any new code)
# 1. For Software Engineers
# Python SDK and APIs available for quick integration of models into tools and softwares.
# 
# **BigML**:
# 
# BigML provides a good GUI which takes the user through 6 steps as following:
# 
# 1. Sources: use various sources of information
# 1. Datasets: use the defined sources to create a dataset
# 1. Models: make predictive models
# 1. Predictions: generate predictions based on the model
# 1. Ensembles: create ensemble of various models
# 1. Evaluation: very model against validation sets
# These processes will obviously iterate in different orders. The BigML platform provides nice visualizations of results and has algorithms for solving classification, regression, clustering, anomaly detection and association discovery problems. They offer several packages bundled together in monthly, quarterly and yearly subscriptions. They even offer a free package but the size of the dataset you can upload is limited to 16MB.
# 
# **Google Cloud AutoML**:
# 
# Cloud AutoML is part of Google’s Machine Learning suite offerings that enables people with limited ML expertise to build high quality models. The first product, as part of the Cloud AutoML portfolio, is Cloud AutoML Vision. This service makes it simpler to train image recognition models. It has a drag-and-drop interface that let’s the user upload images, train the model, and then deploy those models directly on Google Cloud.
# 
# Cloud AutoML Vision is built on Google’s transfer learning and neural architecture search technologies (among others). This tool is already being used by a lot of organizations. Check out this article to see two amazing real-life examples of AutoML in action, and how it’s producing better results than any other tool.
# 
# **Paxata**:
# 
# Paxata is one of the few organizations which focus on data cleaning and preparation, and not the machine learning or statistical modeling part. It is an MS Excel-like application that is easy to use. It also provides visual guidance making it easy to bring together data, find and fix dirty or missing data, and share and re-use data projects across teams. Like the other tools mentioned in this article, Paxata eliminates coding or scripting, hence overcoming technical barriers involved in handling data.
# 
# Paxata platform follows the following process:
# 
# Add Data: use a wide range of sources to acquire data
# 1. Explore: perform data exploration using powerful visuals allowing the user to easily identify gaps in data
# Clean+Change: perform data cleaning using steps like imputation, normalization of similar values using NLP, detecting duplicates
# 1. Shape: make pivots on data, perform grouping and aggregation
# Share+Govern: allows sharing and collaborating across teams with strong authentication and authorization in place
# Combine: a proprietary technology called SmartFusion allows combining data frames with 1 click as it automatically detects the best combination possible; multiple data sets can be combined into a single AnswerSet
# 1. BI Tools: allows easy visualization of the final AnswerSet in commonly used BI tools; also allows easy iterations between data preprocessing and visualization
# Praxata has set its foot in financial services, consumer goods and networking domains. It might be a good tool to use if your work requires extensive data cleaning.
# 
# **Microsoft Azure ML Studio**
# 
# When there are so many big name players in this field, how could Microsoft lag behind? The Azure ML Studio is a simple yet powerful browser based ML platform. It has a visual drag-and-drop environment where there is no requirement of coding. They have published comprehensive tutorials and sample experiments for newcomers to get the hang of the tool quickly. It employs a simple five step process:
# 
# 1. Import your dataset
# 1. Perform data cleaning and other preprocessing steps, if necessary
# 1. Split the data into training and testing sets
# 1. Apply built-in ML algorithms to train your model
# 1. Score your model and get your predictions!
# **Amazon Lex**:
# 
# Amazon Lex provides an easy-to-use console for building your own chatbot in a matter of minutes. You can build conversational interfaces in your applications or website using Lex. All you need to do is supply a few phrases and Amazon Lex does the rest! It builds a complete Natural Language model using which a customer can interact with your app, using both voice and text.
# 
# It also comes with built-in integration with the Amazon Web Services (AWS) platform. Amazon Lex is a fully managed service so as your user engagement increases, you don’t need to worry about provisioning hardware and managing infrastructure to improve your bot experience.
# 
# In this section, we have discussed **various** initiatives working towards automating various aspects of solving a data science problem. Some of them are in a nascent research stage, some are open-source and others are already being used in the industry with millions in funding. All of these pose a potential threat to the job of a data scientist, which is expected to grow in the near future. These tools are best suited for people who are not familiar with programming & coding.
# ###### [go to top](#top)

# <a id="Big Data"></a> <br>
# ## 5- Big Data
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A-Comprehensive-Deep-Learning-Workflow-with-Python](https://www.kaggle.com/mjbahmani/a-comprehensive-deep-learning-workflow-with-python)
# 

# <a id="Data Visualization"></a> <br>
# ## 6- Data Visualization
# for Reading this section **please** fork and upvote  this kernel:
# 
# [Exploratory Data Analysis for Meta Kaggle Dataset](https://www.kaggle.com/mjbahmani/exploratory-data-analysis-for-meta-kaggle-dataset)

# <a id="Data Cleaning"></a> <br>
# ## 7- Data Cleaning
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A-Comprehensive-Deep-Learning-Workflow-with-Python](https://www.kaggle.com/mjbahmani/a-comprehensive-deep-learning-workflow-with-python)

# <a id="How to solve Problem?"></a> <br>
# ## 8- How to solve Problem?
# If you have already read some [machine learning books](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/Ebooks). You have noticed that there are different ways to stream data into machine learning.
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
# ## 8-1 Real world Application Vs Competitions
# Just a simple comparison between real-world apps with competitions:
# <img src="http://s9.picofile.com/file/8339956300/reallife.png" height="600" width="500" />
# **you should	feel free	to	adapt 	this	checklist 	to	your needs**
#  
# ## 8-2 Problem Definition
# I think one of the important things when you start a new machine learning project is Defining your problem. that means you should understand business problem.( **Problem Formalization**)
# 
# Problem Definition has four steps that have illustrated in the picture below:
# <img src="http://s8.picofile.com/file/8338227734/ProblemDefination.png">
#  
# ### 8-2-1 Problem Feature
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
# ### 8-2-2 Aim
# 
# It is your job to predict if a passenger survived the sinking of the Titanic or not.  For each PassengerId in the test set, you must predict a 0 or 1 value for the Survived variable.
# 
#  
# ### 8-2-3 Variables
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
# 
#     * 1st = Upper
#     * 2nd = Middle
#     * 3rd = Lower
#     
# 5. **Embarked** ==>> nominal datatype 
# 6. **Name** ==>> nominal datatype . It could be used in feature engineering to derive the gender from title
# 7. **Sex** ==>>  nominal datatype 
# 8. **Ticket** ==>> that have no impact on the outcome variable. Thus, they will be excluded from analysis
# 9. **Cabin** ==>>  is a nominal datatype that can be used in feature engineering
# 11. **Fare** ==>>  Indicating the fare
# 12. **PassengerID ** ==>> have no impact on the outcome variable. Thus, it will be excluded from analysis
# 11. **Survival** is ==>> **[dependent variable](http://www.dailysmarty.com/posts/difference-between-independent-and-dependent-variables-in-machine-learning)** , 0 or 1
# 
# 
# **<< Note >>**
# 
# > You must answer the following question:
# How does your company expact to use and benfit from your model.
# ###### [Go to top](#top)

# <a id="Machine learning"></a> <br>
# ## 9- Machine learning  
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A Comprehensive ML Workflow with Python](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 
# 

# <a id="Deep Learning"></a> <br>
# ##  10- Deep Learning
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A-Comprehensive-Deep-Learning-Workflow-with-Python](https://www.kaggle.com/mjbahmani/a-comprehensive-deep-learning-workflow-with-python)
# 
# ---------------------------
# 

# <a id="Introducing other sources"></a> <br>
# ## 11- Introducing other sources
# In this section I introduce additional resources for further study.
# ## 11-1 papers
# You may not like these 10 steps or have an idea other than this!!! But I just want to list 10 steps that I consider to be the most important thing to do, and surely other skills are needed for the Data Scientist. here I listed some papers around the internet Which can help everyone better understand the work process!!
# 
# 1- [10-steps-to-become-data-scientist-in-2018](https://dzone.com/articles/10-steps-to-become-data-scientist-in-2018)
# 
# 2- [10-steps-to-become-a-data-scientist](http://techtowntraining.com/resources/tools-resources/10-steps-to-become-a-data-scientist)
# 
# 3- [ultimate-learning-path-becoming-data-scientist-2018](https://www.analyticsvidhya.com/blog/2018/01/ultimate-learning-
# path-becoming-data-scientist-2018/)
# 
# 4- [become-a-data-scientist](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# ## 11-2 Books
# There are plenty of E-books(free). here is **10 free machine learning Ebooks** that can make your dreams come true [4]:
# 
# 1. [Probability and Statistics for Programmers](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/Ebooks)
# 2. [Bayesian Reasoning and Machine Learning](http://web4.cs.ucl.ac.uk/staff/D.Barber/textbook/091117.pdf)
# 2. [An Introduction to Statistical Learning](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/Ebooks)
# 2. [Understanding Machine Learning](http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/index.html)
# 2. [A Programmer’s Guide to Data Mining](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/Ebooks)
# 2. [Mining of Massive Datasets](http://infolab.stanford.edu/~ullman/mmds/book.pdf)
# 2. [A Brief Introduction to Neural Networks](http://www.dkriesel.com/_media/science/neuronalenetze-en-zeta2-2col-dkrieselcom.pdf)
# 2. [Deep Learning](http://www.deeplearningbook.org/)
# 2. [Natural Language Processing with Python](https://www.researchgate.net/publication/220691633_Natural_Language_Processing_with_Python)
# 2. [Machine Learning Yearning](http://www.mlyearning.org/)
# 
# ## 11-3 cheat sheets
# Data Science is an ever-growing field, there are numerous tools & techniques to remember. It is not possible for anyone to remember all the functions, operations and formulas of each concept. That’s why we have cheat sheets.
# 1. [Quick Guide to learn Python for Data Science ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/cheatsheets)
# 1. [Python for Data Science Cheat sheet ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/cheatsheets)
# 1. [Python For Data Science Cheat Sheet NumPy ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/cheatsheets)
# 1. [Exploratory Data Analysis in Python ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/cheatsheets)
# 1. [Data Visualisation in Python ](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist/tree/master/cheatsheets ](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Bokeh_Cheat_Sheet.pdf)
# 1. [Cheat Sheet: Scikit Learn ](https://www.analyticsvidhya.com/infographics/Scikit-Learn-Infographic.pdf)
# 1. [Steps To Perform Text Data Cleaning in Python](https://www.analyticsvidhya.com/blog/2015/06/quick-guide-text-data-cleaning-python/)
# 1. [Probability Basics  Cheat Sheet](http://www.sas.upenn.edu/~astocker/lab/teaching-files/PSYC739-2016/probability_cheatsheet.pdf)
# 1. [Probability cheat sheet for distribution](http://www.cs.elte.hu/~mesti/valszam/kepletek)

# <a id="References"></a> <br>
# ## References:
# 1. [Coursera](https://www.coursera.org/specializations/data-science-python)
# 1. [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)
# 1. [Top 28 Cheat Sheets for Machine Learning, Data Science, Probability, SQL & Big Data](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# 1. [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 

# ---------------------------------------------------------------------
# Fork and Run this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
#  
# 
# -------------------------------------------------------------------------------------------------------------
#  <b>I hope you find this kernel helpful and some <font color="red">UPVOTES</font> would be very much appreciated</b>
#  
#  -----------

# ## Not completed yet!!!
# 
# **Update every two days**
# ###### [go to top](#top)
