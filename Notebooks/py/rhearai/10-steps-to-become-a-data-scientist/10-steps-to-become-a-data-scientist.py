#!/usr/bin/env python
# coding: utf-8

# 
# <img src="http://s9.picofile.com/file/8338833934/DS.png"/>

# 
# 
# ---------------------------------------------------------------------
# Fork and Run this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red"> UPVOTES</font> would be very much appreciated**
#  
#  -----------
# 

#  <a id="0"></a> <br>
# **Notebook Content**
# 
#  [Introduction](#1)
# 1. [Python](#1)
# 1. [Python Packages](#11)
# 1. [Mathematics and Linear Algebra](#46)
# 1. [Programming & Analysis Tools](#47)
# 1. [Big Data](#49)
# 1. [Data visualization](#50)
# 1. [Data Cleaning](#51)
# 1. [How to solve Problem?](#52)
# 1. [Machine Learning](#53)
# 1. [Deep Learning](#54)

#  ## <div align="center">  10 Steps to Become a Data Scientist</div>
#  <div align="center">**quite practical and far from any theoretical concepts**</div>
# <div style="text-align:center">last update: <b>10/30/2018</b></div>

#  <a id="1"></a> <br>
# #  0- Introduction
# If you Read and Follow **Job Ads** to hire a machine learning expert or a data scientist, you find that some skills you should have to get the job.
# 
# In this Kernel, I want to review **10 skills** that are essentials to get the job
# 
# In fact, this kernel is a reference for **ten other kernels**, which you can learn with them,  all of the skills that you need.
# ## 0-1 papers
# You may not like these 10 steps or have an idea other than this!!! But I just want to list 10 steps that I consider to be the most important thing to do, and surely other skills are needed for the Data Scientist. here I listed some papers around the internet Which can help everyone better understand the work process!!
# 
# 1- [10-steps-to-become-data-scientist-in-2018](https://dzone.com/articles/10-steps-to-become-data-scientist-in-2018)
# 
# 2- [10-steps-to-become-a-data-scientist](http://techtowntraining.com/resources/tools-resources/10-steps-to-become-a-data-scientist)
# 
# 3- [ultimate-learning-path-becoming-data-scientist-2018](https://www.analyticsvidhya.com/blog/2018/01/ultimate-learning-
# path-becoming-data-scientist-2018/)
# 
# 4- [become-a-data-scientist](https://elitedatascience.com/become-a-data-scientist)
# ## 0-2 Books
# There are plenty of E-books(free). here is **10 free machine learning Ebooks** that can make your dreams come true [4]:
# 
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
# 
# ## 0-3 cheat sheets
# Data Science is an ever-growing field, there are numerous tools & techniques to remember. It is not possible for anyone to remember all the functions, operations and formulas of each concept. That’s why we have cheat sheets.
# 1. [Quick Guide to learn Python for Data Science ](https://www.analyticsvidhya.com/blog/2015/05/infographic-quick-guide-learn-python-data-science/)
# 1. [Python for Data Science Cheat sheet ](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf)
# 1. [Python For Data Science Cheat Sheet NumPy ](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
# 1. [Exploratory Data Analysis in Python ](https://www.analyticsvidhya.com/blog/2015/06/infographic-cheat-sheet-data-exploration-python/)
# 1. [Data Visualisation in Python ](https://www.analyticsvidhya.com/blog/2015/06/data-visualization-in-python-cheat-sheet/)
# 1. [Python For Data Science Cheat Sheet Bokeh ](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Python_Bokeh_Cheat_Sheet.pdf)
# 1. [Cheat Sheet: Scikit Learn ](https://www.analyticsvidhya.com/infographics/Scikit-Learn-Infographic.pdf)
# 1. [Steps To Perform Text Data Cleaning in Python](https://www.analyticsvidhya.com/blog/2015/06/quick-guide-text-data-cleaning-python/)
# 1. [Probability Basics  Cheat Sheet](http://www.sas.upenn.edu/~astocker/lab/teaching-files/PSYC739-2016/probability_cheatsheet.pdf)
# 1. [Probability cheat sheet for distribution](http://www.cs.elte.hu/~mesti/valszam/kepletek)
# 
# ## 0-4 Data Science vs. Big Data vs. Data Analytics
# <img src="https://www.simplilearn.com/ice9/free_resources_article_thumb/Data_Science_vs_Data_Analytics_vs_Big_Data_2.png">
# 
# 
# 

#  <a id="1"></a> <br>
# # 1-Python
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [numpy-pandas-matplotlib-seaborn-scikit-learn](https://www.kaggle.com/mjbahmani/numpy-pandas-matplotlib-seaborn-scikit-learn)
# # 1-1 Why you sdould use python?
# 
# As **machine learning engineer** I would like to compare 4 machine learning programming languages(tools). Let's take this a bit deeper. Since most of us are concerned with ML and analysis being a big part of why we are using these programs. I want to list a few advantages and disadvantages of each for who want to start learning them as a data scientist.
# 
# 
# ## 1-1-1 R
# 
# R is a language and environment for statistical computing and graphics. It is a GNU project which is similar to the S language and environment which was developed at Bell Laboratories (formerly AT&amp;T, now Lucent Technologies) by **John Chambers** and colleagues. **R** can be considered as a different implementation of S. There are some important differences, but much code written for S runs unaltered under R.
# 
# 
# ### 1-1-1-1 Advantages of R 
# 
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
# * Worse plotting than Matlab and difficult to implement interactive charts
# * Limited capabilities in creating stand-alone applications
# 
# ----------------------------------------------
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
# ----------------------------------------------------
# 
# ## 1-1-3 MATLAB
# 
# **MATLAB (matrix laboratory)** is a multi-paradigm numerical computing environment. A proprietary programming language developed by MathWorks, MATLAB allows matrix manipulations, plotting of functions and data, implementation of algorithms, a creation of user interfaces, and interfacing with programs written in other languages, including C, C++, C#, Java, Fortran, and Python.
# 
# Although MATLAB is intended primarily for numerical computing, an optional toolbox uses the MuPAD symbolic engine, allowing access to symbolic computing abilities. An additional package, Simulink, adds graphical multi-domain simulation and model-based design for dynamic and embedded systems.
# 
# ### 1-1-3-1 Advantages
# 
# Fastest mathematical and computational platform especially vectorized operations/ linear matrix algebra 
# Commercial level packages for all fields of mathematics and trading
# Very short scripts considering the high integration of all packages
# Best visualization of plots and interactive charts
# Well tested and supported due to it being a commercial product
# Easy to manage multithreaded support and garbage collection
# Best debugger
# 
# ### 1-1-3-2 Disadvantages
# 
# Can not execute - must be translated into another language
# Expensive ~1000 per license and 50+ per additional individual package
# Can not integrate well with other languages
# Hard to detect biases in trading systems (it was built for math and engineering simulations) so extensive testing may be required. EG. look ahead bias
# Worst performance for iterative loops
# Can not develop stand-alone applications at all.
# 
# ## 1-1-4 Octave
# 
# Octave is sort of the GNU answer to the commercial language MATLAB. That is, it is a scripting matrix language, and has a syntax that is about 95% compatible with MATLAB. It's a language designed by engineers, and thus is heavily loaded with routines commonly used by engineers. It has many of the same time series analysis routines, statistics routines, file commands, and plotting commands of the MATLAB language.
# 
# ### 1-1-4-1 Advantages
# 
# First of all, there is no robust Octave compiler available and this is not really necessary either since the software can be installed free of charge.
# Looking at the language element the two packages are identical except for some particularities like nested functions. Octave is under constant active development and every deviation from the Matlab syntax is treated as a bug or at least an issue to be resolved.
# There are also plenty of toolboxes available for octave and as long as a program does not require graphical output there is a good chance that it runs under Octave just like under Matlab without considerable modification.
# Graphics capabilities are clearly an advantage of Matlab. The latest versions include a GUI designer on top of excellent visualization features.
# Octave uses either GNU Plot or JHandles as graphics packages, where the latter is somehow closer to what Matlab provides. However, there are no Octave equivalents to a GUI designer and the visualization mechanisms are somehow limited and not Matlab compatible.
# The same holds for an integrated development environment. There is a project called QTOctave but it is still at an early stage.
# Looking at the collaborate efforts taking place around the Octave community it is likely that this software will soon provide better and possibly even compatible graphics and GUI capabilities and it is well worth a look before buying Matlab.
# 
# ### 1-1-4-2 Disadvantages
# 
# it just a free open source of MATLAB and don't bring us anything new
# 
# 
# ![compare1][1]
# ![compare2][2]
# 
# 
# 
# to sum up, there are several tools for data scientist and machine learning engineer in the below chart you can see which one is more popular than others.
# ![compare1][3]
# **[reference][4]**
# 
# 
#   [1]: https://media.licdn.com/dms/image/C4E12AQHC8vSsbqji1A/article-inline_image-shrink_1500_2232/0?e=1543449600&amp;v=beta&amp;t=lUVejbr2Lwdz9hZuYmVY3upQB2B4ZIjJsP6eiwvrW0A
#   [2]: https://media.licdn.com/dms/image/C4E12AQEH61x6adp36A/article-inline_image-shrink_1000_1488/0?e=1543449600&amp;v=beta&amp;t=EJdx7dx7UMFnOpc5QndIulg9GI2Fd1NyAouEM6s945Q
#   [3]: https://media.licdn.com/dms/image/C4D12AQGPCHd41RDuzg/article-inline_image-shrink_1000_1488/0?e=1543449600&amp;v=beta&amp;t=aksgcN2r_TRkBKgaxYbLh-rZHsMa8xqXiBm-oravz-k
#   [4]: https://www.linkedin.com/pulse/r-vs-python-matlab-octave-mohamadjavad-mj-bahmani/
#   
#   [Download paper](https://github.com/mjbahmani/Machine-Learning-Workflow-with-Python/blob/master/Ebooks/R%20vs%20Python%20vs%20MATLAB%20%20vs%20Octave.pdf)

# <a id="11"></a> <br>
# # 2-Python Packages
# * Numpy
# * Pandas
# * Matplotlib
# * Seaborn
# 
# <img src="http://s8.picofile.com/file/8338227868/packages.png">
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# 
# 
# [numpy-pandas-matplotlib-seaborn-scikit-learn](https://www.kaggle.com/mjbahmani/numpy-pandas-matplotlib-seaborn-scikit-learn)

# <a id="44"></a> <br>
# ## 2-4 SKlearn
# 
# <img src="http://scikit-learn.org/stable/_static/scikit-learn-logo-small.png">
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A-Journey-with-scikit-learn](https://www.kaggle.com/mjbahmani/a-journey-with-scikit-learn)

# <a id="45"></a> <br>
# ##  3- Mathematics and Linear Algebra
# 
# 
# <img src=" https://s3.amazonaws.com/www.mathnasium.com/upload/824/images/algebra.jpg " height="300" width="300">
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [Linear Algebra in 60 Minutes](https://www.kaggle.com/mjbahmani/linear-algebra-in-60-minutes)

# <a id="46"></a> <br>
# ## 4- Programming & Analysis Tools
# 
# * RapidMiner:
# 
# RapidMiner (RM) was originally started in 2006 as an open-source stand-alone software named Rapid-I. Over the years, they have given it the name of RapidMiner and also attained ~35Mn USD in funding. The tool is open-source for old version (below v6) but the latest versions come in a 14-day trial period and licensed after that.
# 
# RM covers the entire life-cycle of prediction modeling, starting from data preparation to model building and finally validation and deployment. The GUI is based on a block-diagram approach, something very similar to Matlab Simulink. There are predefined blocks which act as plug and play devices. You just have to connect them in the right manner and a large variety of algorithms can be run without a single line of code. On top of this, they allow custom R and Python scripts to be integrated into the system.
# 
# There current product offerings include the following:
# 
# RapidMiner Studio: A stand-alone software which can be used for data preparation, visualization and statistical modeling
# RapidMiner Server: It is an enterprise-grade environment with central repositories which allow easy team work, project management and model deployment
# RapidMiner Radoop: Implements big-data analytics capabilities centered around Hadoop
# RapidMiner Cloud: A cloud-based repository which allows easy sharing of information among various devices
# RM is currently being used in various industries including automotive, banking, insurance, life Sciences, manufacturing, oil and gas, retail, telecommunication and utilities.
# 
# * DataRobot:
# 
# DataRobot (DR) is a highly automated machine learning platform built by all time best Kagglers including Jeremy Achin, Thoman DeGodoy and Owen Zhang. Their platform claims to have obviated the need for data scientists. This is evident from a phrase from their website – “Data science requires math and stats aptitude, programming skills, and business knowledge. With DataRobot, you bring the business knowledge and data, and our cutting-edge automation takes care of the rest.”
# 
# DR proclaims to have the following benefits:
# 
# Model Optimization
# Platform automatically detects the best data pre-processing and feature engineering by employing text mining, variable type detection, encoding, imputation, scaling, transformation, etc.
# Hyper-parameters are automatically chosen depending on the error-metric and the validation set score
# Parallel Processing
# Computation is divided over thousands of multi-core servers
# Uses distributed algorithms to scale to large data sets
# Deployment
# Easy deployment facilities with just a few clicks (no need to write any new code)
# For Software Engineers
# Python SDK and APIs available for quick integration of models into tools and softwares.
# 
# * BigML:
# 
# BigML provides a good GUI which takes the user through 6 steps as following:
# 
# Sources: use various sources of information
# Datasets: use the defined sources to create a dataset
# Models: make predictive models
# Predictions: generate predictions based on the model
# Ensembles: create ensemble of various models
# Evaluation: very model against validation sets
# These processes will obviously iterate in different orders. The BigML platform provides nice visualizations of results and has algorithms for solving classification, regression, clustering, anomaly detection and association discovery problems. They offer several packages bundled together in monthly, quarterly and yearly subscriptions. They even offer a free package but the size of the dataset you can upload is limited to 16MB.
# 
# * Google Cloud AutoML:
# 
# Cloud AutoML is part of Google’s Machine Learning suite offerings that enables people with limited ML expertise to build high quality models. The first product, as part of the Cloud AutoML portfolio, is Cloud AutoML Vision. This service makes it simpler to train image recognition models. It has a drag-and-drop interface that let’s the user upload images, train the model, and then deploy those models directly on Google Cloud.
# 
# Cloud AutoML Vision is built on Google’s transfer learning and neural architecture search technologies (among others). This tool is already being used by a lot of organizations. Check out this article to see two amazing real-life examples of AutoML in action, and how it’s producing better results than any other tool.
# 
# * Paxata:
# 
# Paxata is one of the few organizations which focus on data cleaning and preparation, and not the machine learning or statistical modeling part. It is an MS Excel-like application that is easy to use. It also provides visual guidance making it easy to bring together data, find and fix dirty or missing data, and share and re-use data projects across teams. Like the other tools mentioned in this article, Paxata eliminates coding or scripting, hence overcoming technical barriers involved in handling data.
# 
# Paxata platform follows the following process:
# 
# Add Data: use a wide range of sources to acquire data
# Explore: perform data exploration using powerful visuals allowing the user to easily identify gaps in data
# Clean+Change: perform data cleaning using steps like imputation, normalization of similar values using NLP, detecting duplicates
# Shape: make pivots on data, perform grouping and aggregation
# Share+Govern: allows sharing and collaborating across teams with strong authentication and authorization in place
# Combine: a proprietary technology called SmartFusion allows combining data frames with 1 click as it automatically detects the best combination possible; multiple data sets can be combined into a single AnswerSet
# BI Tools: allows easy visualization of the final AnswerSet in commonly used BI tools; also allows easy iterations between data preprocessing and visualization
# Praxata has set its foot in financial services, consumer goods and networking domains. It might be a good tool to use if your work requires extensive data cleaning.
# 
# * Microsoft Azure ML Studio
# 
# When there are so many big name players in this field, how could Microsoft lag behind? The Azure ML Studio is a simple yet powerful browser based ML platform. It has a visual drag-and-drop environment where there is no requirement of coding. They have published comprehensive tutorials and sample experiments for newcomers to get the hang of the tool quickly. It employs a simple five step process:
# 
# Import your dataset
# Perform data cleaning and other preprocessing steps, if necessary
# Split the data into training and testing sets
# Apply built-in ML algorithms to train your model
# Score your model and get your predictions!
# * Amazon Lex:
# 
# Amazon Lex provides an easy-to-use console for building your own chatbot in a matter of minutes. You can build conversational interfaces in your applications or website using Lex. All you need to do is supply a few phrases and Amazon Lex does the rest! It builds a complete Natural Language model using which a customer can interact with your app, using both voice and text.
# 
# It also comes with built-in integration with the Amazon Web Services (AWS) platform. Amazon Lex is a fully managed service so as your user engagement increases, you don’t need to worry about provisioning hardware and managing infrastructure to improve your bot experience.
# 
# In this section, we have discussed various initiatives working towards automating various aspects of solving a data science problem. Some of them are in a nascent research stage, some are open-source and others are already being used in the industry with millions in funding. All of these pose a potential threat to the job of a data scientist, which is expected to grow in the near future. These tools are best suited for people who are not familiar with programming & coding.
# 
# Do you know any other startups or initiatives working in this domain? Please feel free to drop a comment below and enlighten us!

# ## 5- Big Data
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A-Comprehensive-Deep-Learning-Workflow-with-Python](https://www.kaggle.com/mjbahmani/a-comprehensive-deep-learning-workflow-with-python)
# 

# ## 6- Data Visualization
# we will release the full version of Deep Learning **Coming Soon**

# ## 7- Data Cleaning
# we will release the full version of Deep Learning **Coming Soon**

# ## 8- How to solve Problem?
# **Data Science has so many techniques and procedures that can confuse anyone.**
# 
# **Step 1**: Translate your business problem statement into technical one
# 
# Analogous to any other software problem, data science aims at solving a business problem. Most of the times, business problem statements are vague and can be interpreted in multiple ways. This occurs mostly because we generally use qualitative words in our language which cannot be directly translated into a machine readable code.
# 
# Eg. Let’s say we need to develop a solution to reduce crime rate of a city. The term “reduce” can be interpreted as:
# 
# Decreasing crime rate of areas with high crime rate
# Decreasing crime rate of the most common type of crime
# It is a good practice to circle back with the client or the business team who define the problem to decide on the right interpretation.
# 
# **Step 2**: Decide on the supervised learning technique
# 
# The end goal of almost every data science problem is usually classification or regression. Deciding the supervised technique for the problem will help you get more clarity on the business statement.
# 
# Eg. Let’s look at our problem of reducing crime rate. While the problem of reducing crime rate is more of a policy decision, depending on the choice above, we would have to decide if we need to do classification or regression.
# 
# If we need to decrease crime rate of areas with high crime rate, we would need to determine the crime rate rate of an area. This is a regression problem.
# If we need to decrease crime rate of most common type of crime, we would need to determine the most common type of crime in an area. This is a classification problem.
# Again it is a good practice to circle back with the client or the business team who define the problem requirements to clarify on the exact requirement.
# 
# **Step 3**: Literature survey
# 
# Literature Survey is one of the most important step (and often most ignored step) to approach any problem. If you read any article about components of Data Science, you will find computer science, statistics / math and domain knowledge. As it is quite inhuman for someone to have subject expertise in all possible fields, literature survey can often help in bridging the gaps of inadequate subject expertise.
# 
# After going through existing literature related to a problem, I usually try to come up with a set of hypotheses that could form my potential set of features. Going through existing literature helps you understand existing proofs in the domain serving as a guide to take the right direction in your problem. It also helps in interpretation of the results obtained from the prediction models.
# 
# Eg. Going back to our problem of reducing crime rate, if you want to predict crime rate of an area, you would consider factors from general knowledge like demographics, neighboring areas, law enforcement rules etc. Literature survey will help you consider additional variables like climate, mode of transportation, divorce rate etc.
# 
# **Step 4**: Data cleaning
# 
# If you speak with anyone who has spent some time in data science, they will always say that most of their time is spent on cleaning the data. Real world data is always messy. Here are a few common discrepancies in most data-sets and some techniques of how to clean them:
# 
# Missing values
# Missing values are values that are blank in the data-set. This can be due to various reasons like value being unknown, unrecorded, confidential etc. Since the reason for a value being missing is not clear, it is hard to guess the value.
# 
# You could try different techniques to impute missing values starting with simple methods like column mean, median etc. and complex methods like using machine leaning models to estimate missing values.
# 
# Duplicate records
# The challenge with duplicate records is identifying a record being duplicate. Duplicate records often occur while merging data from multiple sources. It could also occur due to human error. To identify duplicates, you could approximate a numeric values to certain decimal places and for text values, fuzzy matching could be a good start. Identification of duplicates could help the data engineering team to improve collection of data to prevent such errors.
# 
# Incorrect values
# Incorrect values are mostly due to human error. For Eg. If there is a field called age and the value is 500, it is clearly wrong. Having domain knowledge of the data will help identify such values. A good technique to identify incorrect values for numerical columns could be to manually look at values beyond 3 standard deviations from the mean to check for correctness.
# 
# **Step 5**: Feature engineering
# 
# Feature Engineering is one of the most important step in any data science problem. Good set of features might make simple models work for your data. If features are not good enough, you might need to go for complex models. Feature Engineering mostly involves:
# 
# Removing redundant features
# If a feature is not contributing a lot to the output value or is a function of other features, you can remove the feature. There are various metrics like AIC and BIC to identify redundant features. There are built in packages to perform operations like forward selection, backward selection etc. to remove redundant features.
# 
# Transforming a feature
# A feature might have a non linear relationship with the output column. While complex models can capture this with enough data, simple models might not be able to capture this. I usually try to visualize different functions of each column like log, inverse, quadratic, cubic etc. and choose the transformation that looks closest to a normal curve.
# 
# **Step 6**: Data modification
# 
# Once the data is cleaned, there are a few modifications that might be needed before applying machine learning models. One of the most common modification would be scaling every column to the same range in order to give same weight to all columns. Some of the other required modifications might be data specific Eg. If output column is skewed, you might need to up-sample or down-sample.
# 
# Steps 7 through 9 are iterative.
# 
# **Step 7**: Modelling
# 
# Once I have the data ready, I usually start with trying all the standard machine learning models. If it is a classification problem, a good start will beLogistic Regression, Naive Bayes, k-Nearest Neighbors, Decision Tree etc. If it is a regression problem, you could try linear regression, regression tree etc. The reason for starting with simple models is that simple models have lesser parameters to alter. If we start with a complex model like Neural Network orSupport Vector Machines, there are so many parameters that you could change that trying all options exhaustively might be time consuming.
# 
# Each of the machine learning models make some underlying assumptions about the data. For Eg. Linear Regression / Logistic Regression assumes that the data comes from a linear combination of input parameters. Naive Bayes makes an assumption that the input parameters are independent of each other. Having the knowledge of these assumptions can help you judge the results of the different models. It is often helpful to visualize the actual vs predicted values to see these differences.
# 
# **Step 8**: Model comparison
# 
# One of the most standard technique to evaluate different machine learning models would be through the process of cross validation. I usually choose 10-fold cross validation but you may choose the right cross validation split based on the size of the data. Cross validation basically brings out an average performance of a model. This can help eliminate choosing a model that performs good specific to the data or in other words avoid over-fitting. It is often a good practice to randomize data before cross validation.
# 
# A good technique to compare performance of different models is ROC curves. ROC curves help you visualize performance of different models across different thresholds. While ROC curves give a holistic sense of model performance, based on the business decision, you must choose the performance metric like Accuracy, True Positive Rate, False Positive Rate, F1-Score etc.
# 
# **Step 9**: Error analysis
# 
# At this point, you have tried a bunch of machine learning models and got the results. It is a good usage of time to not just look at the results like accuracy or True Positive Rate but to look at the set of data points that failed in some of the models. This will help you understand the data better and improve the models faster than trying all possible combinations of models. This is the time to try ensemble models like Random Forest, Gradient Boosting or a meta model of your own [Eg. Decision tree + Logistic Regression]. Ensemble models are almost always guaranteed to perform better than any standard model.
# 
# **Step 10**: Improving your best model
# 
# Once I have the best model, I usually plot training vs testing accuracy [or the right metric] against the number of parameters. Usually, it is easy to check training and testing accuracy against number of data points. Basically this plot will tell you whether your model is over-fitting or under-fitting. This articleDetecting over-fitting vs under-fitting explains this concept clearly.
# 
# Understanding if your model is over-fitting or under-fitting will tell you how to proceed with the next steps. If the model is over-fitting, you might consider collecting more data. If the model is under-fitting, you might consider making the models more complex. [Eg. Adding higher order terms to a linear / logistic regression]
# 
# **Step 11**: Deploying the model
# 
# Once you have your final model, you would want the model to be deployed so that it automatically predicts output for new data point without retraining. While you can derive a formula for simple models like Linear Regression, Logistic Regression, Decision Tree etc. , it is not so straight forward for complex models like SVM, Neural Networks, Random Forest etc. I’m not very familiar with other languages but Python has a library called pickle which allows you to save models and use it to predict output for new data.
# 
# **Step 12**: Adding feedback
# 
# Usually, data for any data science problem is historical data. While this might be similar to the current data up-to a certain degree, it might not be able to capture the current trends or changes. For Eg. If you are using population as an input parameter, while population from 2015–2016 might vary slightly, if you use the model after 5 years, it might give incorrect results.
# 
# One way to deal with this problem is to keep retraining your model with additional data. This might be a good option but retraining a model might be time consuming. Also, if you have applications in which data inflow is huge, this might need to be done at regular intervals. An alternative and a better option would be to use active learning. Active learning basically tries to use real time data as feedback and automatically update the model. The most common approaches to do this are Batch Gradient Descent and Stochastic Gradient Descent. It might be appropriate to use the right approach based on the application.
# 
# Concluding remarks
# 
# The field of data science is really vast. People spend their lifetime researching on individual topics discussed above. As a data scientist, you would mostly have to solve business problems than researching on individual subtopics. Additionally, you will have to explain the technical process and results to business teams who might not have enough technical knowledge. Thus, while you might not need a very in-depth knowledge of every technique, you need to have enough clarity to abstract the technical process and results and explain it in business terms.[3]
# 

# ## 9- Machine learning  
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A Comprehensive ML Workflow with Python](https://www.kaggle.com/mjbahmani/a-comprehensive-ml-workflow-with-python)
# 
# 

# ##  10- Deep Learning
# 
# for Reading this section **please** fork and upvote  this kernel:
# 
# [A-Comprehensive-Deep-Learning-Workflow-with-Python](https://www.kaggle.com/mjbahmani/a-comprehensive-deep-learning-workflow-with-python)
# 

# ## References:
# 1. [Coursera](https://www.coursera.org/specializations/data-science-python)
# 2. [Hands-On Machine Learning with Scikit-Learn and TensorFlow](http://shop.oreilly.com/product/0636920052289.do)
# 3. [How to solve Problem](https://www.linkedin.com/pulse/how-i-approach-data-science-problem-ganesh-n-prasad/)
# 4. [Top 28 Cheat Sheets for Machine Learning, Data Science, Probability, SQL & Big Data](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)

# ---------------------------------------------------------------------
# Fork and Run this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
#  
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red">UPVOTES</font> would be very much appreciated**
#  
#  -----------

# # Not completed yet!!!
# 
# **Update every two days**
