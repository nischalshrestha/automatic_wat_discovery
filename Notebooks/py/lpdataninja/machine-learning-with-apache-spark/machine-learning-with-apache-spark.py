#!/usr/bin/env python
# coding: utf-8

# ### Apache Spark : The Unified Analytics Engine
# The largest open source project in data processing framework that can do ETL, analytics, machine learning and graph processing on large volumes of data at rest (batch processing) or in motion (streaming processing) with rich high-level APIs for the programming languages like Scala, Python, Java and R.
# 
# Spark has seen immense growth over the past several years. Hundreds of contributors working collectively have made Spark an amazing piece of technology powering the de facto standard for big data processing and data sciences across all industries. 
# 
# ![](http://)![](http://)Internet powerhouses such as Netflix, Yahoo, and eBay have deployed Spark at massive scale, collectively processing multiple petabytes of data on clusters of over 8,000 nodes. 

# ### Why Spark ?
# 
# Typically when you think of a computer you think about one machine sitting on your desk at home or at work. 
# This machine works perfectly well for applying machine learning on small dataset . However, when you have huge dataset(in tera bytes or giga bytes), there are some things that your computer is not powerful enough to perform.
# One particularly challenging area is data processing. Single machines do not have enough power and resources to perform
# computations on huge amounts of information (or you may have to wait for the computation to finish).
# 
# A cluster, or group of machines, pools the resources of many machines together allowing us to use all the cumulative
# resources as if they were one. Now a group of machines alone is not powerful, you need a framework to coordinate
# work across them. Spark is a tool for just that, managing and coordinating the execution of tasks on data across a
# cluster of computers.

# ### Spark Architecture
# 
# Apache Spark allows you to treat many machines as one machine and this is done via a master-worker type architecture where there is a driver or master node in the cluster, accompanied by worker nodes. The master sends work to the workers and either instructs them to pull to data from memory or from disk (or from another data source).
# 

# ### Read more about Architecture
# https://spark.apache.org/docs/latest/cluster-overview.html

# ### Spark Applications
# 
# Spark Applications consist of a driver process and a set of executor processes. The driver process runs your main()
# function, sits on a node in the cluster, and is responsible for three things: maintaining information about the Spark
# Application; responding to a user’s program or input; and analyzing, distributing, and scheduling work across the
# executors (defined momentarily). The driver process is absolutely essential - it’s the heart of a Spark Application and
# maintains all relevant information during the lifetime of the application.
# 
# The executors are responsible for actually executing the work that the driver assigns them. This means, each
# executor is responsible for only two things: executing code assigned to it by the driver and reporting the state of the
# computation, on that executor, back to the driver node.

# ### Running Code
# 
# I am using freely available databricks stand alone community edition server (https://community.cloud.databricks.com) as Spark library currently not available in Kaggle directly.
# 
# Here is Spark code for LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, Gradient-boosted tree classifier, NaiveBayes & Support Vector Machine on Titanic dataset
# 
# ##### Databricks Notebook :
# https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5722190290795989/3865595167034368/8175309257345795/latest.html
# 
# ##### Github :
# https://github.com/lp-dataninja/SparkML/blob/master/kaggle-titanic-pyspark.ipynb
# 

# ### Reference 
# 
# https://docs.databricks.com/spark/latest/gentle-introduction/gentle-intro.html
# 
# https://docs.databricks.com/spark/latest/gentle-introduction/gentle-intro.html#gentle-introduction-to-apache-spark
# 
# https://docs.databricks.com/spark/latest/gentle-introduction/for-data-scientists.html

# In[ ]:




