#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="https://habrastorage.org/files/fd4/502/43d/fd450243dd604b81b9713213a247aa20.jpg">
#     
# ## [mlcourse.ai](https://mlcourse.ai) – Open Machine Learning Course 
# 
# Author: [Yury Kashnitskiy](https://yorko.github.io). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.

# [mlcourse.ai](https://mlcourse.ai) is an open Machine Learning course by OpenDataScience (a.ka. [ods.ai](http://ods.ai)). It's distinctive features are:
#  - Perfect balance between theory and practice
#  - Quick dive into Kaggle competitions (some assignments require beating baselines)
#  - Highly motivating rating held during the course
#  - Materials as [Kaggle Kernels](https://www.kaggle.com/kashnitsky/mlcourse/kernels)
#  - And much more, check the course [roadmap](https://mlcourse.ai/roadmap) 
#  
# Here we present a simplified version of an assignment on Pandas and preliminary data analysis. Prior to completing the assignment, you might address the 1st part of the course material ["Topic 1. Exploratory Data Analysis with Pandas"](https://www.kaggle.com/kashnitsky/topic-1-exploratory-data-analysis-with-pandas) and/or watch the [1st lecture](https://www.youtube.com/watch?v=fwWCw_cE5aI). Solution to this practice can be found in mlcourse.ai  [Kernels](https://www.kaggle.com/kashnitsky/mlcourse/kernels).
#  
#  PS. If it looks too simple for you, take a look at course [assignments](https://mlcourse.ai/assignments).

# # <center> Topic 1. Exploratory data analysis with Pandas
# <img align="center" src="https://habrastorage.org/files/10c/15f/f3d/10c15ff3dcb14abdbabdac53fed6d825.jpg" width=50% />
# ## <center>Practice. Analyzing "Titanic" passengers
# 
# **Fill in the missing code ("You code here") and choose answers in a [web-form](https://docs.google.com/forms/d/16EfhpDGPrREry0gfDQdRPjoiQX9IumaL2mPR0rcj19k/edit).**

# In[ ]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
pd.set_option("display.precision", 2)


# [](http://)**Read data into a Pandas DataFrame**

# In[ ]:


data = pd.read_csv('../input/train.csv', index_col='PassengerId')


# **First 5 rows**

# In[ ]:


data.head(5)


# In[ ]:


data.describe()


# **Let's select those passengers who embarked in Cherbourg (Embarked=C) and paid > 200 pounds for their ticker (fare > 200).**
# 
# Make sure you understand how actually this construction works.

# In[ ]:


data[(data['Embarked'] == 'C') & (data.Fare > 200)].head()


# **We can sort these people by Fare in descending order.**

# In[ ]:


data[(data['Embarked'] == 'C') & 
     (data['Fare'] > 200)].sort_values(by='Fare',
                               ascending=False).head()


# **Let's create a new feature.**

# In[ ]:


def age_category(age):
    '''
    < 30 -> 1
    >= 30, <55 -> 2
    >= 55 -> 3
    '''
    if age < 30:
        return 1
    elif age < 55:
        return 2
    else:
        return 3


# In[ ]:


age_categories = [age_category(age) for age in data.Age]
data['Age_category'] = age_categories


# **Another way is to do it with `apply`.**

# In[ ]:


data['Age_category'] = data['Age'].apply(age_category)


# **1. How many men/women were there onboard?**
# - 412 men and 479 women
# - 314 men и 577 women
# - 479 men и 412 women
# - 577 men и 314 women

# In[ ]:


# You code here


# **2. Print the distribution of the `Pclass` feature. Then the same, but for men and women separately. How many men from second class were there onboard?**
# - 104
# - 108
# - 112
# - 125

# In[ ]:


# You code here


# **3. What are median and standard deviation of `Fare`?. Round to two decimals.**
# - median is  14.45, standard deviation is 49.69
# - median is 15.1, standard deviation is 12.15
# - median is 13.15, standard deviation is 35.3
# - median is  17.43, standard deviation is 39.1

# In[ ]:


# You code here


# **4. Is that true that the mean age of survived people is higher than that of passengers who eventually died?**
# - Yes
# - No
# 

# In[ ]:


# You code here


# **5. Is that true that passengers younger than 30 y.o. survived more frequently than those older than 60 y.o.? What are shares of survived people among young and old people?**
# - 22.7% among young and 40.6% among old
# - 40.6% among young and 22.7% among old
# - 35.3% among young and 27.4% among old
# - 27.4% among young and  35.3% among old

# In[ ]:


# You code here


# **6. Is that true that women survived more frequently than men? What are shares of survived people among men and women?**
# - 30.2% among men and 46.2% among women
# - 35.7% among men and 74.2% among women
# - 21.1% among men and 46.2% among women
# - 18.9% among men and 74.2% among women

# In[ ]:


# You code here


# **7. What's the most popular first name among male passengers?**
# - Charles
# - Thomas
# - William
# - John

# In[ ]:


# You code here


# **8. How is average age for men/women dependent on `Pclass`? Choose all correct statements:**
# - On average, men of 1 class are older than 40
# - On average, women of 1 class are older than 40
# - Men of all classes are on average older than women of the same class
# - On average, passengers ofthe first class are older than those of the 2nd class who are older than passengers of the 3rd class

# In[ ]:


# You code here

