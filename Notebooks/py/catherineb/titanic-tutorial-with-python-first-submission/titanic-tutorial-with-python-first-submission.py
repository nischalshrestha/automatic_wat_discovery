#!/usr/bin/env python
# coding: utf-8

# # First submission
# 
# Outcome predicted based on gender
# 
# This follows the "Getting started with Python" tutorial:
# https://www.kaggle.com/c/titanic/details/getting-started-with-python

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import relevant packages

import csv as csv # for reading and writing csv files

# open the csv file in to a Python object
csv_file_object = csv.reader(open('../input/train.csv'))

header = next(csv_file_object) # next() command skips the first
                                # line which is a header
print(header)

data = []  # create a variable called 'data'

# run through aech row in the csv file, adding each row
# to the data variable
for row in csv_file_object:
    data.append(row)
    
# covert from a list to an array
# NB each item is currently a STRING
data = np.array(data)


# In[ ]:


print(data) # data is an array with just values (no header)
            # values stored as strings


# In[ ]:


print(data[0]) # see first row
print(data[-1]) # and last row
print(data[0,3]) # see 1st row, 4th column


# In[ ]:


# if want specific column, e.g. gender column:
# data[0::,4]
# will need to convert strings to floats to do calculations
# e.g Pclass into floats: data[0::,2].astype(np.float)

# The size() function counts how many elements are in
# the array and sum() sums the elements in the array

# calculate the proportion of survivors on the Titanic
number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

print(proportion_survivors)


# In[ ]:


# determine number of females and males that survived
women_only_stats = data[0::,4] == "female" # finds where all
                                           # elements of the gender
                                           # column that equal "female"
men_only_stats = data[0::,4] != "female" # finds where all the
                                         # elements do not equal
                                         # female (i.e. male)
# use these new variables as a "mask" on our original data
# to get stats on only men and only women


# In[ ]:


# Using the index from above select femails and males separately
women_onboard = data[women_only_stats,1].astype(np.float)
men_onboard = data[men_only_stats,1].astype(np.float)

# Find the proportion of women and proportion of men that survived
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)


# # reading the test data and writing the gender model as a csv

# In[ ]:


# read in the test.csv and skip the header line
test_file = open('../input/test.csv')
test_file_object = csv.reader(test_file)
header = next(test_file_object)


# In[ ]:


# open a pointer to a new file so we can write to it
# (the file does not exit yet)
prediction_file = open("first_genderbasedmodel.csv", "wt", newline='\n')
prediction_file_object = csv.writer(prediction_file)


# In[ ]:


# read the test file row by row
# see if male or femail and write survival prediction to a new file
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:
    if row[4] == 'female':
        prediction_file_object.writerow([row[0], '1']) # predict 1
    else:
        prediction_file_object.writerow([row[0], '0']) # predict 0
test_file.close()
prediction_file.close()


# # Second submission
# 
# Outcome predicted based on gender, class and fare

# In[ ]:


# idea is to create a table which contains just 1s and 0s.
# this will be a survivor reference table. 
# read test data, find passenger attributes, look them up in the table
# determine if survived or not
# array will be 2x3x4
# ([female/male],[1st/2nd/3rd class],[4 bins of prices])
# assume any fare >40 "equals" 39

# add a ceiling to fares
fare_ceiling = 40
# modify the data in the Fare column to =39 if >=ceiling
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

# know there were 1st, 2nd and 3rd classes
number_of_classes = 3

# better to calculate this from the data directly
# take the length of an array of unique values in column index 2
number_of_classes = len(np.unique(data[0::, 2]))

# initialise the survival table with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))


# In[ ]:


# loop through each variable and find those passengers that agree with the statements:
for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        
        women_only_stats = data[                               (data[0::,4] == "female")                               &(data[0::,2].astype(np.float)                                == i+1)                               &(data[0:,9].astype(np.float)                                >= j*fare_bracket_size)                               &(data[0:,9].astype(np.float)                                < (j+1)*fare_bracket_size)                               , 1]
        
        men_only_stats = data[                            (data[0::,4] != "female")                            &(data[0::,2].astype(np.float)                             == i+1)                            &(data[0:,9].astype(np.float)                             >= j*fare_bracket_size)                            &(data[0:,9].astype(np.float)                             < (j+1)*fare_bracket_size)                            , 1]
        
        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
        survival_table[ survival_table != survival_table ] = 0

print(survival_table)
# e.g. value of 0.914 signifies a 91.4% chance a passenger fitting those criteria survived


# In[ ]:


# assume for any probability >= 0.5 model predicts survival
# and probability < 0.5 predicts not
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1

print(survival_table)


# In[ ]:


# loop through test file, find what criteria passenger fits
# assign them a 1 or 0 according to our survival table
test_file = open('../input/test.csv')
test_file_object = csv.reader(test_file)
header = next(test_file_object)
predictions_file = open("second_genderclassmodel.csv", "wt")
p = csv.writer(predictions_file)
p.writerow(["PassengerId", "Survived"])


# In[ ]:


# fares column in test file not binned
# loop through each bin and see if the price of their ticket
# falls in that bin. if so, break loop and assign that bin
for row in test_file_object:
    for j in range(number_of_price_brackets):
        try: # some passengers have no fare data, so try 
            row[8] = float(row[8])  # to make a float
        except: # if fails, assign bin according to class
            bin_fare = 3 - float(row[1])
            break
        if row[8] > fare_ceiling: # if there is data, handle ceiling
            bin_fare = number_of_price_brackets - 1
            break
        if row[8] >= j*fare_bracket_size        and row[8] < (j+1)*fare_bracket_size:
            bin_fare = j
            break
    
    if row[3] == "female":
        p.writerow([row[0], int(survival_table[0, float(row[1])-1, bin_fare])])
    else:
        p.writerow([row[0], int(survival_table[1, float(row[1])-1, bin_fare])])
            
# close out the files
test_file.close()
predictions_file.close()


# In[ ]:




