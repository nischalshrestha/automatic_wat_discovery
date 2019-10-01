#!/usr/bin/env python
# coding: utf-8

#  # <div align="center">  The Data Scientist’s Toolbox Tutorial - 1 </div>
#  ### <div align="center"><b>Quite Practical and Far from any Theoretical Concepts</b></div>
#  
# <div style="text-align:center">last update: <b>11/19/2018</b></div>

# you may be interested have a look at it: 
# >[**The Data Scientist’s Toolbox Tutorial - 2**](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)
# 
# ---------------------------------------------------------------------
# Fork, Run and Follow this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red">UPVOTES</font> would be very much appreciated**
#  
#  -----------
# 

#  <a id="top"></a> <br>
# **Notebook Content**
# 1. [Introduction](#1)
#     1. [Import](#2)
#     1. [Version](#3)
#     1. [setup](#4)
# 1. [Python](#5)
#     1. [Python Syntax compared to other programming languages](#6)
#     1. [Basics](#2)
#     1. [Functions](#3)
#     1. [Types and Sequences](#4)
#     1. [More on Strings](#5)
#     1. [Reading and Writing CSV files](#6)
#     1. [Dates and Times](#7)
#     1. [Objects and map()](#8)
#     1. [Lambda and List Comprehensions](#9)
#     1. [OOP](#10)
# 1. [Numpy](#12)
#     1. [Creating Arrays](#13)
#     1. [Combining Arrays](#14)
#     1. [Operations](#15)
#     1. [Math Functions](#16)
#     1. [Indexing / Slicing](#17)
#     1. [Copying Data](#18)
#     1. [Iterating Over Arrays](#19)
#     1. [The Series Data Structure](#20)
#     1. [Querying a Series](#21)
# 1. [Pandas](#22)
#     1. [The DataFrame Data Structure](#22)
#     1. [Dataframe Indexing and Loading](#23)
#     1. [Missing values](#24)
#     1. [Merging Dataframes](#25)
#     1. [Making Code Pandorable](#26)
#     1. [Group by](#27)
#     1. [Scales](#28)
#     1. [Pivot Tables](#29)
#     1. [Date Functionality](#30)
#     1. [Distributions in Pandas](#31)
#     1. [Hypothesis Testing](#32)
# 1. [Matplotlib](#33)
#     1. [Scatterplots](#34)
#     1. [Line Plots](#35)
#     1. [Bar Charts](#36)
#     1. [Histograms](#37)
#     1. [Box and Whisker Plots](#38)
#     1. [Heatmaps](#39)
#     1. [Animations](#40)
#     1. [Interactivity](#41)
#     1. [DataFrame.plot](#42)
# 1. [seaborn](#43)
#     1. [Seaborn Vs Matplotlib](#43)
#     1. [Useful Python Data Visualization Libraries](#43)
# 1. [CheatSheet](#60)
# 1. [Conclusion](#61)
# 1. [References](#62)

# <a id="1"></a> <br>
# ## 1-Introduction

# In this kernel, we have a comprehensive tutorials for some important packages in python after that you can read my other kernels about machine learning.

#  <img src="http://s8.picofile.com/file/8340141626/packages1.png"  height="420" width="420">
#  
#  ###### [Go to top](#top)

# <a id="2"></a> <br>
# ## 1-1 Import

# In[ ]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import scipy.stats as stats
import plotly.plotly as py
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import warnings
import sklearn
import scipy
import json
import sys
import csv
import os


# <a id="3"></a> <br>
# ## 1-2 Version

# In[ ]:


print('matplotlib: {}'.format(matplotlib.__version__))
print('sklearn: {}'.format(sklearn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('seaborn: {}'.format(sns.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))


# <a id="4"></a> <br>
# ##  1-3 Setup

# In[ ]:


warnings.filterwarnings('ignore')
sns.set(color_codes=True)
plt.style.available
get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'precision 2')


#  <a id="5"></a> <br>
# # 2-Python
# 
# **Python** is a modern, robust, high level programming language. It is very easy to pick up even if you are completely new to programming.
# 
# It is used for:
# 
# 1. web development (server-side),
# 1. software development,
# 1. mathematics,
# 1. system scripting.
# 
# <a id="6"></a> <br>
# ## 2-1 Python Syntax compared to other programming languages
# 
# 1. Python was designed to for readability, and has some similarities to the English language with influence from mathematics.
# 1. Python uses new lines to complete a command, as opposed to other programming languages which often use semicolons or parentheses.
# 1. Python relies on indentation, using whitespace, to define scope; such as the scope of loops, functions and classes. Other programming languages often use curly-brackets for this purpose.
# ###### [Go to top](#top)

# <a id="7"></a> <br>
# # 2-2 Python: Basics
# 

# In[ ]:


import this


# <a id="8"></a> <br>
# ### 2-2-1 Variables
# A name that is used to denote something or a value is called a variable. In python, variables can be declared and values can be assigned to it as follows,

# In[ ]:


x = 2
y = 5
xy = 'Hey'


# <a id="9"></a> <br>
# ### 2-2-2 Operators

# | Symbol | Task Performed |
# |----|---|
# | +  | Addition |
# | -  | Subtraction |
# | /  | division |
# | %  | mod |
# | *  | multiplication |
# | //  | floor division |
# | **  | to the power of |
# 
# ### Relational Operators
# | Symbol | Task Performed |
# |----|---|
# | == | True, if it is equal |
# | !=  | True, if not equal to |
# | < | less than |
# | > | greater than |
# | <=  | less than or equal to |
# | >=  | greater than or equal to |
# ### Bitwise Operators
# | Symbol | Task Performed |
# |----|---|
# | &  | Logical And |
# | l  | Logical OR |
# | ^  | XOR |
# | ~  | Negate |
# | >>  | Right shift |
# | <<  | Left shift |
# 
# ###### [Go to top](#top)

# <a id="10"></a> <br>
# # 2-3 Python : Functions

# <br>
# `add_numbers` is a function that takes two numbers and adds them together.

# In[ ]:


def add_numbers(x, y):
    return x + y

add_numbers(1, 2)


# <br>
# `add_numbers` updated to take an optional 3rd parameter. Using `print` allows printing of multiple expressions within a single cell.

# In[ ]:


def add_numbers(x,y,z=None):
    if (z==None):
        return x+y
    else:
        return x+y+z

print(add_numbers(1, 2))
print(add_numbers(1, 2, 3))


# <br>
# `add_numbers` updated to take an optional flag parameter.

# In[ ]:


def add_numbers(x, y, z=None, flag=False):
    if (flag):
        print('Flag is true!')
    if (z==None):
        return x + y
    else:
        return x + y + z
    
print(add_numbers(1, 2, flag=True))


# <br>
# Assign function `add_numbers` to variable `a`.

# In[ ]:


def add_numbers(x,y):
    return x+y

a = add_numbers
a(1,2)


# <a id="11"></a> <br>
# # 2-4 Python : Types and Sequences

# <br>
# Use `type` to return the object's type.

# In[ ]:


type('This is a string')


# In[ ]:


type(None)


# In[ ]:


type(1)


# In[ ]:


type(1.0)


# In[ ]:


type(add_numbers)


# <br>
# Tuples are an immutable data structure (cannot be altered).

# In[ ]:


x = (1, 'a', 2, 'b')
type(x)


# <br>
# Lists are a mutable data structure.

# In[ ]:


x = [1, 'a', 2, 'b']
type(x)


# <br>
# Use `append` to append an object to a list.

# In[ ]:


x.append(3.3)
print(x)


# <br>
# This is an example of how to loop through each item in the list.

# In[ ]:


for item in x:
    print(item)


# <br>
# Or using the indexing operator:

# In[ ]:


i=0
while( i != len(x) ):
    print(x[i])
    i = i + 1


# <br>
# Use `+` to concatenate lists.

# In[ ]:


[1,2] + [3,4]


# <br>
# Use `*` to repeat lists.

# In[ ]:


[1]*3


# <br>
# Use the `in` operator to check if something is inside a list.

# In[ ]:


1 in [1, 2, 3]


# <br>
# Now let's look at strings. Use bracket notation to slice a string.
# 
# ###### [Go to top](#top)

# In[ ]:


x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters


# <br>
# This will return the last element of the string.

# In[ ]:


x[-1]


# <br>
# This will return the slice starting from the 4th element from the end and stopping before the 2nd element from the end.

# In[ ]:


x[-4:-2]


# <br>
# This is a slice from the beginning of the string and stopping before the 3rd element.

# In[ ]:


x[:3]


# <br>
# And this is a slice starting from the 3rd element of the string and going all the way to the end.

# In[ ]:


x[3:]


# In[ ]:


firstname = 'MJ'
lastname = 'Bahmani'

print(firstname + ' ' + lastname)
print(firstname*3)
print('mj' in firstname)


# <br>
# `split` returns a list of all the words in a string, or a list split on a specific character.

# In[ ]:


firstname = 'Mr Dr Mj Bahmani'.split(' ')[0] # [0] selects the first element of the list
lastname = 'Mr Dr Mj Bahmani'.split(' ')[-1] # [-1] selects the last element of the list
print(firstname)
print(lastname)


# In[ ]:


'MJ' + str(2)


# <br>
# Dictionaries associate keys with values.

# In[ ]:


x = {'MJ Bahmani': 'Mohamadjavad.bahmani@gmail.com', 'irmatlab': 'irmatlab.ir@gmail.com'}
x['MJ Bahmani'] # Retrieve a value by using the indexing operator


# In[ ]:


x['MJ Bahmani'] = None
x['MJ Bahmani']


# <br>
# Iterate over all of the keys:

# In[ ]:


for name in x:
    print(x[name])


# <br>
# Iterate over all of the values:

# In[ ]:


for email in x.values():
    print(email)


# <br>
# Iterate over all of the items in the list:

# In[ ]:


for name, email in x.items():
    print(name)
    print(email)


# <br>
# You can unpack a sequence into different variables:

# In[ ]:


x = ('MJ', 'Bahmani', 'Mohamadjavad.bahmani@gmail.com')
fname, lname, email = x


# In[ ]:


fname


# In[ ]:


lname


# <a id="5"></a> <br>
# # 2-5 Python: More on Strings
# 
# ###### [Go to top](#top)

# In[ ]:


print('MJ' + str(2))


# <br>
# Python has a built in method for convenient string formatting.

# In[ ]:


sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'MJ'}

sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))


# <a id="6"></a> <br>
# # 2-6 Python:Reading and Writing CSV files

# <br>
# Let's import our datafile train.csv 
# ###### [Go to top](#top)

# In[ ]:




with open('../input/train.csv') as csvfile:
    train = list(csv.DictReader(csvfile))
    
train[:1] # The first three dictionaries in our list.


# <br>
# `csv.Dictreader` has read in each row of our csv file as a dictionary. `len` shows that our list is comprised of 234 dictionaries.

# In[ ]:


len(train)


# <br>
# `keys` gives us the column names of our csv.

# In[ ]:


train[0].keys()


# <br>
# How to do some math action on the data set

# In[ ]:


sum(float(d['Fare']) for d in train) / len(train)


# <br>
# Use `set` to return the unique values for the type of Sex  in our dataset have.

# In[ ]:


Sex = set(d['Sex'] for d in train)
Sex


# <a id="7"></a> <br>
# # 2-7 Python: Dates and Times

# In[ ]:


import datetime as dt
import time as tm


# <br>
# `time` returns the current time in seconds since the Epoch. (January 1st, 1970)
# 
# ###### [Go to top](#top)

# In[ ]:


tm.time()


# <br>
# Convert the timestamp to datetime.

# In[ ]:


dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow


# <br>
# Handy datetime attributes:

# In[ ]:


dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime


# <br>
# `timedelta` is a duration expressing the difference between two dates.

# In[ ]:


delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta


# <br>
# `date.today` returns the current local date.

# In[ ]:


today = dt.date.today()


# In[ ]:


today - delta # the date 100 days ago


# In[ ]:


today > today-delta # compare dates


# <a id="8"></a> <br>
# # 2-8 Python: Objects and map()

# <br>
# An example of a class in python:

# In[ ]:


class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location


# In[ ]:


person = Person()
person.set_name('MJ Bahmani')
person.set_location('MI, Berlin, Germany')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))


# <br>
# Here's an example of mapping the `min` function between two lists.

# In[ ]:


store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest = map(min, store1, store2)
cheapest


# <br>
# Now let's iterate through the map object to see the values.

# In[ ]:


for item in cheapest:
    print(item)


# <a id="9"></a> <br>
# # 2-9-Python : Lambda and List Comprehensions

# <br>
# Here's an example of lambda that takes in three parameters and adds the first two.

# In[ ]:


my_function = lambda a, b, c : a + b


# In[ ]:


my_function(1, 2, 3)


# <br>
# Let's iterate from 0 to 9 and return the even numbers.
# 
# ###### [Go to top](#top)

# In[ ]:


my_list = []
for number in range(0, 9):
    if number % 2 == 0:
        my_list.append(number)
my_list


# <br>
# Now the same thing but with list comprehension.

# In[ ]:


my_list = [number for number in range(0,10) if number % 2 == 0]
my_list


# <a id="10"></a> <br>
# # 2-10 OOP
# 1. **Class** − A user-defined prototype for an object that defines a set of attributes that characterize any object of the class. The attributes are data members (class variables and instance variables) and methods, accessed via dot notation.
# 
# 1. **Class variable** − A variable that is shared by all instances of a class. Class variables are defined within a class but outside any of the class's methods. Class variables are not used as frequently as instance variables are.
# 
# 1. **Data member** − A class variable or instance variable that holds data associated with a class and its objects.
# 
# 1. **Function overloading** − The assignment of more than one behavior to a particular function. The operation performed varies by the types of objects or arguments involved.
# 
# 1. **Instance variable** − A variable that is defined inside a method and belongs only to the current instance of a class.
# 
# 1. **Inheritance** − The transfer of the characteristics of a class to other classes that are derived from it.
# 
# 1. **Instance** − An individual object of a certain class. An object obj that belongs to a class Circle, for example, is an instance of the class Circle.
# 
# 1. **Instantiation** − The creation of an instance of a class.
# 
# 1. **Method** − A special kind of function that is defined in a class definition.
# 
# 1. **Object** − A unique instance of a data structure that's defined by its class. An object comprises both data members (class variables and instance variables) and methods.
# 
# 1. **Operator overloading** − The assignment of more than one function to a particular operator.[4]
# 
# ###### [Go to top](#top)

# In[ ]:


class FirstClass:
    test = 'test'
    def __init__(self,name,symbol):
        self.name = name
        self.symbol = symbol


# In[ ]:


eg3 = FirstClass('Three',3)


# In[ ]:


print (eg3.test, eg3.name)


# In[ ]:


class FirstClass:
    def __init__(self,name,symbol):
        self.name = name
        self.symbol = symbol
    def square(self):
        return self.symbol * self.symbol
    def cube(self):
        return self.symbol * self.symbol * self.symbol
    def multiply(self, x):
        return self.symbol * x


# In[ ]:


eg4 = FirstClass('Five',5)


# In[ ]:


print (eg4.square())
print (eg4.cube())


# In[ ]:


eg4.multiply(2)


# In[ ]:


FirstClass.multiply(eg4,2)


# <a id="11"></a> <br>
# ### 2-10-1 Inheritance
# 
# There might be cases where a new class would have all the previous characteristics of an already defined class. So the new class can "inherit" the previous class and add it's own methods to it. This is called as inheritance.
# 
# Consider class SoftwareEngineer which has a method salary.

# In[ ]:


class SoftwareEngineer:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def salary(self, value):
        self.money = value
        print (self.name,"earns",self.money)


# In[ ]:


a = SoftwareEngineer('Kartik',26)


# In[ ]:


a.salary(40000)


# In[ ]:


dir(SoftwareEngineer)


# In[ ]:


class Artist:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def money(self,value):
        self.money = value
        print (self.name,"earns",self.money)
    def artform(self, job):
        self.job = job
        print (self.name,"is a", self.job)


# In[ ]:


b = Artist('Nitin',20)


# In[ ]:


b.money(50000)
b.artform('Musician')


# In[ ]:


dir(Artist)


# <a id="12"></a> <br>
# ## 2-11 Python JSON
# 

# In[ ]:




# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"])


# <a id="13"></a> <br>
# ## 2-11-1 Convert from Python to JSON
# 

# In[ ]:


# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)


# You can convert Python objects of the following types, into JSON strings:
# 
# * dict
# * list
# * tuple
# * string
# * int
# * float
# * True
# * False
# * None
# 
# ###### [Go to top](#top)

# In[ ]:


print(json.dumps({"name": "John", "age": 30}))
print(json.dumps(["apple", "bananas"]))
print(json.dumps(("apple", "bananas")))
print(json.dumps("hello"))
print(json.dumps(42))
print(json.dumps(31.76))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None))


# Convert a Python object containing all the legal data types:

# In[ ]:


x = {
  "name": "John",
  "age": 30,
  "married": True,
  "divorced": False,
  "children": ("Ann","Billy"),
  "pets": None,
  "cars": [
    {"model": "BMW 230", "mpg": 27.5},
    {"model": "Ford Edge", "mpg": 24.1}
  ]
}

print(json.dumps(x))


# <a id="14"></a> <br>
# ## 2-12 Python PIP
# 

# <a id="15"></a> <br>
# ### 2-12-1 What is a Package?
# A package contains all the files you need for a module.
# 
# Modules are Python code libraries you can include in your project.
# 
# ###### [Go to top](#top)

# <a id="16"></a> <br>
# ### 2-12-2 Install PIP
# If you do not have PIP installed, you can download and install it from this page: https://pypi.org/project/pip/

# <a id="17"></a> <br>
# ## 2-13 Python Try Except
# The **try** block lets you test a block of code for errors.
# 
# The **except** block lets you handle the error.
# 
# The **finally** block lets you execute code, regardless of the result of the try- and except blocks.

# In[ ]:


try:
  print(x)
except NameError:
  print("Variable x is not defined")
except:
  print("Something else went wrong")


# In[ ]:


try:
  print(x)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")


# <a id="18"></a> <br>
# ## 2-14 Python Iterators
# An iterator is an object that contains a countable number of values.
# 
# An iterator is an object that can be iterated upon, meaning that you can traverse through all the values.
# 
# Technically, in Python, an iterator is an object which implements the iterator protocol, which consist of the methods __iter__() and __next__().
# ###### [Go to top](#top)

# Return a iterator from a tuple, and print each value:

# In[ ]:


mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))


# <a id="19"></a> <br>
# ### 2- 14-1 Looping Through an Iterator
# 

# In[ ]:


mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)


# <a id="20"></a> <br>
# ## 2- 15 Dictionary
# A **dictionary** is a collection which is **unordered, changeable and indexed**. In Python dictionaries are written with curly brackets, and they have **keys and values**.

# In[ ]:


thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)


# <a id="21"></a> <br>
# ## 2-16 Tuples
# A **tuple** is a collection which is **ordered and unchangeable**. In Python tuples are written with round brackets.
# 
# 

# In[ ]:


thistuple = ("apple", "banana", "cherry")
print(thistuple)


# <a id="22"></a> <br>
# ## 2-19 Set
# A set is a collection which is unordered and unindexed. In Python sets are written with curly brackets.
# ###### [Go to top](#top)

# In[ ]:


thisset = {"apple", "banana", "cherry"}
print(thisset)


# In[ ]:


thisset = {"apple", "banana", "cherry"}

for x in thisset:
  print(x)


# <a id="23"></a> <br>
# ### 2-17-1 Add Items
# To add one item to a set use the add() method.
# 
# To add more than one item to a set use the update() method.
# ###### [Go to top](#top)

# In[ ]:


thisset = {"apple", "banana", "cherry"}

thisset.add("orange")

print(thisset)


# <a id="11"></a> <br>
# # Python Packages
# * Numpy
# * Pandas
# * Matplotlib
# * Seaborn
# * Sklearn
# * plotly

# <a id="24"></a> <br>
# # 3- Numerical Python (NumPy)

# In[ ]:


import numpy as np


# <a id="25"></a> <br>
# ## 3-1 NumPy :Creating Arrays

# Create a list and convert it to a numpy array

# In[ ]:


mylist = [1, 2, 3]
x = np.array(mylist)
x


# <br>
# Or just pass in a list directly

# In[ ]:


y = np.array([4, 5, 6])
y


# <br>
# Pass in a list of lists to create a multidimensional array.

# In[ ]:


m = np.array([[7, 8, 9], [10, 11, 12]])
m


# <br>
# Use the shape method to find the dimensions of the array. (rows, columns).

# In[ ]:


m.shape


# <br>
# `arange` returns evenly spaced values within a given interval.

# In[ ]:


n = np.arange(0, 30, 2) # start at 0 count up by 2, stop before 30
n


# <br>
# `reshape` returns an array with the same data with a new shape.

# In[ ]:


n = n.reshape(3, 5) # reshape array to be 3x5
n


# <br>
# `linspace` returns evenly spaced numbers over a specified interval.

# In[ ]:


o = np.linspace(0, 4, 9) # return 9 evenly spaced values from 0 to 4
o


# <br>
# `resize` changes the shape and size of array in-place.

# In[ ]:


o.resize(3, 3)
o


# <br>
# `ones` returns a new array of given shape and type, filled with ones.

# In[ ]:


np.ones((3, 2))


# <br>
# `zeros` returns a new array of given shape and type, filled with zeros.

# In[ ]:


np.zeros((2, 3))


# <br>
# `eye` returns a 2-D array with ones on the diagonal and zeros elsewhere.

# In[ ]:


np.eye(3)


# <br>
# `diag` extracts a diagonal or constructs a diagonal array.

# In[ ]:


np.diag(y)


# <br>
# Create an array using repeating list (or see `np.tile`)

# In[ ]:


np.array([1, 2, 3] * 3)


# <br>
# Repeat elements of an array using `repeat`.

# In[ ]:


np.repeat([1, 2, 3], 3)


# <a id="26"></a> <br>
# ## 3-2 Numpy:Combining Arrays
# 
# ###### [Go to top](#top)

# In[ ]:


p = np.ones([2, 3], int)
p


# <br>
# Use `vstack` to stack arrays in sequence vertically (row wise).

# In[ ]:


np.vstack([p, 2*p])


# <br>
# Use `hstack` to stack arrays in sequence horizontally (column wise).

# In[ ]:


np.hstack([p, 2*p])


# <a id="27"></a> <br>
# ## 3-3 Numpy:Operations
# 
# ###### [Go to top](#top)

# Use `+`, `-`, `*`, `/` and `**` to perform element wise addition, subtraction, multiplication, division and power.

# In[ ]:


print(x + y) # elementwise addition     [1 2 3] + [4 5 6] = [5  7  9]
print(x - y) # elementwise subtraction  [1 2 3] - [4 5 6] = [-3 -3 -3]


# In[ ]:


print(x * y) # elementwise multiplication  [1 2 3] * [4 5 6] = [4  10  18]
print(x / y) # elementwise divison         [1 2 3] / [4 5 6] = [0.25  0.4  0.5]


# In[ ]:


print(x**2) # elementwise power  [1 2 3] ^2 =  [1 4 9]


# <br>
# **Dot Product:**  
# 
# $ \begin{bmatrix}x_1 \ x_2 \ x_3\end{bmatrix}
# \cdot
# \begin{bmatrix}y_1 \\ y_2 \\ y_3\end{bmatrix}
# = x_1 y_1 + x_2 y_2 + x_3 y_3$

# In[ ]:


x.dot(y) # dot product  1*4 + 2*5 + 3*6


# In[ ]:


z = np.array([y, y**2])
print(len(z)) # number of rows of array


# <br>
# Let's look at transposing arrays. Transposing permutes the dimensions of the array.

# In[ ]:


z = np.array([y, y**2])
z


# <br>
# The shape of array `z` is `(2,3)` before transposing.

# In[ ]:


z.shape


# <br>
# Use `.T` to get the transpose.

# In[ ]:


z.T


# <br>
# The number of rows has swapped with the number of columns.

# In[ ]:


z.T.shape


# <br>
# Use `.dtype` to see the data type of the elements in the array.

# In[ ]:


z.dtype


# <br>
# Use `.astype` to cast to a specific type.

# In[ ]:


z = z.astype('f')
z.dtype


# <a id="28"></a> <br>
# ## 3-4 Numpy: Math Functions
# 
# ###### [Go to top](#top)

# Numpy has many built in math functions that can be performed on arrays.

# In[ ]:


a = np.array([-4, -2, 1, 3, 5])


# In[ ]:


a.sum()


# In[ ]:


a.max()


# In[ ]:


a.min()


# In[ ]:


a.mean()


# In[ ]:


a.std()


# <br>
# `argmax` and `argmin` return the index of the maximum and minimum values in the array.

# In[ ]:


a.argmax()


# In[ ]:


a.argmin()


# <a id="29"></a> <br>
# 
# ## 3-5 Numpy:Indexing / Slicing
# ###### [Go to top](#top)

# In[ ]:


s = np.arange(13)**2
s


# <br>
# Use bracket notation to get the value at a specific index. Remember that indexing starts at 0.

# In[ ]:


s[0], s[4], s[-1]


# <br>
# Use `:` to indicate a range. `array[start:stop]`
# 
# 
# Leaving `start` or `stop` empty will default to the beginning/end of the array.

# In[ ]:


s[1:5]


# <br>
# Use negatives to count from the back.

# In[ ]:


s[-4:]


# <br>
# A second `:` can be used to indicate step-size. `array[start:stop:stepsize]`
# 
# Here we are starting 5th element from the end, and counting backwards by 2 until the beginning of the array is reached.

# In[ ]:


s[-5::-2]


# <br>
# Let's look at a multidimensional array.

# In[ ]:


r = np.arange(36)
r.resize((6, 6))
r


# <br>
# Use bracket notation to slice: `array[row, column]`.

# In[ ]:


r[2, 2]


# <br>
# And use : to select a range of rows or columns.

# In[ ]:


r[3, 3:6]


# <br>
# Here we are selecting all the rows up to (and not including) row 2, and all the columns up to (and not including) the last column.

# In[ ]:


r[:2, :-1]


# <br>
# This is a slice of the last row, and only every other element.

# In[ ]:


r[-1, ::2]


# <br>
# We can also perform conditional indexing. Here we are selecting values from the array that are greater than 30. (Also see `np.where`)

# In[ ]:


r[r > 30]


# <br>
# Here we are assigning all values in the array that are greater than 30 to the value of 30.
# ###### [Go to top](#top)

# In[ ]:


r[r > 30] = 30
r


# <a id="30"></a> <br>
# ## 3-6 Numpy :Copying Data

# Be careful with copying and modifying arrays in NumPy!
# 
# 
# `r2` is a slice of `r`

# In[ ]:


r2 = r[:3,:3]
r2


# <br>
# Set this slice's values to zero ([:] selects the entire array)

# In[ ]:


r2[:] = 0
r2


# <br>
# `r` has also been changed!

# In[ ]:


r


# <br>
# To avoid this, use `r.copy` to create a copy that will not affect the original array

# In[ ]:


r_copy = r.copy()
r_copy


# <br>
# Now when r_copy is modified, r will not be changed.

# In[ ]:


r_copy[:] = 10
print(r_copy, '\n')
print(r)


# <a id="31"></a> <br>
# ## 3-7 Numpy: Iterating Over Arrays

# Let's create a new 4 by 3 array of random numbers 0-9.

# In[ ]:


test = np.random.randint(0, 10, (4,3))
test


# <br>
# Iterate by row:

# In[ ]:


for row in test:
    print(row)


# <br>
# Iterate by index:

# In[ ]:


for i in range(len(test)):
    print(test[i])


# <br>
# Iterate by row and index:

# In[ ]:


for i, row in enumerate(test):
    print('row', i, 'is', row)


# <br>
# Use `zip` to iterate over multiple iterables.

# In[ ]:


test2 = test**2
test2


# In[ ]:


for i, j in zip(test, test2):
    print(i,'+',j,'=',i+j)


# <a id="32"></a> <br>
# ## 3-8  Numpy: The Series Data Structure
# One-dimensional ndarray with axis labels (including time series)

# In[ ]:


animals = ['Tiger', 'Bear', 'Moose']
pd.Series(animals)


# In[ ]:


numbers = [1, 2, 3]
pd.Series(numbers)


# In[ ]:


animals = ['Tiger', 'Bear', None]
pd.Series(animals)


# In[ ]:


numbers = [1, 2, None]
pd.Series(numbers)


# In[ ]:


import numpy as np
np.nan == None


# In[ ]:


np.nan == np.nan


# In[ ]:


np.isnan(np.nan)


# In[ ]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[ ]:


s.index


# In[ ]:


s = pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s


# In[ ]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports, index=['Golf', 'Sumo', 'Hockey'])
s


# <a id="33"></a> <br>
# # 3-9 Numpy: Querying a Series

# In[ ]:


sports = {'Archery': 'Bhutan',
          'Golf': 'Scotland',
          'Sumo': 'Japan',
          'Taekwondo': 'South Korea'}
s = pd.Series(sports)
s


# In[ ]:


s.iloc[3]


# In[ ]:


s.loc['Golf']


# In[ ]:


s[3]


# In[ ]:


s['Golf']


# In[ ]:


sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)


# In[ ]:


s = pd.Series([100.00, 120.00, 101.00, 3.00])
s


# In[ ]:


total = 0
for item in s:
    total+=item
print(total)


# In[ ]:


total = np.sum(s)
print(total)


# In[ ]:


#this creates a big series of random numbers
s = pd.Series(np.random.randint(0,1000,10000))
s.head()


# In[ ]:


len(s)


# In[ ]:


get_ipython().run_cell_magic(u'timeit', u'-n 100', u'summary = 0\nfor item in s:\n    summary+=item')


# In[ ]:


get_ipython().run_cell_magic(u'timeit', u'-n 100', u'summary = np.sum(s)')


# In[ ]:


s+=2 #adds two to each item in s using broadcasting
s.head()


# In[ ]:


for label, value in s.iteritems():
    s.set_value(label, value+2)
s.head()


# In[ ]:


get_ipython().run_cell_magic(u'timeit', u'-n 10', u's = pd.Series(np.random.randint(0,1000,100))\nfor label, value in s.iteritems():\n    s.loc[label]= value+2')


# In[ ]:


get_ipython().run_cell_magic(u'timeit', u'-n 10', u's = pd.Series(np.random.randint(0,1000,100))\ns+=2')


# In[ ]:


s = pd.Series([1, 2, 3])
s.loc['Animal'] = 'Bears'
s


# In[ ]:


original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)


# In[ ]:


original_sports


# In[ ]:


cricket_loving_countries


# In[ ]:


all_countries


# In[ ]:


all_countries.loc['Cricket']


# <a id="22"></a> <br>
# ## 4- Pandas:The DataFrame Data Structure
#  You'll hone your pandas skills by learning how to organize, reshape, and aggregate multiple data sets to answer your specific questions.
#  **Pandas**:
# Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure.
# 
# Pandas is capable of many tasks including:
# 
# Reading/writing many different data formats
# Selecting subsets of data
# Calculating across rows and down columns
# Finding and filling missing data
# Applying operations to independent groups within the data
# Reshaping data into different forms
# Combing multiple datasets together
# Advanced time-series functionality
# Visualization through matplotlib and seaborn
# Although pandas is very capable, it does not provide functionality for the entire data science pipeline. Pandas is typically the intermediate tool used for data exploration and cleaning squashed between data capturing and storage, and data modeling and predicting.
# ###### [Go to top](#top)

# In[ ]:



purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
df.head()


# In[ ]:


df.loc['Store 2']


# In[ ]:


type(df.loc['Store 2'])


# In[ ]:


df.loc['Store 1']


# In[ ]:


df.loc['Store 1', 'Cost']


# In[ ]:


df.T


# In[ ]:


df.T.loc['Cost']


# In[ ]:


df['Cost']


# In[ ]:


df.loc['Store 1']['Cost']


# In[ ]:


df.loc[:,['Name', 'Cost']]


# In[ ]:


df.drop('Store 1')


# In[ ]:


df


# In[ ]:


copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
copy_df


# In[ ]:


copy_df.drop


# In[ ]:


del copy_df['Name']
copy_df


# In[ ]:


df['Location'] = None
df


# In[ ]:


costs = df['Cost']
costs


# In[ ]:


costs+=2
costs


# In[ ]:


df


# <a id="23"></a> <br>
# # 4-1 Pandas:Dataframe Indexing and Loading
# 
# As a Data Scientist, you'll often find that the data you need is not in a single file. It may be spread across a number of text files, spreadsheets, or databases. You want to be able to import the data of interest as a collection of DataFrames and figure out how to combine them to answer your central questions.
# ###### [Go to top](#top)

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df.columns


# In[ ]:


# Querying a DataFrame


# In[ ]:


df['Survived'] > 0


# In[ ]:


only_Survived = df.where(df['Survived'] > 0)
only_Survived.head()


# In[ ]:


only_Survived['Survived'].count()


# In[ ]:


df['Survived'].count()


# In[ ]:


only_Survived = only_Survived.dropna()
only_Survived.head()


# In[ ]:


only_Survived = df[df['Survived'] > 0]
only_Survived.head()


# In[ ]:


len(df[(df['Survived'] > 0) | (df['Survived'] > 0)])


# In[ ]:


df[(df['Survived'] > 0) & (df['Survived'] == 0)]


# In[ ]:


# Indexing Dataframes


# In[ ]:


df.head()


# In[ ]:


df['PassengerId'] = df.index
df = df.set_index('Survived')
df.head()


# In[ ]:



df = df.reset_index()
df.head()


# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


df['Age'].unique()


# In[ ]:


df=df[df['Age'] == 50]
df.head()


# <a id="24"></a> <br>
# # 4-2 Pandas:Missing values
# 

# In[ ]:


df = pd.read_csv('../input/train.csv')
df


# In[ ]:


df.fillna


# In[ ]:


df = df.set_index('PassengerId')
df = df.sort_index()
df


# In[ ]:


df = df.reset_index()
df = df.set_index(['PassengerId', 'Survived'])
df


# In[ ]:


df = df.fillna(method='ffill')
df.head()


# <a id="25"></a> <br>
# # 4-3 Pandas :Merging Dataframes
# pandas provides various facilities for easily combining together Series, DataFrame, and Panel objects with various kinds of set logic for the indexes and relational algebra functionality in the case of join / merge-type operations.
# 

# In[ ]:


df = pd.DataFrame([{'Name': 'MJ', 'Item Purchased': 'Sponge', 'Cost': 22.50},
                   {'Name': 'Kevyn', 'Item Purchased': 'Kitty Litter', 'Cost': 2.50},
                   {'Name': 'Filip', 'Item Purchased': 'Spoon', 'Cost': 5.00}],
                  index=['Store 1', 'Store 1', 'Store 2'])
df


# In[ ]:


df['Date'] = ['December 1', 'January 1', 'mid-May']
df


# In[ ]:


df['Delivered'] = True
df


# In[ ]:


df['Feedback'] = ['Positive', None, 'Negative']
df


# In[ ]:


adf = df.reset_index()
adf['Date'] = pd.Series({0: 'December 1', 2: 'mid-May'})
adf


# In[ ]:


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR'},
                         {'Name': 'Sally', 'Role': 'Course liasion'},
                         {'Name': 'James', 'Role': 'Grader'}])
staff_df = staff_df.set_index('Name')
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business'},
                           {'Name': 'Mike', 'School': 'Law'},
                           {'Name': 'Sally', 'School': 'Engineering'}])
student_df = student_df.set_index('Name')
print(staff_df.head())
print()
print(student_df.head())


# In[ ]:


pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)


# In[ ]:


pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)


# In[ ]:


pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)


# In[ ]:


pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)


# In[ ]:


staff_df = staff_df.reset_index()
student_df = student_df.reset_index()
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


# In[ ]:


staff_df = pd.DataFrame([{'Name': 'Kelly', 'Role': 'Director of HR', 'Location': 'State Street'},
                         {'Name': 'Sally', 'Role': 'Course liasion', 'Location': 'Washington Avenue'},
                         {'Name': 'James', 'Role': 'Grader', 'Location': 'Washington Avenue'}])
student_df = pd.DataFrame([{'Name': 'James', 'School': 'Business', 'Location': '1024 Billiard Avenue'},
                           {'Name': 'Mike', 'School': 'Law', 'Location': 'Fraternity House #22'},
                           {'Name': 'Sally', 'School': 'Engineering', 'Location': '512 Wilson Crescent'}])
pd.merge(staff_df, student_df, how='left', left_on='Name', right_on='Name')


# In[ ]:


staff_df = pd.DataFrame([{'First Name': 'Kelly', 'Last Name': 'Desjardins', 'Role': 'Director of HR'},
                         {'First Name': 'Sally', 'Last Name': 'Brooks', 'Role': 'Course liasion'},
                         {'First Name': 'James', 'Last Name': 'Wilde', 'Role': 'Grader'}])
student_df = pd.DataFrame([{'First Name': 'James', 'Last Name': 'Hammond', 'School': 'Business'},
                           {'First Name': 'Mike', 'Last Name': 'Smith', 'School': 'Law'},
                           {'First Name': 'Sally', 'Last Name': 'Brooks', 'School': 'Engineering'}])
staff_df
student_df
pd.merge(staff_df, student_df, how='inner', left_on=['First Name','Last Name'], right_on=['First Name','Last Name'])


# <a id="26"></a> <br>
# # 4-4 Idiomatic Pandas: Making Code Pandorable
# 

# In[ ]:



df = pd.read_csv('../input/train.csv')
df


# In[ ]:


df = df[df['Age']==50]
df.set_index(['PassengerId','Survived'], inplace=True)
df.rename(columns={'Pclass': 'pclass'})


# <a id="27"></a> <br>
# ## 4-5 Pandas :Group by

# In[ ]:



df = pd.read_csv('../input/train.csv')
df = df[df['Age']==50]
df


# In[ ]:


df.head()


# <a id="28"></a> <br>
# ## 4-6 Pandas:Scales
# 

# In[ ]:


df = pd.DataFrame(['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D'],
                  index=['excellent', 'excellent', 'excellent', 'good', 'good', 'good', 'ok', 'ok', 'ok', 'poor', 'poor'])
df.rename(columns={0: 'Grades'}, inplace=True)
df


# In[ ]:


df['Grades'].astype('category').head()


# In[ ]:


grades = df['Grades'].astype('category',
                             categories=['D', 'D+', 'C-', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+'],
                             ordered=True)
grades.head()


# In[ ]:


grades > 'C'


# <a id="30"></a> <br>
# ## 4-7 Pandas:Date Functionality
# ###### [Go to top](#top)

# ### 4-7-1 Timestamp

# In[ ]:


pd.Timestamp('9/1/2016 10:05AM')


# ### 4-7-2 Period

# In[ ]:


pd.Period('1/2016')


# In[ ]:


pd.Period('3/5/2016')


# ### 4-7-3 DatetimeIndex

# In[ ]:


t1 = pd.Series(list('abc'), [pd.Timestamp('2016-09-01'), pd.Timestamp('2016-09-02'), pd.Timestamp('2016-09-03')])
t1


# In[ ]:


type(t1.index)


# ### 4-7-4 PeriodIndex

# In[ ]:


t2 = pd.Series(list('def'), [pd.Period('2016-09'), pd.Period('2016-10'), pd.Period('2016-11')])
t2


# In[ ]:


type(t2.index)


# ## 4-8 Pandas: Converting to Datetime

# In[ ]:


d1 = ['2 June 2013', 'Aug 29, 2014', '2015-06-26', '7/12/16']
ts3 = pd.DataFrame(np.random.randint(10, 100, (4,2)), index=d1, columns=list('ab'))
ts3


# In[ ]:


ts3.index = pd.to_datetime(ts3.index)
ts3


# In[ ]:


pd.to_datetime('4.7.12', dayfirst=True)


# In[ ]:


pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')


# ### 4-8-1 Timedeltas

# In[ ]:


pd.Timestamp('9/3/2016')-pd.Timestamp('9/1/2016')


# In[ ]:


pd.Timestamp('9/2/2016 8:10AM') + pd.Timedelta('12D 3H')


# ### 4-8-2 Working with Dates in a Dataframe
# 

# In[ ]:


dates = pd.date_range('10-01-2016', periods=9, freq='2W-SUN')
dates


# In[ ]:


df.index.ravel


# <a id="31"></a> <br>
# ## 4-9 Distributions in Pandas
# ###### [Go to top](#top)

# In[ ]:


np.random.binomial(1, 0.5)


# In[ ]:


np.random.binomial(1000, 0.5)/1000


# In[ ]:


chance_of_tornado = 0.01/100
np.random.binomial(100000, chance_of_tornado)


# In[ ]:


chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))


# In[ ]:


np.random.uniform(0, 1)


# In[ ]:


np.random.normal(0.75)


# In[ ]:


distribution = np.random.normal(0.75,size=1000)

np.sqrt(np.sum((np.mean(distribution)-distribution)**2)/len(distribution))


# In[ ]:


np.std(distribution)


# In[ ]:



stats.kurtosis(distribution)


# In[ ]:


stats.skew(distribution)


# In[ ]:


chi_squared_df2 = np.random.chisquare(2, size=10000)
stats.skew(chi_squared_df2)


# In[ ]:


chi_squared_df5 = np.random.chisquare(5, size=10000)
stats.skew(chi_squared_df5)


# In[ ]:


output = plt.hist([chi_squared_df2,chi_squared_df5], bins=50, histtype='step', 
                  label=['2 degrees of freedom','5 degrees of freedom'])
plt.legend(loc='upper right')


# <a id="33"></a> <br>
# ##  5- Matplotlib
# 
# This Matplotlib tutorial takes you through the basics Python data visualization: the anatomy of a plot, pyplot and pylab, and much more
# ###### [Go to top](#top)

# You can show matplotlib figures directly in the notebook by using the `%matplotlib notebook` and `%matplotlib inline` magic commands. 
# 
# `%matplotlib notebook` provides an interactive environment.

# In[ ]:


# because the default is the line style '-', 
# nothing will be shown if we only pass in one point (3,2)
plt.plot(3, 2)


# In[ ]:


# we can pass in '.' to plt.plot to indicate that we want
# the point (3,2) to be indicated with a marker '.'
plt.plot(3, 2, '.')


# Let's see how to make a plot without using the scripting layer.

# In[ ]:


# First let's set the backend without using mpl.use() from the scripting layer
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

# create a new figure
fig = Figure()

# associate fig with the backend
canvas = FigureCanvasAgg(fig)

# add a subplot to the fig
ax = fig.add_subplot(111)

# plot the point (3,2)
ax.plot(3, 2, '.')

# save the figure to test.png
# you can see this figure in your Jupyter workspace afterwards by going to
# https://hub.coursera-notebooks.org/
canvas.print_png('test.png')


# We can use html cell magic to display the image.

# In[ ]:


get_ipython().run_cell_magic(u'html', u'', u"<img src='test.png' />")


# In[ ]:


# create a new figure
plt.figure()

# plot the point (3,2) using the circle marker
plt.plot(3, 2, 'o')

# get the current axes
ax = plt.gca()

# Set axis properties [xmin, xmax, ymin, ymax]
ax.axis([0,6,0,10])


# In[ ]:


# create a new figure
plt.figure()

# plot the point (1.5, 1.5) using the circle marker
plt.plot(1.5, 1.5, 'o')
# plot the point (2, 2) using the circle marker
plt.plot(2, 2, 'o')
# plot the point (2.5, 2.5) using the circle marker
plt.plot(2.5, 2.5, 'o')


# In[ ]:


# get current axes
ax = plt.gca()
# get all the child objects the axes contains
ax.get_children()


# In[ ]:



plt.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
plt.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
plt.xlim(0.5, 4.5)
plt.show()


# <a id="34"></a> <br>
# ## 5-1 Scatterplots

# In[ ]:




x = np.array([1,2,3,4,5,6,7,8])
y = x

plt.figure()
plt.scatter(x, y) # similar to plt.plot(x, y, '.'), but the underlying child objects in the axes are not Line2D


# In[ ]:




x = np.array([1,2,3,4,5,6,7,8])
y = x

# create a list of colors for each point to have
# ['green', 'green', 'green', 'green', 'green', 'green', 'green', 'red']
colors = ['green']*(len(x)-1)
colors.append('red')

plt.figure()

# plot the point with size 100 and chosen colors
plt.scatter(x, y, s=100, c=colors)


# In[ ]:


# convert the two lists into a list of pairwise tuples
zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])

print(list(zip_generator))
# the above prints:
# [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]

zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
# The single star * unpacks a collection into positional arguments
print(*zip_generator)
# the above prints:
# (1, 6) (2, 7) (3, 8) (4, 9) (5, 10)


# In[ ]:


# use zip to convert 5 tuples with 2 elements each to 2 tuples with 5 elements each
print(list(zip((1, 6), (2, 7), (3, 8), (4, 9), (5, 10))))
# the above prints:
# [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]


zip_generator = zip([1,2,3,4,5], [6,7,8,9,10])
# let's turn the data back into 2 lists
x, y = zip(*zip_generator) # This is like calling zip((1, 6), (2, 7), (3, 8), (4, 9), (5, 10))
print(x)
print(y)
# the above prints:
# (1, 2, 3, 4, 5)
# (6, 7, 8, 9, 10)


# In[ ]:


plt.figure()
# plot a data series 'Tall students' in red using the first two elements of x and y
plt.scatter(x[:2], y[:2], s=100, c='red', label='Tall students')
# plot a second data series 'Short students' in blue using the last three elements of x and y 
plt.scatter(x[2:], y[2:], s=100, c='blue', label='Short students')


# In[ ]:


# add a label to the x axis
plt.xlabel('The number of times the child kicked a ball')
# add a label to the y axis
plt.ylabel('The grade of the student')
# add a title
plt.title('Relationship between ball kicking and grades')


# In[ ]:


# add a legend (uses the labels from plt.scatter)
plt.legend()


# In[ ]:


# add the legend to loc=4 (the lower right hand corner), also gets rid of the frame and adds a title
plt.legend(loc=4, frameon=False, title='Legend')


# In[ ]:


# get children from current axes (the legend is the second to last item in this list)
plt.gca().get_children()


# In[ ]:


# get the legend from the current axes
legend = plt.gca().get_children()[-2]


# In[ ]:


x = np.random.randint(low=1, high=11, size=50)
y = x + np.random.randint(1, 5, size=x.size)
data = np.column_stack((x, y))

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                               figsize=(8, 4))

ax1.scatter(x=x, y=y, marker='o', c='r', edgecolor='b')
ax1.set_title('Scatter: $x$ versus $y$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')

ax2.hist(data, bins=np.arange(data.min(), data.max()),
         label=('x', 'y'))
ax2.legend(loc=(0.65, 0.8))
ax2.set_title('Frequencies of $x$ and $y$')
ax2.yaxis.tick_right()


# <a id="35"></a> <br>
# ## 5-2 Line Plots

# In[ ]:




linear_data = np.array([1,2,3,4,5,6,7,8])
exponential_data = linear_data**2

plt.figure()
# plot the linear data and the exponential data
plt.plot(linear_data, '-o', exponential_data, '-o')


# In[ ]:


# plot another series with a dashed red line
plt.plot([22,44,55], '--r')


# <a id="36"></a> <br>
# ## 5-3 Bar Charts

# In[ ]:


plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3)


# In[ ]:


new_xvals = []

# plot another set of bars, adjusting the new xvals to make up for the first set of bars plotted
for item in xvals:
    new_xvals.append(item+0.3)

plt.bar(new_xvals, exponential_data, width = 0.3 ,color='red')


# In[ ]:


from random import randint
linear_err = [randint(0,15) for x in range(len(linear_data))] 

# This will plot a new set of bars with errorbars using the list of random error values
plt.bar(xvals, linear_data, width = 0.3, yerr=linear_err)


# In[ ]:


# stacked bar charts are also possible
plt.figure()
xvals = range(len(linear_data))
plt.bar(xvals, linear_data, width = 0.3, color='b')
plt.bar(xvals, exponential_data, width = 0.3, bottom=linear_data, color='r')


# In[ ]:


# or use barh for horizontal bar charts
plt.figure()
xvals = range(len(linear_data))
plt.barh(xvals, linear_data, height = 0.3, color='b')
plt.barh(xvals, exponential_data, height = 0.3, left=linear_data, color='r')


# In[ ]:




# Initialize the plot
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# or replace the three lines of code above by the following line: 
#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))

# Plot the data
ax1.bar([1,2,3],[3,4,5])
ax2.barh([0.5,1,2.5],[0,1,2])

# Show the plot
plt.show()


# In[ ]:


plt.figure()
# subplot with 1 row, 2 columns, and current axis is 1st subplot axes
plt.subplot(1, 2, 1)

linear_data = np.array([1,2,3,4,5,6,7,8])

plt.plot(linear_data, '-o')


# In[ ]:


exponential_data = linear_data**2 

# subplot with 1 row, 2 columns, and current axis is 2nd subplot axes
plt.subplot(1, 2, 2)
plt.plot(exponential_data, '-o')


# In[ ]:


# plot exponential data on 1st subplot axes
plt.subplot(1, 2, 1)
plt.plot(exponential_data, '-x')


# In[ ]:


plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.plot(linear_data, '-o')
# pass sharey=ax1 to ensure the two subplots share the same y axis
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
plt.plot(exponential_data, '-x')


# In[ ]:


plt.figure()
# the right hand side is equivalent shorthand syntax
plt.subplot(1,2,1) == plt.subplot(121)


# In[ ]:


# create a 3x3 grid of subplots
fig, ((ax1,ax2,ax3), (ax4,ax5,ax6), (ax7,ax8,ax9)) = plt.subplots(3, 3, sharex=True, sharey=True)
# plot the linear_data on the 5th subplot axes 
ax5.plot(linear_data, '-')


# In[ ]:


# set inside tick labels to visible
for ax in plt.gcf().get_axes():
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_visible(True)
plt.show()


# In[ ]:


# necessary on some systems to update the plot
plt.gcf().canvas.draw()
plt.show()


# <a id="37"></a> <br>
# ## 5-4 Histograms

# In[ ]:


# create 2x2 grid of axis subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1,ax2,ax3,ax4]

# draw n = 10, 100, 1000, and 10000 samples from the normal distribution and plot corresponding histograms
for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample)
    axs[n].set_title('n={}'.format(sample_size))


# In[ ]:


# repeat with number of bins set to 100
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
axs = [ax1,ax2,ax3,ax4]

for n in range(0,len(axs)):
    sample_size = 10**(n+1)
    sample = np.random.normal(loc=0.0, scale=1.0, size=sample_size)
    axs[n].hist(sample, bins=100)
    axs[n].set_title('n={}'.format(sample_size))


# In[ ]:


plt.figure()
Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
plt.scatter(X,Y)


# In[ ]:


# use gridspec to partition the figure into subplots
import matplotlib.gridspec as gridspec

plt.figure()
gspec = gridspec.GridSpec(3, 3)

top_histogram = plt.subplot(gspec[0, 1:])
side_histogram = plt.subplot(gspec[1:, 0])
lower_right = plt.subplot(gspec[1:, 1:])


# In[ ]:


Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
lower_right.scatter(X, Y)
top_histogram.hist(X, bins=100)
s = side_histogram.hist(Y, bins=100, orientation='horizontal')


# In[ ]:


# clear the histograms and plot normed histograms
top_histogram.clear()
top_histogram.hist(X, bins=100, normed=True)
side_histogram.clear()
side_histogram.hist(Y, bins=100, orientation='horizontal', normed=True)
# flip the side histogram's x axis
side_histogram.invert_xaxis()


# In[ ]:


# change axes limits
for ax in [top_histogram, lower_right]:
    ax.set_xlim(0, 1)
for ax in [side_histogram, lower_right]:
    ax.set_ylim(-5, 5)


# <a id="38"></a> <br>
# ## 5-5 Box and Whisker Plots

# In[ ]:



normal_sample = np.random.normal(loc=0.0, scale=1.0, size=10000)
random_sample = np.random.random(size=10000)
gamma_sample = np.random.gamma(2, size=10000)

df = pd.DataFrame({'normal': normal_sample, 
                   'random': random_sample, 
                   'gamma': gamma_sample})


# In[ ]:


df.describe()


# In[ ]:


plt.figure()
# create a boxplot of the normal data, assign the output to a variable to supress output
_ = plt.boxplot(df['normal'], whis='range')


# In[ ]:


# clear the current figure
plt.clf()
# plot boxplots for all three of df's columns
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')


# In[ ]:


plt.figure()
_ = plt.hist(df['gamma'], bins=100)


# In[ ]:


import mpl_toolkits.axes_grid1.inset_locator as mpl_il

plt.figure()
plt.boxplot([ df['normal'], df['random'], df['gamma'] ], whis='range')
# overlay axis on top of another 
ax2 = mpl_il.inset_axes(plt.gca(), width='60%', height='40%', loc=2)
ax2.hist(df['gamma'], bins=100)
ax2.margins(x=0.5)


# In[ ]:


# switch the y axis ticks for ax2 to the right side
ax2.yaxis.tick_right()


# In[ ]:


# if `whis` argument isn't passed, boxplot defaults to showing 1.5*interquartile (IQR) whiskers with outliers
plt.figure()
_ = plt.boxplot([ df['normal'], df['random'], df['gamma'] ] )


# <a id="39"></a> <br>
# ## 5-6 Heatmaps

# In[ ]:


plt.figure()

Y = np.random.normal(loc=0.0, scale=1.0, size=10000)
X = np.random.random(size=10000)
_ = plt.hist2d(X, Y, bins=25)


# In[ ]:


plt.figure()
_ = plt.hist2d(X, Y, bins=100)


# <a id="40"></a> <br>
# ## 5-7 Animations

# In[ ]:


import matplotlib.animation as animation

n = 100
x = np.random.randn(n)


# In[ ]:


# create the function that will do the plotting, where curr is the current frame
def update(curr):
    # check if animation is at the last frame, and if so, stop the animation a
    if curr == n: 
        a.event_source.stop()
    plt.cla()
    bins = np.arange(-4, 4, 0.5)
    plt.hist(x[:curr], bins=bins)
    plt.axis([-4,4,0,30])
    plt.gca().set_title('Sampling the Normal Distribution')
    plt.gca().set_ylabel('Frequency')
    plt.gca().set_xlabel('Value')
    plt.annotate('n = {}'.format(curr), [3,27])


# In[ ]:


fig = plt.figure()
a = animation.FuncAnimation(fig, update, interval=100)


# <a id="41"></a> <br>
# ## 5-8 Interactivity

# In[ ]:


plt.figure()
data = np.random.rand(10)
plt.plot(data)

def onclick(event):
    plt.cla()
    plt.plot(data)
    plt.gca().set_title('Event at pixels {},{} \nand data {},{}'.format(event.x, event.y, event.xdata, event.ydata))

# tell mpl_connect we want to pass a 'button_press_event' into onclick when the event is detected
plt.gcf().canvas.mpl_connect('button_press_event', onclick)


# In[ ]:


from random import shuffle
origins = ['China', 'Brazil', 'India', 'USA', 'Canada', 'UK', 'Germany', 'Iraq', 'Chile', 'Mexico']

shuffle(origins)

df = pd.DataFrame({'height': np.random.rand(10),
                   'weight': np.random.rand(10),
                   'origin': origins})
df


# In[ ]:


plt.figure()
# picker=5 means the mouse doesn't have to click directly on an event, but can be up to 5 pixels away
plt.scatter(df['height'], df['weight'], picker=5)
plt.gca().set_ylabel('Weight')
plt.gca().set_xlabel('Height')


# In[ ]:


def onpick(event):
    origin = df.iloc[event.ind[0]]['origin']
    plt.gca().set_title('Selected item came from {}'.format(origin))

# tell mpl_connect we want to pass a 'pick_event' into onpick when the event is detected
plt.gcf().canvas.mpl_connect('pick_event', onpick)


# In[ ]:


# use the 'seaborn-colorblind' style
plt.style.use('seaborn-colorblind')


# <a id="42"></a> <br>
# ## 5-9 DataFrame.plot

# In[ ]:


np.random.seed(123)

df = pd.DataFrame({'A': np.random.randn(365).cumsum(0), 
                   'B': np.random.randn(365).cumsum(0) + 20,
                   'C': np.random.randn(365).cumsum(0) - 20}, 
                  index=pd.date_range('1/1/2017', periods=365))
df.head()


# In[ ]:


df.plot('A','B', kind = 'scatter');


# You can also choose the plot kind by using the `DataFrame.plot.kind` methods instead of providing the `kind` keyword argument.
# 
# `kind` :
# - `'line'` : line plot (default)
# - `'bar'` : vertical bar plot
# - `'barh'` : horizontal bar plot
# - `'hist'` : histogram
# - `'box'` : boxplot
# - `'kde'` : Kernel Density Estimation plot
# - `'density'` : same as 'kde'
# - `'area'` : area plot
# - `'pie'` : pie plot
# - `'scatter'` : scatter plot
# - `'hexbin'` : hexbin plot
# ###### [Go to top](#top)

# In[ ]:


# create a scatter plot of columns 'A' and 'C', with changing color (c) and size (s) based on column 'B'
df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')


# In[ ]:


ax = df.plot.scatter('A', 'C', c='B', s=df['B'], colormap='viridis')
ax.set_aspect('equal')


# In[ ]:


df.plot.box();


# In[ ]:


df.plot.hist(alpha=0.7);


# [Kernel density estimation plots](https://en.wikipedia.org/wiki/Kernel_density_estimation) are useful for deriving a smooth continuous function from a given sample.

# In[ ]:


df.plot.kde();


# <a id="43"></a> <br>
# # 6- Seaborn
# 
# As you have just read, **Seaborn** is complimentary to Matplotlib and it specifically targets statistical data visualization. But it goes even further than that: Seaborn extends Matplotlib and that’s why it can address the two biggest frustrations of working with Matplotlib. Or, as Michael Waskom says in the “introduction to Seaborn”: “If matplotlib “tries to make easy things easy and hard things possible”, seaborn tries to make a well-defined set of hard things easy too.”
# 
# One of these hard things or frustrations had to do with the default Matplotlib parameters. Seaborn works with different parameters, which undoubtedly speaks to those users that don’t use the default looks of the Matplotlib plots
# Seaborn is a library for making statistical graphics in Python. It is built on top of matplotlib and closely integrated with pandas data structures.
# 
# Here is some of the functionality that seaborn offers:
# 
# A dataset-oriented API for examining relationships between multiple variables
# Specialized support for using categorical variables to show observations or aggregate statistics
# Options for visualizing univariate or bivariate distributions and for comparing them between subsets of data
# Automatic estimation and plotting of linear regression models for different kinds dependent variables
# Convenient views onto the overall structure of complex datasets
# High-level abstractions for structuring multi-plot grids that let you easily build complex visualizations
# Concise control over matplotlib figure styling with several built-in themes
# Tools for choosing color palettes that faithfully reveal patterns in your data
# Seaborn aims to make visualization a central part of exploring and understanding data. Its dataset-oriented plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots.
# 
# Here’s an example of what this means:

# <a id="43"></a> <br>
# ## 6-1 Seaborn Vs Matplotlib
# 
# It is summarized that if Matplotlib “tries to make easy things easy and hard things possible”, Seaborn tries to make a well defined set of hard things easy too.”
# 
# Seaborn helps resolve the two major problems faced by Matplotlib; the problems are
# 
# * Default Matplotlib parameters
# * Working with data frames
# 
# As Seaborn compliments and extends Matplotlib, the learning curve is quite gradual. If you know Matplotlib, you are already half way through Seaborn.
# 
# Important Features of Seaborn
# Seaborn is built on top of Python’s core visualization library Matplotlib. It is meant to serve as a complement, and not a replacement. However, Seaborn comes with some very important features. Let us see a few of them here. The features help in −
# 
# * Built in themes for styling matplotlib graphics
# * Visualizing univariate and bivariate data
# * Fitting in and visualizing linear regression models
# * Plotting statistical time series data
# * Seaborn works well with NumPy and Pandas data structures
# * It comes with built in themes for styling Matplotlib graphics
# 
# In most cases, you will still use Matplotlib for simple plotting. The knowledge of Matplotlib is recommended to tweak Seaborn’s default plots.

# In[ ]:


def sinplot(flip = 1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5): 
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
sinplot()
plt.show()


# In[ ]:


def sinplot(flip = 1):
   x = np.linspace(0, 14, 100)
   for i in range(1, 5):
      plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
 
sns.set()
sinplot()
plt.show()


# In[ ]:


np.random.seed(1234)

v1 = pd.Series(np.random.normal(0,10,1000), name='v1')
v2 = pd.Series(2*v1 + np.random.normal(60,15,1000), name='v2')


# In[ ]:


plt.figure()
plt.hist(v1, alpha=0.7, bins=np.arange(-50,150,5), label='v1');
plt.hist(v2, alpha=0.7, bins=np.arange(-50,150,5), label='v2');
plt.legend();


# In[ ]:


plt.figure()
# we can pass keyword arguments for each individual component of the plot
sns.distplot(v2, hist_kws={'color': 'Teal'}, kde_kws={'color': 'Navy'});


# In[ ]:


sns.jointplot(v1, v2, alpha=0.4);


# In[ ]:


grid = sns.jointplot(v1, v2, alpha=0.4);
grid.ax_joint.set_aspect('equal')


# In[ ]:


sns.jointplot(v1, v2, kind='hex');


# In[ ]:


# set the seaborn style for all the following plots
sns.set_style('white')

sns.jointplot(v1, v2, kind='kde', space=0);


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# ## 6-2  10 Useful Python Data Visualization Libraries

# I am giving an overview of 10 interdisciplinary Python data visualization libraries, from the well-known to the obscure.
# 
# * 1- matplotlib
# 
# matplotlib is the O.G. of Python data visualization libraries. Despite being over a decade old, it’s still the most widely used library for plotting in the Python community. It was designed to closely resemble MATLAB, a proprietary programming language developed in the 1980s.
# 
# * 2- Seaborn
# 
# Seaborn harnesses the power of matplotlib to create beautiful charts in a few lines of code. The key difference is Seaborn’s default styles and color palettes, which are designed to be more aesthetically pleasing and modern. Since Seaborn is built on top of matplotlib, you’ll need to know matplotlib to tweak Seaborn’s defaults.
# 
# * 3- ggplot
# 
# ggplot is based on ggplot2, an R plotting system, and concepts from The Grammar of Graphics. ggplot operates differently than matplotlib: it lets you layer components to create a complete plot. For instance, you can start with axes, then add points, then a line, a trendline, etc. Although The Grammar of Graphics has been praised as an “intuitive” method for plotting, seasoned matplotlib users might need time to adjust to this new mindset.
# 
# 
# * 4- Bokeh
# 
# Like ggplot, Bokeh is based on The Grammar of Graphics, but unlike ggplot, it’s native to Python, not ported over from R. Its strength lies in the ability to create interactive, web-ready plots, which can be easily outputted as JSON objects, HTML documents, or interactive web applications. Bokeh also supports streaming and real-time data.
# 
# 
# * 5- pygal
# 
# Like Bokeh and Plotly, pygal offers interactive plots that can be embedded in the web browser. Its prime differentiator is the ability to output charts as SVGs. As long as you’re working with smaller datasets, SVGs will do you just fine. But if you’re making charts with hundreds of thousands of data points, they’ll have trouble rendering and become sluggish.
# 
# * 6- Plotly
# 
# You might know Plotly as an online platform for data visualization, but did you also know you can access its capabilities from a Python notebook? Like Bokeh, Plotly’s forte is making interactive plots, but it offers some charts you won’t find in most libraries, like contour plots, dendograms, and 3D charts.
# 
# * 7- geoplotlib
# 
# geoplotlib is a toolbox for creating maps and plotting geographical data. You can use it to create a variety of map-types, like choropleths, heatmaps, and dot density maps. You must have Pyglet (an object-oriented programming interface) installed to use geoplotlib. Nonetheless, since most Python data visualization libraries don’t offer maps, it’s nice to have a library dedicated solely to them.
# 
# * 8- Gleam
# 
# Gleam is inspired by R’s Shiny package. It allows you to turn analyses into interactive web apps using only Python scripts, so you don’t have to know any other languages like HTML, CSS, or JavaScript. Gleam works with any Python data visualization library. Once you’ve created a plot, you can build fields on top of it so users can filter and sort data.
# 
# 
# * 9- missingno
# 
# Dealing with missing data is a pain. missingno allows you to quickly gauge the completeness of a dataset with a visual summary, instead of trudging through a table. You can filter and sort data based on completion or spot correlations with a heatmap or a dendrogram.
# 
# 
# * 10- Leather
# 
# Leather’s creator, Christopher Groskopf, puts it best: “Leather is the Python charting library for those who need charts now and don’t care if they’re perfect.” It’s designed to work with all data types and produces charts as SVGs, so you can scale them without losing image quality. Since this library is relatively new, some of the documentation is still in progress. The charts you can make are pretty basic—but that’s the intention.
# 
# At the end, nice cheatsheet on how to best visualize your data. I think I will print it out as a good reminder of "best practices". Check out the link for the complete cheatsheet, also as a PDF. hashtag#data hashtag#visualization hashtag#datascience
# 
# Link: https://github.com/mjbahmani/Machine-Learning-Workflow-with-Python
# ![cheatsheet ][1]
# [Reference][2]
# 
# 
#   [1]: http://s8.picofile.com/file/8340669884/53f6a826_d7df_4b55_81e6_7c23b3fff0a3_original.png
#   [2]: https://blog.modeanalytics.com/python-data-visualization-libraries/

# <a id="59"></a> <br>
# ## 8- Plotly
# How to use **Plotly** offline inside IPython notebooks.

# <a id="5"></a> <br>
# ## 8-1 New to Plotly?
# Plotly, also known by its URL, Plot.ly, is a technical computing company headquartered in Montreal, Quebec, that develops online data analytics and visualization tools. Plotly provides online graphing, analytics, and statistics tools for individuals and collaboration, as well as scientific graphing libraries for Python, R, MATLAB, Perl, Julia, Arduino, and REST.

# In[ ]:


# example for plotly
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
from plotly import tools
import plotly.figure_factory as ff
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
trace = go.Scatter(x=X[:, 0],
                   y=X[:, 1],
                   mode='markers',
                   marker=dict(color=np.random.randn(150),
                               size=10,
                               colorscale='Viridis',
                               showscale=False))

layout = go.Layout(title='Training Points',
                   xaxis=dict(title='Sepal length',
                            showgrid=False),
                   yaxis=dict(title='Sepal width',
                            showgrid=False),
                  )
 
fig = go.Figure(data=[trace], layout=layout)


# In[ ]:


py.iplot(fig)


# <a id="5"></a> <br>
# ## 8-2 Plotly Offline from Command Line
# You can plot your graphs from a python script from command line. On executing the script, it will open a web browser with your Plotly Graph drawn.

# In[ ]:


import plotly.graph_objs as go

plot([go.Scatter(x=[1, 2, 3], y=[3, 1, 6])])


# <a id="5"></a> <br>
# ## 8-3 Generating Offline Graphs within Jupyter Notebook
# You can also plot your graphs offline inside a Jupyter Notebook Environment. First you need to initiate the Plotly Notebook mode as below:

# In[ ]:


init_notebook_mode(connected=True)


# Run at the start of every ipython notebook to use plotly.offline. This injects the plotly.js source files into the notebook.
# 
# 

# In[ ]:


iplot([{"x": [1, 2, 3], "y": [3, 1, 6]}])


# In[ ]:


import plotly.graph_objs as go

import numpy as np

x = np.random.randn(2000)
y = np.random.randn(2000)
iplot([go.Histogram2dContour(x=x, y=y, contours=dict(coloring='heatmap')),
       go.Scatter(x=x, y=y, mode='markers', marker=dict(color='white', size=3, opacity=0.3))], show_link=False)


# ## 8-4 Plotting Offline with Cufflinks
# 

# In[ ]:


import cufflinks as cf
iplot(cf.datagen.lines().iplot(asFigure=True,
                               kind='scatter',xTitle='Dates',yTitle='Returns',title='Returns'))


# <a id="60"></a> <br>
# ## 9- Resource
# In the end, I have prepared some resource that can be useful for you.

# <a id="61"></a> <br>
# ## 9-1 Courses
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
#  <a id="61"></a> <br>
# ## 9-2 Ebooks
# If you love reading , here is **10 free machine learning books**
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
# ## 9-3 Cheat Sheets
# Data Science is an ever-growing field, there are numerous tools & techniques to remember. It is not possible for anyone to remember all the functions, operations and formulas of each concept. That’s why we have cheat sheets. But there are a plethora of cheat sheets available out there, choosing the right cheat sheet is a tough task. So, I decided to collect them here
# 
# Here I have selected the cheat sheets on the following criteria: comprehensiveness, clarity, and content [5]:
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
# <a id="61"></a> <br>
# ## 10 -Conclusion
# You have got an introduction to the main packages and ideas in the **The Data Scientist’s Toolbox Tutorial - 1**. I hope,  you have had fun with it also, I want to hear your voice to update this kernel. in addition, there is the new and second version of the kernel that introduces the next packages. to continue please click on [**The Data Scientist’s Toolbox Tutorial - 2**](https://www.kaggle.com/mjbahmani/the-data-scientist-s-toolbox-tutorial-2)

# 
# 
# ---------------------------------------------------------------------
# Fork, Run and Follow this kernel on GitHub:
# > ###### [ GitHub](https://github.com/mjbahmani/10-steps-to-become-a-data-scientist)
# 
# 
# -------------------------------------------------------------------------------------------------------------
#  **I hope you find this kernel helpful and some <font color="red">UPVOTES</font> would be very much appreciated.**
#  
#  -----------
# 

# <a id="62"></a> <br>
# ## 11- References
# 1. [Coursera](https://www.coursera.org/specializations/data-science-python)
# 1. [GitHub](https://github.com/mjbahmani)
# 1. [plot.ly](https://plot.ly/python/offline/)
# 1. [tutorialspoint](https://www.tutorialspoint.com/python/python_classes_objects.htm)
# 1. [Top 28 Cheat Sheets for Machine Learning](https://www.analyticsvidhya.com/blog/2017/02/top-28-cheat-sheets-for-machine-learning-data-science-probability-sql-big-data/)
# ###### [Go to top](#top)
