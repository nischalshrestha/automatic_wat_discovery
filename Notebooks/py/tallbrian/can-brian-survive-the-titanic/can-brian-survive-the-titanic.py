#!/usr/bin/env python
# coding: utf-8

# Hello World
# ==========
# This is my first attempt at Titanic...  I am new to data science and pandas but I am excited to learn.
# --------------------------------------------------------------------------------------------------------------------

# In the first cell I am __importing several libraries__
# ---------------------------
# 
#  - numpy to take means
#  - csv to pull in comma separated value file
#  - matplotlib to do some visualizations
# 
# then reading in the training data file
# --------------------

# In[ ]:


import numpy as np
import csv

with open('../input/train.csv') as csvfile:
    reader = csv.reader(csvfile)
    h = reader.__next__()
    data = [r for r in reader]
    
print(h)
print(data[0])


# In[ ]:


p_class = lambda x: x[-10]
p_namelen = lambda x: len(x[-9])
p_title = lambda x: [w for w in x[-9].split() if w.find('.') != -1][0]
p_gender = lambda x: x[-8]
p_age = lambda x: x[-7]
p_numsibs = lambda x: x[-6]
p_parch = lambda x: x[-5]
p_ticket = lambda x: x[-4]
p_fare = lambda x: x[-3]
p_cabin_let = lambda x: [y[0] for y in x[-2].split()] if len(x[-2])>0 else 'Z'
p_cabin_num = lambda x: [y[1:] for y in x[-2].split()] if len(x[-2])>0 else 0
p_embark = lambda x: x[-1]

print([[p_cabin_let(x), p_cabin_num(x)] for x in data])

metric_attribute = lambda x: [p_class(x), p_title(x), p_gender(x), p_parch(x)]
metric_continuous = lambda x: [p_namelen(x), p_age(x), p_numsibs(x), p_fare(x)]


# First just get an boolean array of who survived.  This will be used to compare against all feature sets...

# In[ ]:


Survived = [True if r[1] == '1' else False for r in data]
import matplotlib.pyplot as plt
 
# Setup Plot Data
labels = 'Survived', 'Perished'
sizes = [sum(person == True for person in Survived), sum(person != True for person in Survived)]
colors = ['green', 'red']
explode = (0.1, 0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=-20)
 
plt.axis('equal')
plt.show()


# Next, see how survivors can be predicted based on class

# In[ ]:


Class = [r[2] for r in data]
ClassList = list(set(Class))
Survivor_by_class =    [np.mean([1 if Survived[i] else 0 for i in range(len(Survived)) if Class[i] == c]) for c in ClassList]
print('Survivors by Class', [c for c in ClassList], Survivor_by_class)

ind = np.arange(len(ClassList))    # the x locations for the groups
width = 0.5       # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, Survivor_by_class, width, color='r')
plt.ylabel('% Surviving')
plt.title('Survivors by Class')
plt.xticks(ind + width/2., ClassList)
plt.yticks(np.arange(0, 1.01, .1))
plt.show()


# Next, see how survivors can be predicted based on length of name.  A little silly but maybe rich people had longer names and more rich people survived?

# In[ ]:


NameLen = [len(r[3]) for r in data]
NameLenList = list(set(NameLen))
Survivor_by_namelen =    [np.mean([1 if Survived[i] else 0 for i in range(len(Survived)) if NameLen[i] == n]) for n in NameLenList]
print('Survivors by NameLength', [n for n in NameLenList], Survivor_by_namelen)

ind = np.arange(len(NameLenList))    # the x locations for the groups
width = 1      # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, Survivor_by_namelen, width, color='r')
plt.ylabel('% Surviving')
plt.title('Survivors by NameLength')
plt.xticks(ind + width/2., NameLenList, rotation='vertical', fontsize=6)
plt.yticks(np.arange(0, 1.01, .1))
plt.show()


# In[ ]:


TitleString = [[w for w in r[3].split() if w.find('.') != -1][0] for r in data]
TitleList = list(set(TitleString))
TitlesEnum = [TitleList.index(t) for t in TitleString]
Survivor_by_title =    [np.mean([1 if Survived[i] else 0 for i in range(len(Survived)) if TitleString[i] == t]) for t in TitleList]
print('Survivors by Title', [t for t in TitleList], Survivor_by_title)

ind = np.arange(len(TitleList))    # the x locations for the groups
width = 1      # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, Survivor_by_title, width, color='r')
plt.ylabel('% Surviving')
plt.title('Survivors by Title')
plt.xticks(ind + width/2., TitleList, rotation='vertical')
plt.yticks(np.arange(0, 1.01, .1))
plt.show()


# In[ ]:


Gender = [r[4] for r in data]
GenderList = list(set(Gender))
GenderEnum = [GenderList.index(g) for g in Gender]
Age = [r[5] for r in data]

Survivor_by_gender =    [np.mean([1 if Survived[i] else 0 for i in range(len(Survived)) if Gender[i] == g]) for g in GenderList]
print('Survivors by Gender', [g for g in GenderList], Survivor_by_gender)

ind = np.arange(len(GenderList))    # the x locations for the groups
width = 1      # the width of the bars: can also be len(x) sequence
p1 = plt.bar(ind, Survivor_by_gender, width, color='r')
plt.ylabel('% Surviving')
plt.title('Survivors by Title')
plt.xticks(ind + width/2., GenderList, rotation='vertical')
plt.yticks(np.arange(0, 1.01, .1))
plt.show()


# Roll all the estimators into a complete predictor
# ===================================

# In[ ]:


with open('../input/test.csv') as csvfile:
    reader = csv.reader(csvfile)
    h = reader.__next__()
    test_data = [r for r in reader]

for p in test_data:
    prob_by_class = Survivor_by_class[ClassList.index(p[1])] if p[1] in ClassList else 0.5
    prob_by_namelength = Survivor_by_namelen[NameLenList.index(len(p[2]))] if len(p[2]) in NameLenList else 0.5
    prob_by_title = Survivor_by_title[TitleList.index([w for w in p[2].split() if w.find('.') != -1][0])] if [w for w in p[2].split() if w.find('.') != -1][0] in TitleList else 0.5
    prob_by_gender = Survivor_by_gender[GenderList.index(p[3])] if p[3] in GenderList else 0.5

weights = [[sum(p[1]==d[2] for d in data),
            sum(len(p[2])==len(d[3]) for d in data),
            sum([w for w in p[2].split() if w.find('.') != -1][0]  == [x for x in d[3].split() if x.find('.')!= -1][0] for d in data),
            sum(p[3]==d[4] for d in data)] for p in test_data]

norm_weights = [[ww / sum(w) for ww in w] for w in weights]
    
survival_vec = [[Survivor_by_class[ClassList.index(p[1])] if p[1] in ClassList else 0.5, 
                 Survivor_by_namelen[NameLenList.index(len(p[2]))] if len(p[2]) in NameLenList else 0.5,
                 Survivor_by_title[TitleList.index([w for w in p[2].split() if w.find('.') != -1][0])] if [w for w in p[2].split() if w.find('.') != -1][0] in TitleList else 0.5,
                 Survivor_by_gender[GenderList.index(p[3])] if p[3] in GenderList else 0.5]
                for p in test_data]


survival_probability = [np.dot(x[0], x[1]) for x in zip(survival_vec, norm_weights)]

projected_survivors = [1 if p >= .50 else 0 for p in survival_probability]
plt.plot(survival_probability)
plt.show()
print("Projected survivors:", sum(projected_survivors))
print("Projected survival rate:", sum(projected_survivors)/len(projected_survivors))


# In[ ]:


import pandas as pd

submission = pd.DataFrame({
        "PassengerId": [p[0] for p in test_data],
        "Survived": projected_survivors
    })
submission.to_csv('titanic.csv', index=False)
print('Complete!')

