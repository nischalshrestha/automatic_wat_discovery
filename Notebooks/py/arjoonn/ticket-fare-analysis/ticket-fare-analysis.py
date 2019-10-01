#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.info()
len(data.Ticket.unique())


# Since the length is different from the expected value of 891 we know that there are some duplicates. What are the Fare values of these duplicate tickets?

# In[ ]:


counts = []
for tk in data.Ticket.unique():
    this = data.loc[data.Ticket == tk, 'Fare']
    l = len(this.unique())
    counts.append(l)
    if l > 1:
        print(data.loc[data.Ticket == tk])
print(set(counts))


# We see that there is just one case where the same ticket has a different Fare for the people associated with it. Who are these people?  
# According to **[Encyclopedia-Titanica](http://www.encyclopedia-titanica.org/titanic-victim/olaf-elon-osen.html)** they were travelling companions.
# 
# The fare differs since they had different destinations. What about the other cases?

# In[ ]:


data.Fare.hist(figsize=(15, 10), bins=50)
for tk in data.Ticket.unique():
    this = data.loc[data.Ticket == tk, 'Fare']
    no_of_people = this.count()
    distribute_evenly = this / no_of_people
    data.loc[data.Ticket == tk, 'Fare'] = distribute_evenly
data.Fare.hist(bins=50, alpha=0.5)


# As we can see, the histogram changes a lot since the Fare column was earlier denoting **Fare per Ticket**  
# and now denotes **Fare per person**.
