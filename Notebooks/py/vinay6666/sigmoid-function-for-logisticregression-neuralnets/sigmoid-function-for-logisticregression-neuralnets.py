#!/usr/bin/env python
# coding: utf-8

# # ***Sigmoid Function / Logistic Function***

# ### Most often , we would want to predict our outcomes as YES/NO (1/0) . 
# ### For example:
# * Is your favourite football team going to win the match today ? -> yes/no (1/0)
# * Did a particular passenger on titanic survive ? -> yes/no (1/0)

# ## The Sigmoid Function / Logistic Function:

# ![Sigmoid function](https://qph.ec.quoracdn.net/main-qimg-05edc1873d0103e36064862a45566dba?convert_to_webp=true)

# # S(x)=1/(1+e^-x)

# * The sigmoid function gives an 'S' shaped curve. 

# * This curve has a finite limit of :
# * '0' as x approaches -infinity  
# * '1' as x approaches +infinity

# ## The output of sigmoid function when x=0 is 0.5 . Thus , if the output is more than 0.5 , we can classify the outcome as 1 (or Yes) and if it is less than 0.5 , we can classify it as 0 (or No)

# ## For example:
# * If the output is 0.65 , we can say in terms of probability as : "There is a 65 percent chance that your favourite foot ball team is going to win today". 

# ## Thus the output of the sigmoid can not just be used to classify as Yes/No . The output value can also be used to determine the probability of Yes/No. 

# # ***Now we shall realise it Python***

# ## Imports:
# * We need math for writing the sigmoid function
# * numpy to define the values for X-axis
# * matplotlib to plot the output

# In[ ]:


import math
import matplotlib.pyplot as plt
import numpy as np


# ## Next we shall define the sigmoid function as described in the above equation.

# In[ ]:


def sigmoid(x):
    a = []
    for item in x:
               #(the sigmoid function)
        a.append(1/(1+math.exp(-item)))
    return a


# ## Now , we shall generate some values for x :
# * This will have values from -10 to +10 with increment as 0.2  ->(-10.0 , -9.8 ,...0,0.2,0.4,...9.8)

# In[ ]:


x = np.arange(-10., 10., 0.2)


# ## We shall pass the values of 'x' to our sigmoid function and store it's output in variable 'y'

# In[ ]:


y = sigmoid(x)


# ## We will plot the 'x' values in X-axis and 'y'  Values in Y-axis to realise our sigmoid curve.

# In[ ]:


plt.plot(x,y)
plt.show()


# ## We can observer that , if 'x' is very negative output is almost zero and if 'x'  is very positive its almost one. But when 'x' is 0 , y is 0.5.

# In[ ]:




