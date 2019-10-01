#!/usr/bin/env python
# coding: utf-8

# This notebook contains a first neural network attempt at the Titanic problem. The first half, written by my teammate [Kyle Lang](https://www.kaggle.com/noogakl81), performs some pre-processing on the data. The second half is my attempt at classifying the data using a simple two hidden-layer neural network.
# 
# Some experiments suggest that a range of 32-128 features in the first hidden layer is sufficient. Not yet sure what the impact is of feature count selection in the second hidden layer. With this setup I achieve approximately 96% classification accuracy on the training set and 71% on the test data so there is much room for improvement.
# 
# Outside of more intense data munging (e.g. extracting titles, separating embarkment data) some things I will try next:
# 
# * dropout in the hidden layers - the NN gets stuck at around 96%. This might suggest overfitting. Implementing [dropout](https://www.cs.toronto.edu/%7Ehinton/absps/JMLRdropout.pdf) might help.
# * different neuron activation functions - sigmoid is the vanilla standard. I don't know much about ReLU but people use it and effectively.  Remember to initialize with slightly positive bias in this case.
# * don't bother with sparely connected or partitioned layers - if the problem had some degree of locality to it then maybe. Like convolutions, I don't see a strong argument to attempt this. Prove me wrong.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




