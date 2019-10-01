#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().magic(u'matplotlib inline')


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


# I'm curious to see whether I can do interactive visualizations in the kaggle notebook. I can
# add interactivity to a local jupyter notebook using the `ipywidgets` module. I can store the
# notebook on github, but the rendering is static and the widgets don't display. If that's not
# the case here, I could use Kaggle as a platform for hosting one-off interactive notebooks, rather
# than rolling a `tmpnb` solution.

# In[ ]:


import ipywidgets


# Okay, ipywidgets is there. Now, can we use it?

# In[ ]:


from ipywidgets import interactive
from scipy.stats import gamma
import numpy as np
from IPython.display import display, clear_output

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[ ]:


get_ipython().magic(u'config InlineBackend.close_figures=False')

x = np.arange(0, 40, 0.005)
def gamma_plot_setup():
    shape = 5
    scale = 0.5
    fig, ax = plt.subplots()
    y = gamma.pdf(x, shape, scale=scale)
    line = ax.plot(x, y)
    ax.set_ylim((0,0.5))
    return fig, ax, line[0]

def gamma_update(shape, scale):
    y = gamma.pdf(x, shape, scale=scale)
    line.set_ydata(y)
    fig.canvas.draw()
    #clear_output(wait=True)
    display(fig)


# In[ ]:


fig, ax, line = gamma_plot_setup()


# In[ ]:


v = interactive(gamma_update, shape=(0.1, 10.0), scale=(0.3, 3.0))
display(v)


# The interactive widget (a pair sliders in this case) doesn't get displayed. So, this won't work the
# way I wanted it to.

# In[ ]:




