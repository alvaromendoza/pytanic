
# coding: utf-8

# # Compare cross-validated models

# ## IPython magics

# In[22]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Imports

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import titanic.tools as tools

from pprint import pprint
from titanic.modelling import ExtendedClassifier

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = -1

plt.rcParams['figure.figsize'] = [15, 4.5]


# ## Deserialize models

# In[24]:


models = dict()
models['LogisticRegression'] = ExtendedClassifier.deserialize(r'../models/logreg.pickle')
models['RandomForestClassifier'] = ExtendedClassifier.deserialize(r'../models/forest.pickle')
models['SVC'] = ExtendedClassifier.deserialize(r'../models/svc.pickle')
models


# ## Compare models

# In[25]:


scores = {name: model.cvs_stamp['score'] for name, model in models.items()}
scores


# In[26]:


list(scores.keys())


# In[27]:


fig = plt.figure(figsize=[15, 5])
ax = fig.add_subplot(111)
sns.barplot(x=list(scores.keys()), y=list(scores.values()), orient='v', ax=ax)
ax.set_ylim(bottom=0.8)
ax.set_title('Cross-validated accuracy of candidate models', fontsize=15)
for p in ax.patches:
    ax.annotate("%.5f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=13, color='k', alpha=0.8,
                xytext=(0, 20), textcoords='offset points')
plt.show()


# ## Save as .py

# In[29]:


get_ipython().system('jupyter nbconvert --to script 02_compare_models.ipynb')

