
# coding: utf-8

# # Exploratory data analysis of Titanic dataset

# ## Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[20]:


import importlib
import os
import time
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns; sns.set()
import titanic.exploratory_data_analysis as eda
importlib.reload(eda)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.max_colwidth = -1

plt.rcParams['figure.figsize'] = [15, 4.5]


# ## Load data

# In[21]:


train = pd.read_csv('../data/raw/train.csv')
train.head(15)


# In[22]:


train.info()
print('train.shape:', train.shape)


# In[23]:


data_dict = pd.read_excel('../references/data_dict.xlsx')
data_dict


# ## Missing values

# In[24]:


importlib.reload(eda)
eda.get_nan_counts(train)


# ## Correlation

# In[25]:


importlib.reload(eda)
eda.association_test(train.loc[:, [
                     'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], train['Survived'])


# In[26]:


train['SexNum'] = train['Sex'].replace({'male': 1, 'female': 0}).astype(int)
train.head()


# In[27]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
sns.heatmap(train[['Survived', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Age', 'SexNum']].corr(method='pearson'), 
            ax=ax[0], cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
sns.heatmap(train[['Survived', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Age', 'SexNum']].corr(method='spearman'), 
            ax=ax[1], cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
ax[0].set_title("Pearson's r", fontsize=15)
ax[1].set_title("Spearman's rho", fontsize=15)
fig.suptitle("Correlation matrices", fontsize=15)
fig.subplots_adjust(top=.92);


# ## Survived

# In[28]:


eda.get_count_percentage(train, 'Survived', sort='count')


# In[29]:


sns.countplot(y='Survived', data=train)
plt.gcf().suptitle('Survival count', fontsize=15);


# ## Pclass

# In[30]:


eda.get_count_percentage(train, 'Pclass')


# In[48]:


sns.countplot(y='Pclass', data=train)
plt.gca().set_title('Pclass count', fontsize=15);


# In[38]:


g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Pclass', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# In[33]:


g = sns.catplot(x="Pclass", y="Survived", hue='Sex', data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5)
g.fig.suptitle('Survival rate vs Pclass vs Sex', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Sex

# In[39]:


eda.get_count_percentage(train, 'Sex')


# In[50]:


sns.countplot(y='Sex', data=train)
plt.gca().set_title('Sex count', fontsize=15);


# In[42]:


g = sns.catplot(x="Sex", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Sex', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## SibSp

# In[44]:


eda.get_count_percentage(train, 'SibSp')


# In[51]:


sns.countplot(y='SibSp', data=train)
plt.gca().set_title('SibSp count', fontsize=15);


# In[43]:


g = sns.catplot(x="SibSp", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs SibSp', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Parch

# In[52]:


eda.get_count_percentage(train, 'Parch')


# In[59]:


sns.countplot(y='Parch', hue='Sex', data=train)
plt.gca().set_title('Parch count', fontsize=15);


# In[56]:


g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Parch', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Embarked

# In[60]:


eda.get_count_percentage(train, 'Embarked')


# In[66]:


sns.countplot(y='Embarked', hue='Sex', data=train)
plt.gca().set_title('Embarked count vs Sex', fontsize=15);


# In[67]:


sns.countplot(y='Embarked', hue='Pclass', data=train)
plt.gca().set_title('Embarked count vs Pclass', fontsize=15);


# In[70]:


g = sns.catplot(x="Embarked", y="Survived", hue='Sex', data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Embarked vs Sex', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# In[90]:


g = sns.FacetGrid(train, row='Embarked', col='Pclass', height=2.2, aspect=1.9)
g.map(sns.barplot, 'Sex', 'Survived', alpha=0.8, order=['female', 'male'])
g.fig.subplots_adjust(top=.9)
plt.gcf().suptitle('Survival rate vs Sex vs Embarked vs Pclass', fontsize=15);


# ## Age

# In[111]:


sns.distplot(train['Age'][~train['Age'].isnull()])
plt.gca().set_title('Distplot of Age', fontsize=15);


# In[107]:


g = sns.FacetGrid(train, col='Survived', height=4, aspect=1.5)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.85)
plt.gcf().suptitle('Distplot of Age vs Survived', fontsize=15);


# In[117]:


g = sns.FacetGrid(train, row='Sex', col='Survived', height=3, aspect=2)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.87)
plt.gcf().suptitle('Distplot of Age vs Survived vs Sex', fontsize=15);


# In[122]:


g = sns.FacetGrid(train, row='Pclass', col='Survived', height=3, aspect=2)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.91)
plt.gcf().suptitle('Distplot of Age vs Survived vs Pclass', fontsize=15);


# In[131]:


train['Age'].describe()


# ## AgeBin

# In[139]:


train['Age'].fillna(train['Age'].median(), inplace=True)
train['AgeBin'] = pd.cut(train['Age'], 10)
eda.get_count_percentage(train, 'AgeBin')


# In[141]:


sns.countplot(x='AgeBin', data=train)
plt.gca().set_title('AgeBin count', fontsize=15);


# In[138]:


sns.barplot(x='AgeBin', y='Survived', data=train)
plt.gca().set_ylabel('Survival rate')
plt.gca().set_title('Survival rate vs AgeBin', fontsize=15);


# ## Fare

# In[144]:


train['Fare'].describe()


# In[150]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 4])
sns.distplot(train['Fare'], ax=ax[0])
sns.distplot(np.log1p(train['Fare']), ax=ax[1], axlabel='Log1p Fare')
fig.suptitle('Distplots of Fare vs Log1p Fare');


# In[214]:


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[15, 6])
sns.stripplot(y='Pclass', x='Fare', data=train, ax=ax[0], orient='h', s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
sns.boxenplot(y='Pclass', x='Fare', data=train, ax=ax[1], orient='h')
ax[0].set_xlabel('')
ax[0].set_title('Stripplot of Fare vs Pclass', fontsize=15)
ax[1].set_title('Boxenplot of Fare vs Pclass', fontsize=15);


# In[217]:


g = sns.catplot(x='SibSp', y='Fare', col='Pclass', data=train, kind='strip', 
                sharey=False, height=4, aspect=1, s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
g.fig.suptitle('Fare vs SibSp vs Pclass')
g.fig.subplots_adjust(top=.85)
sns.catplot(x='SibSp', y='Fare', col='Pclass', data=train, kind='boxen', 
                sharey=False, height=4, aspect=1);


# In[218]:


g = sns.catplot(x='Parch', y='Fare', col='Pclass', data=train, kind='strip', 
                sharey=False, height=4, aspect=1, s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
g.fig.suptitle('Stripplot of Fare vs Parch vs Pclass')
g.fig.subplots_adjust(top=.85)
sns.catplot(x='Parch', y='Fare', col='Pclass', data=train, kind='boxen', 
                sharey=False, height=4, aspect=1);

