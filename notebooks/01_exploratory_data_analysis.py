
# coding: utf-8

# # Exploratory data analysis of Titanic dataset

# ## IPython magics

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Imports

# In[2]:


import importlib
import os
import time
import re
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

# In[3]:


train = pd.read_csv('../data/raw/train.csv')
train.head(15)


# In[4]:


train.info()
print('train.shape:', train.shape)


# In[5]:


data_dict = pd.read_excel('../references/data_dict.xlsx')
data_dict


# ## Missing values

# In[6]:


importlib.reload(eda)
eda.get_nan_counts(train)


# ## Survived

# In[10]:


eda.get_count_percentage(train, 'Survived', sort='count')


# In[11]:


sns.countplot(y='Survived', data=train)
plt.gcf().suptitle('Survival count', fontsize=15);


# ## Pclass

# In[12]:


eda.get_count_percentage(train, 'Pclass')


# In[13]:


sns.countplot(y='Pclass', data=train)
plt.gca().set_title('Pclass count', fontsize=15);


# In[14]:


g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Pclass', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# In[15]:


g = sns.catplot(x="Pclass", y="Survived", hue='Sex', data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5)
g.fig.suptitle('Survival rate vs Pclass vs Sex', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Title

# In[16]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[17]:


train['Title'].replace('Mlle', 'Miss', inplace=True)
train['Title'].replace('Ms', 'Miss', inplace=True)
train['Title'].replace('Mme', 'Mrs', inplace=True)
pd.crosstab(train['Title'], train['Sex'])


# In[18]:


title_other_filter = ~train['Title'].isin(['Mr', 'Master', 'Mrs', 'Miss'])
train.loc[title_other_filter, 'Title'] = 'Other'
pd.crosstab(train['Title'], train['Sex'])


# In[19]:


g = sns.catplot(x="Title", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Title', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Sex

# In[20]:


eda.get_count_percentage(train, 'Sex')


# In[21]:


sns.countplot(y='Sex', data=train)
plt.gca().set_title('Sex count', fontsize=15);


# In[22]:


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

# In[23]:


eda.get_count_percentage(train, 'SibSp')


# In[24]:


sns.countplot(y='SibSp', data=train)
plt.gca().set_title('SibSp count', fontsize=15);


# In[25]:


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

# In[26]:


eda.get_count_percentage(train, 'Parch')


# In[27]:


sns.countplot(y='Parch', hue='Sex', data=train)
plt.gca().set_title('Parch count', fontsize=15);


# In[28]:


g = sns.catplot(x="Parch", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Parch', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## FamilySize

# In[29]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
eda.get_count_percentage(train, 'FamilySize')


# In[30]:


g = sns.catplot(x="FamilySize", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs FamilySize', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Embarked

# In[31]:


eda.get_count_percentage(train, 'Embarked')


# In[32]:


sns.countplot(y='Embarked', hue='Sex', data=train)
plt.gca().set_title('Embarked count vs Sex', fontsize=15);


# In[33]:


sns.countplot(y='Embarked', hue='Pclass', data=train)
plt.gca().set_title('Embarked count vs Pclass', fontsize=15);


# In[34]:


g = sns.catplot(x="Embarked", y="Survived", hue='Sex', data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Embarked vs Sex', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# In[35]:


g = sns.FacetGrid(train, row='Embarked', col='Pclass', height=2.2, aspect=1.9)
g.map(sns.barplot, 'Sex', 'Survived', alpha=0.8, order=['female', 'male'])
g.fig.subplots_adjust(top=.9)
plt.gcf().suptitle('Survival rate vs Sex vs Embarked vs Pclass', fontsize=15);


# ## Deck

# In[36]:


has_cabin = train.loc[~train['Cabin'].isnull(), :]
has_cabin.head()


# In[37]:


eda.get_count_percentage(has_cabin, 'Pclass')


# In[38]:


eda.get_count_percentage(has_cabin, 'Sex')


# In[39]:


deck = train['Cabin'].apply(lambda x: ''.join(re.findall("[a-zA-Z]+", str(x))))
deck.value_counts()


# In[40]:


train['Deck'] = train['Cabin'].str.extract(r'([A-Z])+', expand=False)
train['Deck'].fillna('X', inplace=True)
train['Deck'].value_counts()


# In[41]:


sns.countplot(x='Deck', data=train)
plt.gca().set_title('Deck count', fontsize=15);


# In[42]:


g = sns.catplot(x="Deck", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Deck', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# ## Age

# In[43]:


sns.distplot(train['Age'][~train['Age'].isnull()])
plt.gca().set_title('Distplot of Age', fontsize=15);


# In[44]:


g = sns.FacetGrid(train, col='Survived', height=4, aspect=1.5)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.85)
plt.gcf().suptitle('Distplot of Age vs Survived', fontsize=15);


# In[45]:


g = sns.FacetGrid(train, row='Sex', col='Survived', height=3, aspect=2)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.87)
plt.gcf().suptitle('Distplot of Age vs Survived vs Sex', fontsize=15);


# In[46]:


g = sns.FacetGrid(train, row='Pclass', col='Survived', height=3, aspect=2)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.91)
plt.gcf().suptitle('Distplot of Age vs Survived vs Pclass', fontsize=15);


# In[47]:


train['Age'].describe()


# ## AgeBin

# In[48]:


train['Age'].fillna(train['Age'].median(), inplace=True)
train['AgeBin'] = pd.cut(train['Age'], 10)
eda.get_count_percentage(train, 'AgeBin')


# In[49]:


sns.countplot(x='AgeBin', data=train)
plt.gca().set_title('AgeBin count', fontsize=15);


# In[50]:


sns.barplot(x='AgeBin', y='Survived', data=train)
plt.gca().set_ylabel('Survival rate')
plt.gca().set_title('Survival rate vs AgeBin', fontsize=15);


# ## Fare

# In[51]:


train['Fare'].describe()


# In[52]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 4])
sns.distplot(train['Fare'], ax=ax[0])
sns.distplot(np.log1p(train['Fare']), ax=ax[1], axlabel='Log1p Fare')
fig.suptitle('Distplots of Fare vs Log1p Fare');


# In[53]:


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[15, 6])
sns.stripplot(y='Pclass', x='Fare', data=train, ax=ax[0], orient='h', s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
sns.boxenplot(y='Pclass', x='Fare', data=train, ax=ax[1], orient='h')
ax[0].set_xlabel('')
ax[0].set_title('Stripplot of Fare vs Pclass', fontsize=15)
ax[1].set_title('Boxenplot of Fare vs Pclass', fontsize=15);


# In[54]:


g = sns.catplot(x='SibSp', y='Fare', col='Pclass', data=train, kind='strip', 
                sharey=False, height=4, aspect=1, s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
g.fig.suptitle('Fare vs SibSp vs Pclass')
g.fig.subplots_adjust(top=.85)
sns.catplot(x='SibSp', y='Fare', col='Pclass', data=train, kind='boxen', 
                sharey=False, height=4, aspect=1);


# In[55]:


g = sns.catplot(x='Parch', y='Fare', col='Pclass', data=train, kind='strip', 
                sharey=False, height=4, aspect=1, s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
g.fig.suptitle('Stripplot of Fare vs Parch vs Pclass')
g.fig.subplots_adjust(top=.85)
sns.catplot(x='Parch', y='Fare', col='Pclass', data=train, kind='boxen', 
                sharey=False, height=4, aspect=1);


# ## Correlation

# In[57]:


importlib.reload(eda)
eda.association_test(train.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp',
                                   'Parch', 'FamilySize', 'Fare', 'Embarked']], train['Survived'])


# In[59]:


train['SexNum'] = train['Sex'].replace({'male': 1, 'female': 0}).astype(int)
train['SexNum'].head()


# In[9]:


CORR_COLS = ['Survived', 'Pclass', 'Fare', 'SibSp', 'FamilySize', 'Parch', 'Age', 'SexNum']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
sns.heatmap(train[CORR_COLS].corr(method='pearson'), 
            ax=ax[0], cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
sns.heatmap(train[CORR_COLS].corr(method='spearman'), 
            ax=ax[1], cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 12})
ax[0].set_title("Pearson's r", fontsize=15)
ax[1].set_title("Spearman's rho", fontsize=15)
fig.suptitle("Correlation matrices", fontsize=15)
fig.subplots_adjust(top=.92);


# ## Save as .py

# In[56]:


get_ipython().system('jupyter nbconvert --to script 01_exploratory_data_analysis.ipynb')

