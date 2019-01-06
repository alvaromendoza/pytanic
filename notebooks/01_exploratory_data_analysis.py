
# coding: utf-8

# # Exploratory data analysis of Titanic dataset

# ## IPython magics

# In[65]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Imports

# In[66]:


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

# In[67]:


train = pd.read_csv('../data/raw/train.csv')
train.head(15)


# In[68]:


train.info()
print('train.shape:', train.shape)


# In[69]:


data_dict = pd.read_excel('../references/data_dict.xlsx')
data_dict


# ## Missing values

# In[70]:


importlib.reload(eda)
eda.get_nan_counts(train)


# ## Correlation

# In[71]:


importlib.reload(eda)
eda.association_test(train.loc[:, [
                     'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']], train['Survived'])


# In[72]:


train['SexNum'] = train['Sex'].replace({'male': 1, 'female': 0}).astype(int)
train.head()


# In[73]:


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

# In[74]:


eda.get_count_percentage(train, 'Survived', sort='count')


# In[75]:


sns.countplot(y='Survived', data=train)
plt.gcf().suptitle('Survival count', fontsize=15);


# ## Pclass

# In[76]:


eda.get_count_percentage(train, 'Pclass')


# In[77]:


sns.countplot(y='Pclass', data=train)
plt.gca().set_title('Pclass count', fontsize=15);


# In[78]:


g = sns.catplot(x="Pclass", y="Survived", data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Pclass', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# In[79]:


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

# In[116]:


train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train['Title'], train['Sex'])


# In[117]:


train['Title'].replace('Mlle', 'Miss', inplace=True)
train['Title'].replace('Ms', 'Miss', inplace=True)
train['Title'].replace('Mme', 'Mrs', inplace=True)
pd.crosstab(train['Title'], train['Sex'])


# In[118]:


title_other_filter = ~train['Title'].isin(['Mr', 'Master', 'Mrs', 'Miss'])
train.loc[title_other_filter, 'Title'] = 'Other'
pd.crosstab(train['Title'], train['Sex'])


# In[119]:


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

# In[80]:


eda.get_count_percentage(train, 'Sex')


# In[81]:


sns.countplot(y='Sex', data=train)
plt.gca().set_title('Sex count', fontsize=15);


# In[82]:


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

# In[83]:


eda.get_count_percentage(train, 'SibSp')


# In[84]:


sns.countplot(y='SibSp', data=train)
plt.gca().set_title('SibSp count', fontsize=15);


# In[85]:


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

# In[86]:


eda.get_count_percentage(train, 'Parch')


# In[87]:


sns.countplot(y='Parch', hue='Sex', data=train)
plt.gca().set_title('Parch count', fontsize=15);


# In[88]:


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

# In[120]:


train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
eda.get_count_percentage(train, 'FamilySize')


# In[121]:


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

# In[89]:


eda.get_count_percentage(train, 'Embarked')


# In[90]:


sns.countplot(y='Embarked', hue='Sex', data=train)
plt.gca().set_title('Embarked count vs Sex', fontsize=15);


# In[91]:


sns.countplot(y='Embarked', hue='Pclass', data=train)
plt.gca().set_title('Embarked count vs Pclass', fontsize=15);


# In[92]:


g = sns.catplot(x="Embarked", y="Survived", hue='Sex', data=train, kind="bar", palette="deep", 
                height=4.5, aspect=2.5, orient='v')
g.fig.suptitle('Survival rate vs Embarked vs Sex', fontsize=15)
g.set_ylabels("Survival rate")
g.fig.subplots_adjust(top=.9)
ax = plt.gca()
for p in ax.patches:
    ax.text(p.get_x() + p.get_width()/2., p.get_height()/2.2, '{:0.1f}%'.format(p.get_height() * 100), 
            fontsize=13, ha='center', va='bottom')


# In[93]:


g = sns.FacetGrid(train, row='Embarked', col='Pclass', height=2.2, aspect=1.9)
g.map(sns.barplot, 'Sex', 'Survived', alpha=0.8, order=['female', 'male'])
g.fig.subplots_adjust(top=.9)
plt.gcf().suptitle('Survival rate vs Sex vs Embarked vs Pclass', fontsize=15);


# ## Deck

# In[94]:


has_cabin = train.loc[~train['Cabin'].isnull(), :]
has_cabin.head()


# In[95]:


eda.get_count_percentage(has_cabin, 'Pclass')


# In[96]:


eda.get_count_percentage(has_cabin, 'Sex')


# In[97]:


deck = train['Cabin'].apply(lambda x: ''.join(re.findall("[a-zA-Z]+", str(x))))
deck.value_counts()


# In[98]:


train['Deck'] = train['Cabin'].str.extract(r'([A-Z])+', expand=False)
train['Deck'].fillna('X', inplace=True)
train['Deck'].value_counts()


# In[99]:


sns.countplot(x='Deck', data=train)
plt.gca().set_title('Deck count', fontsize=15);


# In[100]:


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

# In[101]:


sns.distplot(train['Age'][~train['Age'].isnull()])
plt.gca().set_title('Distplot of Age', fontsize=15);


# In[102]:


g = sns.FacetGrid(train, col='Survived', height=4, aspect=1.5)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.85)
plt.gcf().suptitle('Distplot of Age vs Survived', fontsize=15);


# In[103]:


g = sns.FacetGrid(train, row='Sex', col='Survived', height=3, aspect=2)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.87)
plt.gcf().suptitle('Distplot of Age vs Survived vs Sex', fontsize=15);


# In[104]:


g = sns.FacetGrid(train, row='Pclass', col='Survived', height=3, aspect=2)
g = g.map(sns.distplot, "Age")
g.fig.subplots_adjust(top=.91)
plt.gcf().suptitle('Distplot of Age vs Survived vs Pclass', fontsize=15);


# In[105]:


train['Age'].describe()


# ## AgeBin

# In[106]:


train['Age'].fillna(train['Age'].median(), inplace=True)
train['AgeBin'] = pd.cut(train['Age'], 10)
eda.get_count_percentage(train, 'AgeBin')


# In[107]:


sns.countplot(x='AgeBin', data=train)
plt.gca().set_title('AgeBin count', fontsize=15);


# In[108]:


sns.barplot(x='AgeBin', y='Survived', data=train)
plt.gca().set_ylabel('Survival rate')
plt.gca().set_title('Survival rate vs AgeBin', fontsize=15);


# ## Fare

# In[109]:


train['Fare'].describe()


# In[110]:


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[15, 4])
sns.distplot(train['Fare'], ax=ax[0])
sns.distplot(np.log1p(train['Fare']), ax=ax[1], axlabel='Log1p Fare')
fig.suptitle('Distplots of Fare vs Log1p Fare');


# In[111]:


fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=[15, 6])
sns.stripplot(y='Pclass', x='Fare', data=train, ax=ax[0], orient='h', s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
sns.boxenplot(y='Pclass', x='Fare', data=train, ax=ax[1], orient='h')
ax[0].set_xlabel('')
ax[0].set_title('Stripplot of Fare vs Pclass', fontsize=15)
ax[1].set_title('Boxenplot of Fare vs Pclass', fontsize=15);


# In[112]:


g = sns.catplot(x='SibSp', y='Fare', col='Pclass', data=train, kind='strip', 
                sharey=False, height=4, aspect=1, s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
g.fig.suptitle('Fare vs SibSp vs Pclass')
g.fig.subplots_adjust(top=.85)
sns.catplot(x='SibSp', y='Fare', col='Pclass', data=train, kind='boxen', 
                sharey=False, height=4, aspect=1);


# In[113]:


g = sns.catplot(x='Parch', y='Fare', col='Pclass', data=train, kind='strip', 
                sharey=False, height=4, aspect=1, s=8, edgecolor='white', 
                linewidth=0.6, jitter=0.3)
g.fig.suptitle('Stripplot of Fare vs Parch vs Pclass')
g.fig.subplots_adjust(top=.85)
sns.catplot(x='Parch', y='Fare', col='Pclass', data=train, kind='boxen', 
                sharey=False, height=4, aspect=1);


# ## Save as .py

# In[114]:


get_ipython().system('jupyter nbconvert --to script 01_exploratory_data_analysis.ipynb')

