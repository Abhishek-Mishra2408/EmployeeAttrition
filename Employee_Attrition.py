#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io


# In[2]:


# !gdown 16KtxSt_QEGQvfluEaMls5cCHPwhRXgCk


# In[3]:


df = pd.read_csv("HR-Employee-Attrition.csv")
df.info()


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.hist(figsize = (20,20))
plt.show()


# In[7]:


df.drop(['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18'], axis=1, inplace=True)


# In[8]:


df


# In[9]:


df.info()


# In[10]:


def unique_vals(col):
    if col.dtype == "object":
        print(f'{col.name}: {col.nunique()}')

df.apply(lambda col: unique_vals(col))


# In[11]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Create a label encoder object
le = LabelEncoder()
def label_encode(ser):
    if ser.dtype=="object" and ser.nunique() <= 2:
        print(ser.name)
        le.fit(ser)
        ser = le.transform(ser)
    return ser

df = df.apply(lambda col: label_encode(col))


# In[12]:


# convert rest of categorical variable into dummy
df = pd.get_dummies(df, columns = ["BusinessTravel", "Department", "MaritalStatus"], drop_first = True)


# In[13]:


df


# # Lets analyse the target feature now

# In[14]:


target = df['Attrition'].copy()
df = df.drop(["Attrition"], axis = 1)
type(target)


# In[15]:


target.value_counts()


# # Note :- The dataset is extremely imbalanced
# so we will use SMOTE oversampling technique to balance the data.
But SMOTE is only applied to training data. so first we will split the dataset first and then we will use SMOTE.
# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df,target,test_size=0.25,random_state=7,stratify=target)
print("Number transactions X_train dataset: ", X_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", X_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)


# In[17]:


len(X_train.columns)


# # Now we will first perform target encoding

# In[18]:


# !pip install category_encoders


# In[19]:


import category_encoders as ce
ce_target = ce.TargetEncoder(cols= ['EducationField','JobRole'])
X_train = ce_target.fit_transform(X_train,y_train)
X_test = ce_target.transform(X_test)


# In[20]:


from imblearn.over_sampling import SMOTE
from collections import Counter
smt = SMOTE()
X_sm,y_sm = smt.fit_resample(X_train,y_train)
print('Resampled dataset shape {}'.format(Counter(y_sm)))


# In[21]:


X_sm.shape


# In[22]:


X_sm


# # Preprocessed data

# In[23]:


# !gdown 19L3rYatfhbBL1r5MHrv-p_oM2wlvrhqk
# !gdown 1OHLKJwA3qZopKPvlKoRldM6BvA1A4dYF
# !gdown 1N7O_fWCTJLu8SIa_paKcDEzllgpMk8sK
# !gdown 12Bh2AN8LcZAlg20ehpQrEWccUDaSdsOG


# In[24]:


import pickle
# Load data (deserialize)
with open('preprocessed_X_sm.pickle', 'rb') as handle:
    X_sm = pd.read_pickle(handle)

with open('X_test.pickle', 'rb') as handle:
    X_test = pd.read_pickle(handle)

with open('y_sm.pickle', 'rb') as handle:
    y_sm = pd.read_pickle(handle)

with open('y_test.pickle', 'rb') as handle:
    y_test = pd.read_pickle(handle)


# # Use Logistic Regression

# Logistic Regression Works bettter with Linearly seperable data.
# This is a Binary class classification

# In[25]:


import sklearn
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_sm,y_sm)
clf.score(X_sm,y_sm)


# # KNN 

# In[26]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
neigh = KNeighborsClassifier(n_neighbors=10)
cross_val_score(neigh,X_sm,y_sm,cv=5)


# # Decision Tree From Scratch

# In[27]:


def calculate_weighted_entropy(feature,y):
    categories = feature.unique()

    weighted_entropy = 0

    for category in categories:
        y_category = y[feature == category]
        entropy_category = entropy_df(y_category)
        weighted_entropy += y_category.shape[0]/y.shape[0]*entropy_category

    
    return weighted_entropy


# In[28]:


def entropy_df(y):
    print(y)
    probs = y.value_counts()/y.shape[0]
    entropy = np.sum(-probs * np.log2(probs + 1e-9)) # adding delta 1e-9 in case p = 0 as log(0) is not defined

    return(entropy)


# In[29]:


def information_gain_entropy(feature,y):
    parent_entropy = entropy_df(y)

    child_entropy = calculate_weighted_entropy(feature,y)

    ig = parent_entropy - child_entropy

    return ig


# # Gini Impurity

# In[30]:


def gini_impurity(y):
    p = y.value_counts()/y.shape[0]
    gini = 1-np.sum(p**2)
    return gini


# In[31]:


def calculate_weighted_gini(feature, y):
    categories = feature.unique()

    weighted_gini_impurity = 0

    for category in categories:
        y_category = y[feature == category]
        gini_impurity_category = gini_impurity(y_category)
        # print(category)
        # print(gini_impurity_category)
        weighted_gini_impurity += y_category.shape[0]/y.shape[0]*gini_impurity_category

    
    return weighted_gini_impurity


# In[32]:


def information_gain_gini(feature,y):
    parent_gini = gini_impurity(y)

    child_gini = calculate_weighted_gini(feature,y)

    ig = parent_gini - child_gini

    return ig


# In[33]:


X_sm


# In[34]:


X_sub = pd.concat([X_sm["JobLevel"], X_sm["Gender"], X_sm["Education"]], axis=1)
X_sub


# In[35]:


entropy_root = entropy_df(y_sm)
entropy_root


# In[ ]:





# In[36]:


gini_root = gini_impurity(y_sm)
gini_root


# In[37]:


mxIG = 0
splitFeatureEntropy = ""

mxGiniReduction = 0
splitFeatureGini = ""

for feature in X_sub.columns[:]:
    infoGain = information_gain_entropy(X_sub[feature], y_sm)
    giniReduction = information_gain_gini(X_sub[feature], y_sm)  
    print(f'{feature}:\n\tEntropy: {infoGain}\n\tGini Impurity: {giniReduction}\n')
    if infoGain > mxIG:
        splitFeatureEntropy = feature
        mxIG = infoGain

    if giniReduction > mxGiniReduction:
        splitFeatureGini = feature
        mxGiniReduction = giniReduction


# Note :- So which feature should we use to split the root node ,Wrt entropy

# In[44]:


splitFeatureEntropy


# In[45]:


splitFeatureGini


# In[47]:


X_sub


# In[48]:


X_sub[splitFeatureEntropy].unique()


# In[51]:


unq_vals = X_sub[splitFeatureEntropy].unique()
children = 1

for i in unq_vals :
    print(f'Child {children}:\n')
    children += 1

    print(y_sm[X_sub[splitFeatureEntropy] == i].value_counts())
    
    print()


# In[ ]:




