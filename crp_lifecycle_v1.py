#!/usr/bin/env python
# coding: utf-8

# # Feature Pre-Processing Churn Risk Predictor
# <img src="http://ran-tools.vodafone.com.au/tools/coras/cmodel/aa_pipe.png" alt="Advanced Analytics Pipeline" title="Advanced Analytics Pipeline" />
# 
# ***
#     @author: Jonny Keane
#     @created: 19/08/2019
#     @updated: 25/09/2019
#     @version: 0.9
#     @latest:
#     - Added y_pred and y_proba for full dataset
#     @resources:
#     - -
#     @to do:
#     - -
# 
# ***
# ## Objective: Create a pipeline phase to perform data cleansing, scaling, sampling and correlation validation so that the data can be passed to the model factory for classification
# 
# 
# The **steps** taken in the phase are as follows:
# 1. [Import packages](#import-packages) to support analysis.
# 2. [Import the data](#import-data) in the form of a csv flat file.
# 
# <img src="http://ran-tools.vodafone.com.au/tools/coras/cmodel/New_VF_Icon_CMYK_RED.png" align="right" alt="Vodafone Hutchison Australia" title="Vodafone Hutchison Australia" />  

# In[1]:


# System
import os
import sys
import time
import random
import re

# Data Processing
import pandas as pd
import itertools

# Linear Algebra
import numpy as np

# Data Visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
from pandas.plotting import scatter_matrix

 # Scaling and Sampling
import imblearn
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from matplotlib.ticker import PercentFormatter

# Accuracy Measures
from sklearn import model_selection
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import roc_curve, precision_recall_curve, recall_score, precision_score, roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix, zero_one_loss
from yellowbrick.features.importances import FeatureImportances
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import shap

# Feature Selection
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import xgboost as xgb
import lightgbm as lgb
import catboost as cat
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# File management and databases
import csv
#import teradatasql
from zipfile import ZipFile
from urllib.request import urlopen
import fnmatch
import pathlib

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

print("Packages Imported")


# In[2]:


na = ('','#N/A','#N/A N/A','#NA','-1.#IND','-1.#QNAN','-NaN','-nan','1.#IND','1.#QNAN','N/A',
      'NA','NULL','NaN','n/a','nan','null','NAN','?','?\r')

subscribers = pd.read_csv("/vha/home/61072380/Subscriber_Features_Dev-TAB.txt", 
                         sep='\t', lineterminator='\n', na_values=na, error_bad_lines=False)
    
print("Initial Dataset Imported")


# In[3]:


# Remove unwanted characters in feature name
subscribers.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', ' ', '/', '+', '<'))) else col for col in subscribers.columns.values]
subscribers.columns = map(str.lower, subscribers.columns)
list(subscribers.columns)


# In[4]:


subscribers.head(10)


# In[5]:


pd.DataFrame(subscribers['contract_group'].value_counts())


# In[6]:


'''
contract_group_sum = subscribers.groupby(['contract_group']).agg({'Vol_Churn_Flag': 'sum', 'Portout_Flag': 'sum',
                                                                  'Discon_Flag': 'sum', 'Invol_Churn_Flag': 'sum'})
contract_group_pcts = contract_group_sum.apply(lambda x: round(100 * x / float(x.sum()), 1))
contract_group_pcts.sort_values(by=['Vol_Churn_Flag'], ascending=False)
'''
contract_group_sum = subscribers.groupby(['contract_group']).agg({'upgrade_flag': 'sum', 'portout_flag': 'sum'})
contract_group_pcts = contract_group_sum.apply(lambda x: round(100 * x / float(x.sum()), 1))
contract_group_pcts.sort_values(by=['portout_flag'], ascending=False)


# In[7]:


#subscribers.groupby('Contract_Group')[['Vol_Churn_Flag', 'Portout_Flag', 'Discon_Flag', 'Invol_Churn_Flag']].sum().sort_values(by=['Vol_Churn_Flag'], ascending=False)
subscribers.groupby('contract_group')[['upgrade_flag',
                                       'portout_flag']].sum().sort_values(by=['portout_flag'], ascending=False)


# In[8]:


dtype_df = subscribers.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()


# In[9]:


list(subscribers.columns[subscribers.dtypes=='object'])


# In[10]:


if subscribers.id_subs_id.nunique() == subscribers.shape[0]:
    print('id_subs_id is unique')
    subscribers = subscribers.set_index('id_subs_id')
else:
    print('id_subs_id is not unique')
    values = pd.DataFrame(subscribers['id_subs_id'].value_counts())
    print(values[values['id_subs_id']>1])
    subscribers = subscribers[~subscribers['id_subs_id'].isin(list(values[values['id_subs_id']>1].index))]

datasetHasNan = False
if subscribers.count().min() == subscribers.shape[0]:
    print('No missing values present') 
else:
    datasetHasNan = True
    print('NaN values present')


# In[11]:


# Save the full dataset incase we reduce it
subscribers_full = subscribers.copy()
print("Dataset copied")


# In[12]:


# Revert to full dataset
subscribers = subscribers_full.copy()
print("Dataset reset")


# In[13]:


threshold = 0.25

# Check for missing data
if datasetHasNan == True:
    nas = pd.concat([subscribers.isnull().sum()], axis=1, keys=['NaN values %'], sort=True)
    to_drop = list(nas[nas.sum(axis=1) > subscribers.shape[0]*threshold].index)
    print("NaN in the data set >", round(100*threshold),"%:", len(to_drop), "features\n", to_drop)


# In[14]:


# Address missing data
for col in subscribers.columns:
    try:
        subscribers[col].fillna(subscribers[col].median(), inplace=True)
    except:
        subscribers[col] = subscribers[col].fillna("Unknown")
        
        subscribers[col] = subscribers[col].str.replace('+', '__ps__')
        subscribers[col] = subscribers[col].str.replace('\\r', '')
        subscribers[col] = subscribers[col].str.replace('\\', '__bs__')
        subscribers[col] = subscribers[col].str.replace('/', '__fs__')
        subscribers[col] = subscribers[col].str.replace('.', '__dt__')
        subscribers[col] = subscribers[col].str.replace('[', '__lb__')
        subscribers[col] = subscribers[col].str.replace(']', '__rb__')
        subscribers[col] = subscribers[col].str.replace('(', '__lp__')
        subscribers[col] = subscribers[col].str.replace(')', '__rp__')
        subscribers[col] = subscribers[col].str.replace('>', '__gr__')
        subscribers[col] = subscribers[col].str.replace('<', '__lt__')
    
    if "\r" in col:
        subscribers = subscribers = subscribers.rename(columns={col: col.replace('\r', '')})


# In[15]:


subscribers.head(10)


# In[16]:


# Check if only one value is present for all subscribers
for col in subscribers.columns:
    if len(subscribers[col].value_counts())==1:
        to_drop.append(col)

print(to_drop)


# In[17]:


def remove_flags(cols, a, lifecycle, verbose=0):
    
    # remove subscriber GBs as they are incorrect
    a.extend(['subscriber_gb', 'bonus_gb'])

    # remove churn flags
    a.extend(['portout_flag', 'upgrade_flag', 'contract_group'])

    a.extend([col for col in cols if 'campaign' in col])
    a.extend(['noof_com_campaigncodes', 'contact_commercial'])
    
    if verbose>0 and len(list(set(a)))>0:
        print("Columns to drop:", len(list(set(a))), "\n", list(set(a)))

    return list(set(a))

print("remove_flags() ready")


# In[18]:


def rando_under(df, to_drop, flag, verbose=0):
    
    a = []
    
    # As the dataset is biased towards non-churners we must perform sampling
    y = df[flag].copy()
    X = df[:].copy()
    X = X.reset_index(drop=False)

    for n in range(len(to_drop)):
        try:
            X.drop(to_drop[n], axis=1, inplace=True)
        except:
            a.append(to_drop[n])
    
    if verbose>0 and len(a)>0:
        print("\nUnable to drop:\n", list(set(a)))

    rus = RandomUnderSampler(ratio='majority', random_state=42, replacement=False)
    X_rus, y_rus = rus.fit_sample(X, y)
    
    df_rus = pd.DataFrame(X_rus, columns=X.columns)
    df_rus[flag] = y_rus
    df_rus = df_rus.set_index('id_subs_id', drop=True)

    return df_rus

print("rando_under() ready")


# In[19]:


def dtypes_consistency(orig, sampled):
    array_int = orig.select_dtypes(include=['int', 'int64']).columns
    array_float = orig.select_dtypes(include=['float', 'float64']).columns
    array_object = orig.select_dtypes(include=['object']).columns

    sampled[array_int] = sampled[array_int].astype('int64')
    sampled[array_float] = sampled[array_float].astype('float64')
    sampled[array_object] = sampled[array_object].astype('object')
    
    return sampled

print("dtypes_consistency() ready")


# In[20]:


def dtype_split(df, verbose=0):
    try:
        a = df.select_dtypes(include=['object', 'category']).copy()
        if verbose>0:
            print("\nCategories Dataframe created")
    except:
        print("\nNo categories or objects found")

    try:
        _ = list(df.dtypes[(df.dtypes != 'object') & (df.dtypes != 'category')].index)
        b = df[_].copy()
        if verbose>0:
            print("Numerical Dataframe created")
    except:
        print("No numerics found")
    
    return a, b

print("dtype_split() ready")


# In[21]:


def run_correlation(df, verbose=0, threshold=0.5):
    
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    if verbose>0:
        print("\nCorrelation Matrix built\n")

    primary = []
    correlated = []

    for col in upper.columns:
        checks = pd.DataFrame(upper[col]>=threshold)
        if checks[checks[col]==True].shape[0]>0:
            for check in checks[checks[col]==True].index:
                if col not in primary and col not in correlated:
                    primary.append(col)
                elif check not in correlated:
                    correlated.append(check)
                    
                    if verbose>0:
                        print("Removed", check, "due to high correlation with", col)
        
    return df.drop(correlated, axis=1)

print("run_correlation() ready")


# In[89]:


def remove_zereos(df, flag, verbose=0, threshold=0.8):
    a = []
    for col in df.columns:
        vals = pd.DataFrame(df[col].value_counts())
        percentage = int(vals[col].iloc[0]) / len(df[col])

        if percentage>=threshold:
            a.append(col)
    
    if verbose>0:
        print("\nRemoving high zeroes prevalence:\n", list(set(a)))
    
    return df.drop(list(set(a)-set(flag)), axis=1)

print("remove_zereos() ready")


# In[23]:


def create_dummies(df, threshold=2):

    a = df.columns
    cats_used = np.array([])
    
    for c in a:
        values = df[c].value_counts()
        
        if len(values)>threshold:
            try:
                value = values[threshold]
                cats = list(values[values >= value].index)
                cats_used = np.append(cats_used, cats)
                df[c] = [cat if cat in cats else 'Other' for cat in df[c]]
            except:
                print(c)
                df = df.drop(c)
        else:
            cats_used = np.append(cats_used, list(values.index))
            
    return df, cats_used

print("create_dummies() ready")


# In[24]:


def preset_dummies(df, cats):

    dum_df = pd.DataFrame()
    a = df.columns

    for c in a:
        try:
            dum_df[c] = [cat if cat in cats else 'Other' for cat in df[c]]
        except:
            print(c)

    return dum_df

print("preset_dummies() ready")


# In[25]:


def stratify(df, lifecycle, flag):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    
    for train_index, test_index in split.split(df, df[flag]):
        strat_train_set = df.iloc[train_index]
        strat_test_set = df.iloc[test_index]
    
    print(lifecycle, "\nTraining Set:\n", strat_train_set[flag].value_counts(), "\nTest Set:\n", strat_test_set[flag].value_counts())
    
    return strat_train_set, strat_test_set

print("stratify() ready")


# In[26]:


def scale(num_cols, dum_cols, strat_set, flag):

    y = pd.DataFrame(strat_set[flag].copy())
    X = strat_set[num_cols].copy()
    X.drop([flag], axis=1, inplace=True)

    X_index = X.index
    
    X_scaler = StandardScaler().fit_transform(X)
    X_scaled = pd.DataFrame(X_scaler, columns=X.columns, index=X_index)
    
    X_concat = pd.concat([X_scaled, strat_set[dum_cols]], axis=1)
    
    return X_concat, y

print("scale() ready")


# In[27]:


def execute_classifier(ph, name, clf, X, y, X_valid, y_valid, n=20):
    
    clf.fit(X, y)
    y_pred = clf.predict(X_valid)
    try:
        y_proba = clf.predict_proba(X_valid)
        auc = round(roc_auc_score(y_valid, y_proba[:,1]) * 100, 2)
    except:
        y_proba = []
        auc = 0
    
    try:
        fi_ = pd.DataFrame({'variable': X.columns,
                            'feature_importance': clf.feature_importances_}).sort_values('feature_importance',
                                                                                         ascending=False)
        fi_n = list(fi_['variable'].iloc[:n])
    except:
        fi_n = ""
    
    acc = round(accuracy_score(y_valid, y_pred) * 100, 2)
    f1 = round(f1_score(y_valid, y_pred) * 100, 2)
    recall = round(recall_score(y_valid, y_pred) * 100, 2)
    precision = round(precision_score(y_valid, y_pred) * 100, 2)
    
    try:
        con_mat = confusion_matrix(y_valid.tolist(), y_pred.tolist())
    except:
        con_mat = []
        
    params = clf.get_params()
    
    return pd.DataFrame({'Phase': [ph],
                         'Model': [name],
                         'Params': [params],
                         'Dataset': ["Test"],
                         'Accuracy' : [acc],
                         'ROC/AUC' : [auc],
                         'F1' : [f1],
                         'Recall' : [recall],
                         'Precision' : [precision],
                         'Top n' : [fi_n],
                         '# Features' : [len(X.columns)],
                         'Features' : [list(X.columns)],
                         'Confusion Matrix': [con_mat]
                        })

print("execute_classifier() ready")


# In[28]:


def run_classifiers(phase, X_train, y_train, X_test, y_test):

    names = ["RF", "XGB", "CAT", "LGB", "ADB",
             "DT", "NNet", "GNB", "KNN", "QDA", "GB", "ET", "BNB", "PAC", "LR", "SDG", "PER", "LSV"
            ]

    classifiers = [
        RandomForestClassifier(random_state=42),
        xgb.XGBClassifier(random_state=42),
        cat.CatBoostClassifier(logging_level='Silent', random_state=42),
        lgb.LGBMClassifier(random_state=42),
        AdaBoostClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        MLPClassifier(random_state=42),
        GaussianNB(),
        KNeighborsClassifier(),
        QuadraticDiscriminantAnalysis(),
        GradientBoostingClassifier(random_state=42),
        ExtraTreesClassifier(random_state=42),
        BernoulliNB(),
        PassiveAggressiveClassifier(random_state=42),
        LogisticRegression(random_state=42),
        SGDClassifier(random_state=42),
        Perceptron(random_state=42),
        LinearSVC(random_state=42)
    ]

    cols=['Phase', 'Model', 'Params', 'Dataset', 'Accuracy', 'ROC/AUC', 'F1', 'Recall', 'Precision', 'Top n',
          '# Features', 'Features', 'Confusion Matrix']

    results = pd.DataFrame(columns=cols)

    for name, clf in zip(names, classifiers):
        results = results.append(execute_classifier(phase, name, clf, X_train, y_train, X_test, y_test))    
    
    return results

print("run_classifiers() ready")


# # Run Pre-Processing and Execute default modelling

# In[100]:


timestr = time.strftime("%Y%m%d_%H%M%S")
subscriber_res = pd.DataFrame()


# In[105]:


classifier_flags = ["portout_flag", "upgrade_flag"]

contract_groups = ["In-Life HS", "Mid Life SIMO", "Mid Life HS", "LOOC SIMO", "Loyalty SIMO",
                   "Early Life SIMO", "Early Life HS", "LOOC HS", "OOC SIMO", "No Contract SIMO",
                   "In-Life SIMO", "Loyalty HS", "No Contract HS", "OOC HS"]
contract_groups = ["Loyalty SIMO"]

v = 0
cat_res = []

for i in range(len(contract_groups)):
    
    for j in range(len(classifier_flags)):
        
        a = remove_flags(subscribers.columns, to_drop, contract_groups[i], v)
        
        subscribers_rus = rando_under(subscribers[subscribers['contract_group']==contract_groups[i]], a,
                                      classifier_flags[j], v)
        subscribers_rus = dtypes_consistency(subscribers[subscribers_rus.columns], subscribers_rus)
        
        cat_df, num_df = dtype_split(subscribers_rus[1:], v)
        
        subscribers_cor = run_correlation(num_df, v, 0.9)
        subscribers_num = remove_zereos(subscribers_cor, classifier_flags[j], v, 0.9)
        
        dummies_df, cats = create_dummies(cat_df, 10)
        subscribers_dum = pd.get_dummies(dummies_df, sparse=False, drop_first=True)
        cat_res.append(cats)
        #subscribers_dum = subscribers_dum.drop([col for col in subscribers_dum.columns if 'Unknown' in col], axis=1)
        
        subscriber_fin = pd.concat([subscribers_num, subscribers_dum], axis=1)
        
        train, test = stratify(subscriber_fin, contract_groups[i], classifier_flags[j])
        
        X_train, y_train = scale(subscribers_num.columns, subscribers_dum.columns, train, classifier_flags[j])
        X_test, y_test = scale(subscribers_num.columns, subscribers_dum.columns, test, classifier_flags[j])
        
        output_train = pd.concat([X_train, y_train], axis=1)
        output_train['dataset'] = "training set"
        output_test = pd.concat([X_test, y_test], axis=1)
        output_test['dataset'] = "testing set"

        output = pd.concat([output_train, output_test], axis=0)
        output['segment'] = contract_groups[i]+" "+classifier_flags[j]
        
        #output.to_csv("/vha/home/61072380/train_test_"+contract_groups[i].replace(" ", "_")+"_"+classifier_flags[j]+"_"+timestr+".csv")
        
        subscriber_res = subscriber_res.append(run_classifiers(contract_groups[i]+" "+classifier_flags[j], X_train, y_train, X_test, y_test))
               
subscriber_res = subscriber_res.sort_values(by=['ROC/AUC'], ascending=False)
subscriber_res = subscriber_res.reset_index(drop=True)


# In[106]:


idx = subscriber_res.groupby(['Phase'])['ROC/AUC'].transform(max) == subscriber_res['ROC/AUC']
run = subscriber_res[idx].sort_values(by=['Phase'])
run.sort_values(by=['ROC/AUC'], ascending=False)


# # Build y_pred and y_proba for entire subscriber population

# In[ ]:


subscribers_shap = subscribers.copy()
outer = list(set(to_drop)-set(subscribers_shap.columns))
subscribers_shap.drop(list(set(to_drop)-set(outer)), axis=1, inplace=True)
subscribers_shap['tmp'] = 0

subscribers_cat, subscribers_num = dtype_split(subscribers_shap, 0)
dummies_selected = preset_dummies(subscribers_cat, list(cat_res[1]))

subscribers_dum = pd.get_dummies(dummies_selected, sparse=False, drop_first=True)
subscriber_fin = pd.concat([subscribers_num.reset_index(), subscribers_dum], axis=1)
subscriber_fin = subscriber_fin.set_index('ID_Subs_id')

#subscriber_fin.drop('tmp', axis=1).to_csv("/home/61072380/full_"+timestr+".csv")

X_full, y_full = scale(subscribers_num.columns, subscribers_dum.columns, subscriber_fin, 'tmp')

#X_full.to_csv("/home/61072380/scaled_"+timestr+".csv")


# In[ ]:


classifier_flag = "Upgrade_flag"
contract_group = "Loyalty SIMO"

for filename in os.listdir('/vha/home/61072380/'):
    if fnmatch.fnmatch(filename, "train_test_"+contract_group.replace(" ", "_")+"_"+classifier_flag+"*.csv"):
        print(filename)
        subscriber_train_test = pd.read_csv("/vha/home/61072380/"+filename, index_col='ID_Subs_id')

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
subscriber_train_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in subscriber_train_test.columns.values]


# In[ ]:


X_train = subscriber_train_test.copy()
y_train = subscriber_train_test.copy()
X_test = subscriber_train_test.copy()
y_test = subscriber_train_test.copy()

X_train = X_train[X_train['dataset']=="training set"]
X_train.drop([classifier_flag, 'dataset', 'segment'], axis=1, inplace=True)

y_train = y_train[y_train['dataset']=="training set"]
y_train = y_train[classifier_flag]

X_test = X_test[X_test['dataset']=="testing set"]
X_test.drop([classifier_flag, 'dataset', 'segment'], axis=1, inplace=True)

y_test = y_test[y_test['dataset']=="testing set"]
y_test = y_test[classifier_flag]

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# In[ ]:


if len(list(set(X_train.columns)-set(X_full.columns)))==0:
    best_params = subscriber_res.loc[1].Params
    best_cols = subscriber_res.loc[1].Features
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred_port = model.predict(X_full[best_cols])
    y_proba_port = model.predict_proba(X_full[best_cols])
    
    output_port = pd.DataFrame(y_proba_port)
    output_port = output_port.set_index(X_full.index)
    output_port = output_port.join(subscribers['Contract_Group'])
    
    output_port.to_csv("/vha/home/61072380/y_proba_"+contract_group+"_"+classifier_flag+"_"+timestr+".csv")
    output_port.head(5)
else:
    print("Column inconsistency detected\n", list(set(X_train.columns)-set(X_full.columns)))


# In[ ]:


if len(list(set(X_train.columns)-set(X_full.columns)))==0:
    best_params = subscriber_res.loc[12].Params
    best_cols = subscriber_res.loc[12].Features
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    y_pred_up = model.predict(X_full[best_cols])
    y_proba_up = model.predict_proba(X_full[best_cols])
    
    output_up = pd.DataFrame(y_proba_up)
    output_up = output_up.set_index(X_full.index)
    output_up = output_up.join(subscribers['Contract_Group'])
    
    output_up.to_csv("/vha/home/61072380/y_proba_"+contract_group+"_"+classifier_flag+"_"+timestr+".csv")
    output_up.head(5)
else:
    print("Column inconsistency detected\n", list(set(X_train.columns)-set(X_full.columns)))


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=y_proba_port, bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y_proba')
plt.ylabel('Subs')
plt.title('Loyalty SIMO Port Out y_proba Histogram')


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=y_proba_up, bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y_proba')
plt.ylabel('Subs')
plt.title('Loyalty SIMO Upgrade y_proba Histogram')


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=output_port[output_port['Contract_Group']==contract_group][1], bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y_proba')
plt.ylabel('Subs')
plt.title(contract_group+' Port Out y_proba Histogram')


# In[ ]:


# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=output_up[output_up['Contract_Group']==contract_group][1], bins='auto', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('y_proba')
plt.ylabel('Subs')
plt.title(contract_group+' Upgrade y_proba Histogram')


# In[ ]:




