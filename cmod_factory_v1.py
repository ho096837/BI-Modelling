#!/usr/bin/env python
# coding: utf-8

# # Binary Classification Modelling Factory
# <img src="http://ran-tools.vodafone.com.au/tools/coras/cmodel/aa_pipe.png" alt="Advanced Analytics Pipeline" title="Advanced Analytics Pipeline" />
# 
# ***
#     @author: Jonny Keane
#     @created: 25/07/2019
#     @updated: 05/08/2019
#     @version: 0.9.1
#     @latest:
#     - Added RFE feature selection function
#     - Made hyper-parameter optimisation function agnostic to classifier method
#     - Added SHAP analysis for best model
#     @resources:
#     - https://bit.ly/2UNdPmx
#     - https://github.com/slundberg/shap/issues/29 (explanation of SHAP for trees versus boosting methods)
#     @to do:
#     - Explore Deep Feature Synthesis (https://bit.ly/2YUOkoV)
# 
# ***
# ## Objective: Create a recursive function to perform model selection, feature selection and hyper-parameter tuning in a binary classification problem
# 
# This phase of the advanced analytics pipeline is the "modelling" phase. Before this phase is performance, its assume feature optimisation has been completed and the data has been pre-processed as follows:
# - under/over sampled
# - cleanised
# - stratified
# - scaled
# - features not highly correlated with each other
# 
# The **steps** taken in the phase are as follows:
# 1. [Import packages](#import-packages) to support analysis.
# 2. [Import the data](#import-data) in the form of a csv flat file.
# 3. [Separate the data](#separate-data) into training and test sets, as defined in the pre-processing phase.
# 4. [Run base classifiers](#base) on Random Forest, Xtreme Gradient Boost, CatBoost, LightGBM, AdaBoost, K Nearest Neighbors, Linear SVM, RBF SVM, Gaussian Process, Decision Tree, MLP Neural Networks, Gaussian Naive Bayes, Quadratic Discriminant Analysis, Gradient Boosting, Extra Trees, Bernoulli Naive Bayes, Passive Aggressive, Logistic Regression, Stochastic Gradient Descent, Perceptron, Multinomial Naive Bayes, Linear SVC.
# 5. [Run hyper-parameter optimisation](#hyper-base) using the top n feature importance from base classifiers
# 6. [Compare base and hyper-parameter optimised results](#compare-hb)
# 7. [Run RFE function](#rfe) to provide a second method of feature selection
# 8. [Run hyper-parameter optimisation](#hyper-rfe) using the RFE feature selection
# 9. [Compare base, base hyper-parameter optimised, base RFE and base RFE hyper-parameter optimised results](#compare-hbr)
# 10. [Select optimum model and perform proba()](#compare-hbr)
# 
# NB. confusion matrix measured as:  
# <img align="left" src="http://ran-tools.vodafone.com.au/tools/coras/cmodel/confusion_matrix.png" alt="Confusion Matrix" title="Confusion Matrix" />
# 
# <img src="http://ran-tools.vodafone.com.au/tools/coras/cmodel/New_VF_Icon_CMYK_RED.png" align="right" alt="Vodafone Hutchison Australia" title="Vodafone Hutchison Australia" />  

# <a id="import-packages"></a>
# ## Import packages
# ***

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
import fnmatch
import pathlib

#import teradatasql
from zipfile import ZipFile
from urllib.request import urlopen

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

print("Packages Imported")


# <a id="import-data"></a>
# ## Import the data
# ***

# In[2]:


na = ('','#N/A','#N/A N/A','#NA','-1.#IND','-1.#QNAN','-NaN','-nan','1.#IND','1.#QNAN','N/A',
      'NA','NULL','NaN','n/a','nan','null','NAN','?')

subscribers = pd.read_csv("/vha/home/61072380/Subscriber_Features_Dev-TAB.txt", 
                         sep='\t', lineterminator='\n', na_values=na, error_bad_lines=False)

print("Initial Dataset Imported")


# In[6]:


# Remove unwanted characters in feature name
subscribers.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', ' ', '/', '+', '<'))) else col for col in subscribers.columns.values]
subscribers.columns = map(str.lower, subscribers.columns)

# Address missing data
for col in subscribers.columns:
    if "\r" in col:
        subscribers = subscribers = subscribers.rename(columns={col: col.replace('\r', '')})
        
list(subscribers.columns)


# In[4]:


clrs = ["Vol_Churn_Flag", "Portout_Flag", "Invol_Churn_Flag", "Discon_Flag", "Upgrade_flag"]
cgrs = ["In-Life HS", "Mid Life SIMO", "Mid Life HS", "LOOC SIMO", "Loyalty SIMO", "Early Life SIMO",
        "Early Life HS", "LOOC HS", "OOC SIMO", "No Contract SIMO", "In-Life SIMO", "Loyalty HS",
        "No Contract HS", "OOC HS", "ModelB"]

classifier_flag = "portout_flag"
contract_group = "Loyalty SIMO"


# In[7]:


for filename in os.listdir('/vha/home/61072380/'):
    if fnmatch.fnmatch(filename, "train_test_"+contract_group.replace(" ", "_")+"_"+classifier_flag+"*.csv"):
        print(filename)
        subscriber_train_test = pd.read_csv("/vha/home/61072380/"+filename, index_col='id_subs_id')

regex = re.compile(r"\[|\]|<", re.IGNORECASE)
subscriber_train_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in subscriber_train_test.columns.values]
subscriber_train_test.head(5)


# In[8]:


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


# In[9]:


def results_reset(df):
    cols=['Phase', 'Model', 'Params', 'Dataset', 'Accuracy', 'ROC/AUC', 'F1', 'Recall', 'Precision', 'Top n',
          '# Features', 'Features', 'Confusion Matrix']

    df = df.sort_values(by=['ROC/AUC'], ascending=False)
    df = df.reset_index(drop=True)
    df = df[cols]
    
    return df

print("results_reset() ready.")


# In[10]:


# print ROC/AUC, Accuracy etc. and confusion matrix for high level understanding of model performance
def print_accuracy(model, X, y):
    y_pred = model.predict(X)
    
    try:
        y_proba = model.predict_proba(X)
        print(f'ROC AUC of RF classifier on test set {roc_auc_score(y, y_proba[:,1]):.2f}')
    except:
        print("ROC AUC of RF classifier on test set n/a")
    
    print('Accuracy of classifier on test set {:.2f}'.format(accuracy_score(y, y_pred)))
    print('F1 of classifier on test set {:.2f}'.format(recall_score(y, y_pred)))
    print('Recall of classifier on test set {:.2f}'.format(recall_score(y, y_pred)))
    print('Precision of classifier on test set {:.2f}'.format(precision_score(y, y_pred)))
    print(classification_report(y, y_pred))

    con_mat = confusion_matrix(y.tolist(), y_pred.tolist())
    print(con_mat)
    
print("print_accuracy() ready.")


# In[11]:


# user defined accuracy measure using cross-validation
def get_cv_accuracy(model, X, y, s="accuracy", cv=10):
    return round(cross_val_score(model, X, y, cv=cv, scoring=s).mean() * 100, 2)

print("get_cv_accuracy() ready.")


# In[12]:


def run_classifier(ph, name, clf, X, y, X_valid, y_valid, n=20):
    
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
    con_mat = confusion_matrix(y_valid.tolist(), y_pred.tolist())
    
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

print("run_classifier() ready.")


# In[13]:


out = False

# Grab a time stamp at the beginning of the run to use in the filenames etc.
timestr = time.strftime("%Y%m%d_%H%M%S")

# Our comprehensive list of classifiers
names = ["RF", "XGB", "CAT", "LGB", "ADB", "DT", "NNet", "GNB", "KNN", "QDA", "GB", "ET", "BNB", "PAC", "LR",
         "SDG", "PER", "LSV"]

# This is a factory!! Why not run a base classifier using every classifier we can think of.
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
    results = results.append(run_classifier("Base", name, clf, X_train, y_train, X_test, y_test, 20))
    if out:
        results.to_csv("/vha/home/61072380/factory_results_"+timestr+".csv")

results = results_reset(results)

if out:
    print("Filename: factory_results_"+timestr+".csv")


# In[14]:


results


# <a id="hyper-base"></a>
# # Hyper-Parameter Tuning
# ***
# In contrast to model parameters which are learned during training, hyper-parameters are presets by the user ahead of training.
# 
# There are 4 approaches to hyper-parameter tuning. Ultimatelym each of these are different ways that guess the best configuration for a given model and training dataset.
# 
# 1. **Manual:** select hyper-parameters based on intuition/experience/guessing, train the model with the hyperparameters, and score on the validation data. Repeat process until you run out of patience or are satisfied with the results.
# 2. **Grid Search:** set up a grid of hyper-parameter values and for each combination, train a model and score on the validation data. In this approach, every single combination of hyper-parameters values is tried which can be very inefficient!
# 3. **Random search:** set up a grid of hyper-parameter values and select random combinations to train the model and score. The number of search iterations is set based on time/resources.
# 4. **Automated Hyperparameter Tuning:** use methods such as gradient descent, Bayesian Optimization, or evolutionary algorithms to conduct a guided search for the best hyper-parameters.
# 
# ### 3 steps in the Hyper-Parameter Tuning process
# 1. **Objective function:** a function that takes in hyper-parameters and returns a score we are trying to minimize or maximize.
# 2. **Domain:** the set of hyper-parameter values over which we want to search.
# 3. **Algorithm:** method for selecting the next set of hyper-parameters to evaluate in the objective function.
# 
# ### n_estimators
# 
# For those classifiers that improve accuracy by adding more trees (or estimators), in doing so, the training error will always decrease because the capacity of the model increases. Although this might be seen to be a positive, it means that the model will start to memorise the training data and then will not perform well on new testing data. The variance of the model increases as we continue adding estimators because the model starts to rely too heavily on the training data (high variance means overfitting).
# 
# To avoid this, **early_stopping_rounds=100** is applied.
# 
# ***
# ## 1. Objective Function

# In[15]:


def objective(model, hyperparameters, X, y):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""
    
    N_FOLD = 5
    EARLY_STOP = 50
    SEED = 42
    N_BOOST_RND = 50
    V = 0
    METRICS = 'auc'
    METRICS_RF = 'roc_auc'
    score = 0
    
    # Number of estimators will be found using early stopping (only for classifiers with that parameter)
    if ('early_stopping_rounds' in hyperparameters.keys()) & ('n_estimators' in hyperparameters.keys()):
        del hyperparameters['n_estimators']
    
    if model=="LGB":
        dtrain = lgb.Dataset(data=X, label=y)
        cv_results = lgb.cv(hyperparameters, dtrain, num_boost_round=N_BOOST_RND, nfold=N_FOLD,
                            early_stopping_rounds=EARLY_STOP, metrics=METRICS, seed=SEED, verbose_eval=V)

        score = cv_results['auc-mean'][-1]
        estimators = len(cv_results['auc-mean'])
        hyperparameters['n_estimators'] = estimators
        
    elif model=="CAT":
        dtrain = cat.Pool(data=X, label=y)
        
        cv_results = cat.cv(dtrain, hyperparameters, num_boost_round=N_BOOST_RND, fold_count=N_FOLD,
                        early_stopping_rounds=EARLY_STOP, seed=SEED, logging_level='Silent')
        
        score = cv_results['train-RMSE-mean'][len(cv_results['train-RMSE-mean'])-1]
        estimators = len(cv_results['train-RMSE-mean'])
        hyperparameters['n_estimators'] = len(cv_results['train-RMSE-mean'])
        
    elif model=="XGB":
        dtrain = xgb.DMatrix(X, label=y)
        
        hyperparameters['random_state'], hyperparameters['seed'] = SEED, SEED
        hyperparameters['nthread'], hyperparameters['silent'] = -1, True
        
        clf = xgb.XGBClassifier(**hyperparameters)
        
        cv_results = xgb.cv(clf.get_params(), dtrain, num_boost_round=N_BOOST_RND, nfold=N_FOLD, 
                            early_stopping_rounds=EARLY_STOP, metrics=[METRICS])

        score = cv_results['train-auc-mean'][len(cv_results['train-auc-mean'])-1]
        estimators = len(cv_results['train-auc-mean'])
        hyperparameters['n_estimators'] = len(cv_results['train-auc-mean'])

    elif model in ["RF", "ADB"]:
        if model=="RF":
            clf = RandomForestClassifier(**hyperparameters)
        else:
            clf = AdaBoostClassifier(**hyperparameters)
            
        cv = StratifiedKFold(n_splits=N_FOLD, random_state=SEED)
        cv_results = cross_val_score(clf, X, y, cv=cv, scoring=METRICS_RF, n_jobs=-1, verbose=V)
        score = cv_results.mean()
        
    return hyperparameters

print("objective() ready.")


# ## 2. Domain
# The domain is the list of parameters, and the range of their values, we are going to explore to see if we can generate a higher score than by running our base classifers (which were run with default parameter settings).
# 
# *Grid Search* will iterate through the range for each parameter and remember the best setting, which could be slow as noted above. Whereas, *Random Search* will select a parameter at random and hope to hit a higher accuracy, in doing so it will forget the best setting and try at random again.

# In[16]:


max_depth = list(np.arange(3, 15))
max_depth.append(None)

xgb_grid = {
    'colsample_bytree': list(np.linspace(0.5, 0.9, 3)),
    'gamma': list(np.linspace(0.1, 0.5, 5)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base=10, num=100)),
    'max_depth': max_depth[0:len(max_depth)-1],
    'min_child_weight': list(np.linspace(1, 7, 7)),
    'random_state': [42, 42],
    'scale_pos_weight': list(np.arange(1, 3)),
    'subsample': list(np.linspace(0.5, 0.9, 4))
}

lgbm_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'class_weight': [None, 'balanced'],
    'colsample_bytree': list(np.linspace(0.6, 2, 10)),
    'is_unbalance': [True, False],
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 100)),
    'min_child_samples': list(range(20, 500, 5)),
    'num_leaves': list(range(20, 150)),
    'reg_alpha': list(np.linspace(0, 1)),
    'random_state': [42, 42],
    'reg_lambda': list(np.linspace(0, 1)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'subsample_for_bin': list(range(20000, 300000, 20000))
}

rf_grid = {
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'criterion': ['gini', 'entropy'],
    'max_depth': max_depth,
    'max_features': ['auto', 'sqrt', 'log2', None],
    'min_samples_leaf': list(np.arange(1, 10, 2)),
    'min_samples_split': list(np.arange(2, 21, 3)),
    'n_estimators': list(np.arange(100, 2000, 100)),
    'n_jobs': [-1, -1],
    'oob_score': [True, False],
    'random_state': [42, 42]
}

adb_grid = {
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 100)),
    'n_estimators': list(np.arange(100, 2000, 100)),
    'random_state': [42, 42]
}

cat_grid = {
    'bagging_temperature': list(np.linspace(0, 1)),
    'colsample_bylevel': list(np.linspace(0.5, 1, 5)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(1), base=10, num=1000)),
    'logging_level': ['Silent', 'Silent'],
    'one_hot_max_size': list(np.linspace(2, 20, 10)),
    'random_strength': list(np.linspace(1, 20, 20)),
    'random_state': [42, 42],
    'reg_lambda': list(np.arange(1, 100,))
    #'task_type': ["GPU", "GPU"]
}

print("Search params ready.")


# In[17]:


list(np.arange(1, 10, 2))


# For *LightGBM*, if **boosting_type** is **goss**, then we cannot use **subsample** (which refers to training on only a fraction of the rows in the training data, a technique known as *Stochastic Gradient Boosting*). Therefore, we will need a line of logic in our algorithm that sets the **subsample=1** (which means use all the rows) if **boosting_type=goss**. As an example below, if we randomly select a set of hyper-parameters, and the boosting type is **goss**, then we set the **subsample** to 1.0.
# ***
# ## 3 Alogrithm
# 
# ### 3.1 Grid Search
# 
# Grid search is best described as exhaustive guess and check. The grid search method for finding the best parameter settings is to try all combinations of values in the domain and hope that the best combination is in the grid (in reality, we will never know if we found the best settings unless we have an infinite hyper-parameter grid which would then require an infinite amount of time to run).
# 
# Grid search suffers from one limiting problem: it is extremely computationally expensive because we have to perform cross validation with every single combination of hyper-parameters in the grid!

# In[18]:


def grid_search(model, param_grid, X, y, max_evals=5):
    """Grid search algorithm (with limit on max evals)"""
    
    res = pd.DataFrame(columns=['score', 'params', 'iteration'], index=list(range(max_evals)))
    keys, values = zip(*param_grid.items())
    i=0
    
    # Iterate through every possible combination of hyperparameters
    for v in itertools.product(*values):
        
        hyperparameters = dict(zip(keys, v))
        
        if model=="LGB":
            hyperparameters['subsample']=1.0 if hyperparameters['boosting_type']=='goss' else hyperparameters['subsample']
        elif hyperparameters['oob_score'] & hyperparameters['bootstrap']==False:
            hyperparameters['oob_score']=False
        
        eval_results = objective(model, hyperparameters, X, y)
        res.loc[i, :] = eval_results
        i+=1
        
        # Normally would not limit iterations
        if i > max_evals:
            break
       
    res.sort_values('score', ascending=False, inplace=True)
    res.reset_index(inplace=True)
    
    return res.loc[0, 'params']

print("grid_search() ready.")


# ### 3.2 Random Search

# In[19]:


def random_search(df, model, param_grid, X, y, X_valid, y_valid, max_evals=5):
    """Random search for hyperparameter optimization"""

    count = df.shape[0]
    roc_auc = df['ROC/AUC'].iloc[0]
    
    for i in range(max_evals):
        
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        
        if model=="LGB":
            hyperparameters['subsample']=1.0 if hyperparameters['boosting_type']=='goss' else hyperparameters['subsample']
        if model=="RF":
            if hyperparameters['oob_score'] & hyperparameters['bootstrap']==False:
                hyperparameters['oob_score']=False
        
        print(model, i+1, "\n", hyperparameters)
        
        params = objective(model, hyperparameters, X, y)
        
        if model=="LGB":
            rs_model = lgb.LGBMClassifier(**params)
        elif model=="CAT":
            rs_model = cat.CatBoostClassifier(**params)
        elif model=="XGB":
            rs_model = xgb.XGBClassifier(**params)
        elif model=="RF":
            rs_model = RandomForestClassifier(**params)
        elif model=="ADB":
            rs_model = AdaBoostClassifier(**params)
        
        result = run_classifier("RS", model, rs_model, X, y, X_valid, y_valid)
        result.index = range(count, count+1)
        count+=1
        
        if result['ROC/AUC'].iloc[0]>roc_auc:
            df = df.append(result)
            df.to_csv("/vha/home/61072380/factory_results_"+contract_group.replace(" ", "_")+"_"+classifier_flag+"_"+timestr+".csv")
            roc_auc = result['ROC/AUC'].iloc[0]
    
    return df

print("random_search() ready.")


# In[20]:


def hyper_param(df, model, X, y, X_valid, y_valid):
    
    MAX_EVAL = 200
    df_searched = pd.DataFrame()
    
    #if model=="LGB":
        #df_searched = random_search(df, model, lgbm_grid, X, y, X_valid, y_valid, MAX_EVAL)
    #elif model=="CAT":
        #df_searched = random_search(df, model, cat_grid, X, y, X_valid, y_valid, MAX_EVAL)
    if model=="XGB":
        df_searched = random_search(df, model, xgb_grid, X, y, X_valid, y_valid, MAX_EVAL)
    elif model=="RF":
        df_searched = random_search(df, model, rf_grid, X, y, X_valid, y_valid, MAX_EVAL)
    #elif model=="ADB":
        #df_searched = random_search(df, model, adb_grid, X, y, X_valid, y_valid, MAX_EVAL)
    else:
        return df
    
    return df_searched
    
print("hyper_param() ready.")


# In[21]:


res_length = results.shape[0]

# get unique list from all Top n feaure importance where x>=n
top_x = list(set(itertools.chain.from_iterable(results[results['Top n']!=""]['Top n'].tolist())))

for n in range(res_length):
    if results.loc[n, 'Phase']=="Base":
        name = results.loc[n, 'Model']
        
        # All features
        results = hyper_param(results, name, X_train, y_train, X_test, y_test)

        top_n = results.loc[n, 'Top n']

        if len(top_n)>0:
            rand_n = random.randint(10, len(top_n))
            rand_nlist = set(X_train[random.choices(top_n, k=rand_n)].columns)
            
            # Random list of top n features
            #results = hyper_param(results, name, X_train[rand_nlist], y_train, X_test[rand_nlist], y_test)
            
            # Top n features
            results = hyper_param(results, name, X_train[top_n], y_train, X_test[top_n], y_test)
            
        rand_x = random.randint(10, len(top_x))
        rand_xlist = set(X_train[random.choices(top_x, k=rand_x)].columns)
        
        # Random list of top x features (x=agg(n))
        #results = hyper_param(results, name, X_train[rand_xlist], y_train, X_test[rand_xlist], y_test)
        
        # Top x features
        results = hyper_param(results, name, X_train[top_x], y_train, X_test[top_x], y_test)

results = results_reset(results)
        
if out:
    print("Updated:", "/vha/home/61072380/factory_results_"+contract_group.replace(" ", "_")+"_"+classifier_flag+"_"+timestr+".csv")
    


# In[22]:


results


# <a id="rfe"></a>
# ## Recursive Feature Elimination

# In[ ]:


def rfe_feature_select(count, features, X, y, model_type="RF", model=RandomForestClassifier(random_state=42),
                       v=0, scores=pd.DataFrame(columns=['model_type', 'model','params', 'n_features',
                                                         'score', 'features','ranking', 'weighted_score'])):
    
    start = time.time()
    helicopter = [count, round(count*0.8), round(count*0.6), round(count*0.4), round(count*0.2)]
    weightings = [0.98, 0.985, 0.99, 0.995, 1]
    
    if v>0:
        print(model)
        
    score = pd.DataFrame()
    
    for n in range(len(helicopter)):
        eliminator = RFE(estimator=model, n_features_to_select=helicopter[n], step=1, verbose=v)
        eliminator.fit(X, y)
        feats = pd.Series(eliminator.support_, index=features)
        weight  = 100*eliminator.score(X, y)
        params = model.get_params()
        
        score = score.append({'model_type' : model_type,
                                'model' : model,
                                'params': params,
                                'n_features' : eliminator.n_features_,
                                'score' : round(score, 2),
                                'features' : list(feats[feats==True].index),
                                'ranking' : list(eliminator.ranking_),
                                'weighted_score' : round(weightings[n]*weight, 2)},
                               ignore_index=True)
    if v>0:
        print("%d features took %d seconds, or %d minutes" % (count, round(time.time()-start), round((time.time()-start)/60)))
        print("Tested feature combinations: %s" % helicopter)
        print(score)
    
    df = score.sort_values(by='weighted_score', ascending=False)
    
    print("Selected feature combination: %d" % df.iloc[0].n_features)
    
    if helicopter[0]-helicopter[1]==1: # We've reached the end
        return scores
    elif df.iloc[0].n_features==count: # The highest feature count is the best so try -1 features
        tmp_features = df.iloc[0].features
        tmp_count  = len(tmp_features)-1
        
        scores = rfe_feature_select(tmp_count, tmp_features[0:tmp_count], X_train[tmp_features[0:tmp_count]],
                                y_train, df.iloc[0].model_type, df.iloc[0].model, v, scores)
    else:
        scores = rfe_feature_select(df.iloc[0].n_features, df.iloc[0].features, X_train[df.iloc[0].features],
                                y_train, df.iloc[0].model_type, df.iloc[0].model, v, scores)
    return scores
        
print("RFE Feature selection ready")


# In[ ]:


cols = X_train.columns
rfe_results = pd.DataFrame()

models = list(set(results[results['Phase']=="RS"]['Model']))

# Our comprehensive list of classifiers
names = ["RF", "XGB", "CAT", "LGB", "ADB", "DT", "NNet", "GNB", "KNN", "QDA", "GB", "ET", "BNB", "PAC", "LR", 
         "SDG", "PER", "LSV"]

names_best = ["RF", "XGB", "CAT", "LGB", "ADB"]
names_best = ["RF"]

# This is a factory!! Why not run a base classifier using every classifier we can think of.
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
'''
params_best = [
    results[results['Model']=="RF"].iloc[0]['Params'],
    results[results['Model']=="XGB"].iloc[0]['Params'],
    results[results['Model']=="CAT"].iloc[results['ROC/AUC'].idxmax()]['Params'],
    results[results['Model']=="LGB"].iloc[results['ROC/AUC'].idxmax()]['Params'],
    results[results['Model']=="ADB"].iloc[results['ROC/AUC'].idxmax()]['Params']
]
'''
params_best = [
    results[results['Model']=="RF"].iloc[0]['Params']
]
'''
classifiers_best = [
    RandomForestClassifier(**params_best[0]),
    xgb.XGBClassifier(**params_best[1]),
    cat.CatBoostClassifier(**params_best[2]),
    lgb.LGBMClassifier(**params_best[3]),
    AdaBoostClassifier(**params_best[4])
]
'''
classifiers_best = [
    RandomForestClassifier(**params_best[0])
]

# Best run
for name, clf in zip(names_best, classifiers_best):
    cols = results[results['Model']==name].iloc[0]['Features']
    rfe_results = rfe_results.append(rfe_feature_select(len(cols), list(cols), X_train[cols], y_train, name, clf, 1))
    print(rfe_results)
    
    if out:
        #rfe_results.to_csv("/vha/home/61072380/rfe_results_"+timestr+".csv")
        print("Updated: /vha/home/61072380/rfe_results_"+timestr+".csv")


# In[ ]:


#timestr = "20190814_095708"
#rfe_results = pd.read_csv("/vha/home/61072380/rfe_results_"+timestr+".csv", index_col=0, quoting=2)
rfe_results


# In[ ]:


best_grid_hyp = grid_hyp.iloc[grid_hyp['score'].idxmax()].copy()
best_random_hyp = random_hyp.iloc[random_hyp['score'].idxmax()].copy()

idx = rfe_features.groupby(['model_type'])['weighted_score'].transform(max) == rfe_features['weighted_score']
max_rfe = rfe_features[idx]
cols = ['run', 'model', 'asessed_recall_params', 'best_recall_params', 'recall_score',
        'asessed_precision_params', 'best_precision_params', 'precision_score', 'feature_count', 'features']

df = pd.DataFrame(columns=cols)

rf_list = max_rfe[max_rfe['model_type']=="RF"].features.iloc[0]
xgb_list = max_rfe[max_rfe['model_type']=="XGB"].features.iloc[0]

rf_list.extend(['dataset', 'Portout_Flag'])
xgb_list.extend(['dataset', 'Portout_Flag'])


# In[ ]:


#out = True
res_length = results.shape[0]

# get unique list from all Top n feaure importance where x>=n
top_x = list(set(itertools.chain.from_iterable(results[results['Top n']!=""]['Top n'].tolist())))

for n in range(res_length):
    if results.loc[n, 'Phase']=="Base":
        name = results.loc[n, 'Model']
        results = hyper_param(results, name, X_train, y_train, X_test, y_test)

        top_n = results.loc[n, 'Top n']

        if len(top_n)>0:
            results = hyper_param(results, name, X_train[top_n], y_train, X_test[top_n], y_test)
            results = hyper_param(results, name, X_train[top_x], y_train, X_test[top_x], y_test)

results = results_reset(results)
        
if out:
    print("Updated: /vha/home/61072380/factory_results_"+timestr+".csv")


# ## Reviewing the Results & SHAP file creation

# In[ ]:


shap.initjs()


# In[ ]:


name = "XGB"

best_params = results.loc[11]['Params']
#best_params = results[results['Model']==name].iloc[results['ROC/AUC'].idxmax()]['Params']
#if results[results['Model']==name].iloc[results['ROC/AUC'].idxmax()]['ROC/AUC']<84.61:
    #best_params = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bynode': 1, 'colsample_bytree': 0.7, 'gamma': 0.30000000000000004, 'learning_rate': 0.048850497864961276, 'max_delta_step': 0, 'max_depth': 10, 'min_child_weight': 2.0, 'missing': None, 'n_estimators': 50, 'n_jobs': 1, 'nthread': -1, 'objective': 'binary:logistic', 'random_state': 42, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': 42, 'silent': True, 'subsample': 0.9, 'verbosity': 1}

X = X_train[results.loc[11]['Features']]
#X = X_train[results[results['Model']==name].iloc[results['ROC/AUC'].idxmax()]['Features']]
#if results[results['Model']==name].iloc[results['ROC/AUC'].idxmax()]['ROC/AUC']<84.61:
    #X = X_train[['Country_Type_Other', 'Regn_Nm_Idfcn_Oceania', 'Sub_Tenure_months', 'Plan_Tenure_Group_03. Contract: At Risk', 'Plan_Tenure_Group_02. Contract: In Life', 'Idfcn_Type_desc_Passport', 'Quota_Bucket_07. Greater than 25, less than 30', 'NetworkGen_Upgrade_Flag', 'Subregn_Nm_Idfcn_Southern Asia', 'Regn_Nm_Idfcn_Europe', 'TelstraOptus_PortIn_Flag', 'Prvsnd_Gen_Other', "Idfcn_Type_desc_Driver's Licence", 'Regn_Nm_Idfcn_Asia', 'INB_Vc_Mns_3m', 'Country_United Kingdom', 'Total_Vc_Mns_3m', 'SA2Name_Other', 'ttl_inbnd_cmnty_size_cnt_3m', 'txt_inbnd_cmnty_size_cnt_3m']]

if name=="RF":
    model = RandomForestClassifier(**best_params)
elif name=="XGB":
    model = xgb.XGBClassifier(**best_params)
elif name=="CAT":
    model = cat.CatBoostClassifier(**best_params)
elif name=="LGB":
    model = lgb.LGBMClassifier(**best_params)
elif name=="ADB":
    model = AdaBoostClassifier(**best_params)
    
model.fit(X, y_train)

print_accuracy(model, X_test[X.columns], y_test)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# In[ ]:


# summarize the effects of all the features
shap.summary_plot(shap_values, X)


# In[ ]:


shap.summary_plot(shap_values, X, plot_type="bar")


# In[ ]:


i=1
shap.force_plot(explainer.expected_value, shap_values[i,:], X.iloc[i,:])


# In[ ]:


shap.dependence_plot("Current_Inuse_Dvc_Tenure", shap_values, X) #, interaction_index="SA2Name_Sydney - Haymarket - The Rocks")


# In[ ]:


shap.force_plot(explainer.expected_value, shap_values, X)


# In[ ]:


timestr = time.strftime("%Y%m%d_%H%M%S")

y_proba = model.predict_proba(X)

subs = pd.DataFrame(shap_values, index=X.index)
subs.columns = ["shap_" + s for s in X.columns]
subs = subs.join(subscribers)
subs['y_proba'] = y_proba[:, 1]
top1, top2, top3, bot1, bot2, bot3 = [], [], [], [], [], []

for n in range(subs.shape[0]):
    tops = list(subs[[col for col in subs.columns if 'shap_' in col]].iloc[n].nlargest(3).index)
    bots = list(subs[[col for col in subs.columns if 'shap_' in col]].iloc[n].nsmallest(3).index)
    
    top1.append(tops[0][5:])
    top2.append(tops[1][5:])
    top3.append(tops[2][5:])
    bot1.append(bots[0][5:])
    bot2.append(bots[1][5:])
    bot3.append(bots[2][5:])

subs['top1'] = top1
subs['top2'] = top2
subs['top3'] = top3
subs['bot1'] = bot1
subs['bot2'] = bot2
subs['bot3'] = bot3

subs['shap_output'] = subs[["shap_" + s for s in X.columns]].sum(axis=1)
subs['shap_base'] = explainer.expected_value
subs = subs.set_index(X.index)

df = subs.melt(var_name="name", value_name="value")


# In[ ]:


subs.sort_values(by='shap_output', ascending=False).head()


# In[ ]:


# Transpose SHAP for Qlik
df = subs.reset_index()

nums = list(subscribers.columns[subscribers.dtypes!='object'])
nums.append('ID_Subs_id')
orig_num = df[nums]

# Create a table of original numerical features
t1 = orig_num.melt(id_vars=['ID_Subs_id'], var_name="feat_name",  value_name="feat_value")

shap_nums = ["shap_" + s for s in subscribers.columns[subscribers.dtypes!='object']]
diff_nums = list(set(shap_nums)-set(df.columns))
shap_nums.append('ID_Subs_id')
shap_num = df[(set(shap_nums)-set(diff_nums))]
tmp = shap_num
shap_num.columns = [x.strip().replace('shap_', '') for x in shap_num.columns]

# Create a table of SHAP analysed numerical features
t2 = shap_num.melt(id_vars=['ID_Subs_id'], var_name="feat_name",  value_name="shap_value")

# Merge original and SHAP numerical tables
qlik_num = pd.merge(t1, t2, on=['ID_Subs_id', 'feat_name'], how='left')
qlik_num['dtype'] = "Numeric"

dums = list(subscribers.columns[subscribers.dtypes=='object'])
dums.append('ID_Subs_id')
orig_dum = df[dums]

# Create a table of original categorical features
t3 = orig_dum.melt(id_vars=['ID_Subs_id'], var_name="feat_name",  value_name="feat_value")

# Potential list of categorical features from orinigal input
shap_dums = ["shap_"+s for s in subscribers.columns[subscribers.dtypes=='object']]
tmp1 = ['ID_Subs_id']
tmp2 = []

# Delta loop to remove any feature dropped in pre-processing
for name in shap_dums:
    cats = [col for col in df.columns if name in col]
    if len(cats)>0:
        tmp1.extend(cats)
        tmp2.append(name)

# Filter for features and subs with a SHAP value not equal to zero
shap_dums_real = df[list(set(tmp1))].loc[(df[list(set(tmp1))].sum(axis=1)!=0), (df[list(set(tmp1))].sum(axis=0)!=0)]

shap_dums = list(set(tmp2))
tmp2 = []

# Delta loop to remove any feature with a SHAP value equal to zero
for name in shap_dums:
    cats = [col for col in shap_dums_real.columns if name in col]
    
    if len(cats)>0:
        tmp2.append(name)

shap_dums = list(set(tmp2))

t4 = pd.DataFrame(columns=['ID_Subs_id', 'feat_name', 'feat_value', 'shap_value', 'dtype'])
progress = list(np.arange(int(shap_dums_real.shape[0]*.1), shap_dums_real.shape[0], int(shap_dums_real.shape[0]*.1)))

for row in range(shap_dums_real.shape[0]):
    
    if row in progress:
        print(round(100*row/shap_dums_real.shape[0],2), "%", shap_dums_real.shape[0])
        
    for name in shap_dums:
        
        cats = [col for col in shap_dums_real.columns if name in col]
        
        # Need a better way to handle duplicate returns from string searches. Manual for now.
        if name=="shap_Country":
            dup_cats = [col for col in shap_dums_real.columns if "shap_Country_Type" in col]
            
            for dup_name in dup_cats:
                cats.remove(dup_name)
            
        values = [x.strip().replace(name+"_", '') for x in cats]
        l = len(values)

        i = []
        i += l * [shap_dums_real.loc[row].ID_Subs_id]

        n = []
        n += l * [name.replace("shap_", '')]
        
        c = []
        c += l * ["Category"]
        
        #print(cats, i, n, values, list(shap_dums_real[cats].loc[row, :]), c)

        t4 = t4.append(pd.DataFrame({"ID_Subs_id": i,
                                     "feat_name": n,
                                     "feat_value": values,
                                     "shap_value": list(shap_dums_real[cats].loc[row, :]),
                                     "dtype": c}))

# Merge original and SHAP categorical tables
qlik_dum = pd.merge(t3, t4, on=['ID_Subs_id', 'feat_name', 'feat_value'], how='left')

qlik = pd.concat([qlik_num, qlik_dum], axis=0)
qlik = qlik.set_index('ID_Subs_id', drop=True)

qlik.to_csv("/vha/home/61072380/shap_"+contract_group.replace(" ", "_")+"_"+classifier_flag+"_"+timestr+".csv")


# In[ ]:


qlik[qlik['dtype']=="Category"].sort_values(by='shap_value', ascending=False).head(20)


# In[ ]:


qlik[qlik['dtype']=="Category"].sort_values(by='shap_value', ascending=True).head(20)


# In[ ]:


qlik[qlik['dtype']=="Numeric"].sort_values(by='shap_value', ascending=False).head(20)


# In[ ]:


qlik[qlik['dtype']=="Numeric"].sort_values(by='shap_value', ascending=True).head(20)


# In[ ]:




