#!/usr/bin/env python
# coding: utf-8

# In[1]:


# System
import os
import sys
import time
from datetime import datetime
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

# Data Processing
import pandas as pd

# Linear Algebra
import numpy as np

# Data Visualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from matplotlib import style
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

 # Scaling and Sampling
import imblearn
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from matplotlib.ticker import PercentFormatter

# Clustering Algorithms
from tslearn.clustering import KShape
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

#Outlier Detection
from minisom import MiniSom

# File management and databases
import csv
#import teradatasql
from zipfile import ZipFile
from urllib.request import urlopen

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)

print("Packages Imported")


# In[6]:


# hourly data per cell with downtime less than 5 minutes in an hour
seca_hh = pd.DataFrame()

try:
    seca_hh = seca_hh.append(pd.read_csv("/vha/home/61072380/seca_hh_norm.csv"))
    
    cols = ['starttime', 'yyyy', 'mm', 'dd', 'hh', 'weekday', 'cellname', 'rrc_connreq_att', 'dl_mac_mb',
            'thrp_bits_ul', 'traffic_user_avg', 'traffic_user_max', 'hho_interenb_interfr_att',
            'ra_ta_ue_index3', 'traffic_activeuser_dl_qci_1', 'traffic_activeuser_dl_qci_8',
            'traffic_activeuser_dl_qci_9', 'traffic_activeuser_dl_avg', 'prb_dl_uti', 'observed', 'season',
            'trend', 'remainder', 'remainder_l1', 'remainder_l2', 'anomaly', 'recomposed_l1', 'recomposed_l2']
    
    seca_hh = seca_hh[cols]
except:
    print("file not found")


# In[71]:


seca_hh_anomolies = pd.DataFrame()

try:
    seca_hh_anomolies = seca_hh_anomolies.append(pd.read_csv("/vha/home/61072380/seca_hh_anomolies.csv"))
except:
    print("file not found")


# In[83]:


anomolies = list(set(seca_hh_anomolies['cellname']))
len(anomolies)


# In[9]:


seca_hh.describe()


# In[10]:


cells = list(set(seca_hh['cellname']))
starttimes = list(set(seca_hh['starttime']))
len(cells)


# In[11]:


missing = pd.DataFrame(seca_hh['cellname'].value_counts())<len(starttimes)
cells_missing = list(missing[missing['cellname']==True].index)
full_cells = list(set(cells)-set(cells_missing))

toolow = pd.DataFrame(seca_hh['cellname'].value_counts())<168
cells_toolow = list(toolow[toolow['cellname']==True].index)
full_cells = list(set(full_cells)-set(cells_toolow))
cells_missing = list(set(cells_missing)-set(cells_toolow))

duplicates = pd.DataFrame(seca_hh['cellname'].value_counts())>len(starttimes)
cells_duplicated = list(duplicates[duplicates['cellname']==True].index)
full_cells = list(set(full_cells)-set(cells_duplicated))
len(full_cells)


# In[12]:


len(cells_missing)


# In[13]:


len(cells_duplicated)


# In[14]:


num_cores = multiprocessing.cpu_count()


# In[105]:


seca_hh['users'] = seca_hh['traffic_user_avg']
seca_hh_anom = seca_hh[seca_hh['cellname'].isin(anomolies)]
seca_hh_norm = seca_hh[seca_hh['cellname'].isin(list(set(full_cells)-set(anomolies)))]

a_idxs = seca_hh_anom[seca_hh_anom['anomaly']=="Yes"].index
n_idxs = seca_hh_norm[seca_hh_norm['observed']==-1].index

seca_hh_anom['users'].loc[a_idxs] = seca_hh_anom['trend'].loc[a_idxs].copy()
seca_hh_norm['users'].loc[n_idxs] = seca_hh_norm['trend'].loc[n_idxs].copy()

seca_hh_fin = pd.concat([seca_hh_anom, seca_hh_norm], ignore_index=True)

df = pd.concat([pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['users'].median()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['users'].max()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['users'].min()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['users'].mean()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['users'].std()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['users'].quantile(0.9)),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['traffic_user_avg'].median()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['traffic_user_avg'].max()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['traffic_user_avg'].min()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['traffic_user_avg'].mean()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['traffic_user_avg'].std()),
                pd.DataFrame(seca_hh_fin.groupby(seca_hh_fin.starttime)['traffic_user_avg'].quantile(0.9))],
               axis=1)

df.columns = ["users Median", "users Max", "users Min", "users Mean", "users Standard Deviation", "users 90th Perc",
             "traffic_user_avg Median", "traffic_user_avg Max", "traffic_user_avg Min", "traffic_user_avg Mean", "traffic_user_avg Standard Deviation", "traffic_user_avg 90th Perc"]
df


# In[106]:


index = pd.date_range(starttimes[0], periods=336, freq="h", name="date")
wide_df = pd.DataFrame(np.array(df[['users Median', 'traffic_user_avg Median']]), index,
                       ['users Median', 'traffic_user_avg Median'])

plt.figure(figsize=(16, 6))
ax = sns.lineplot(data=wide_df, dashes=False)


# # Address missing values

# In[ ]:


def fillinblank(x, y, c, d):

    return {'starttime': d,
            'yyyy': x.strftime("%Y"),
            'mm': x.strftime("%m"),
            'dd': x.strftime("%d"),
            'hh': x.strftime("%H"),
            'weekday': x.strftime("%w"),
            'cellname': c,
            'rrc_connreq_att': y[y['hh']==int(x.strftime("%H"))]['rrc_connreq_att'].median(),
            'dl_mac_mb': y[y['hh']==int(x.strftime("%H"))]['dl_mac_mb'].median(),
            'thrp_bits_ul': y[y['hh']==int(x.strftime("%H"))]['thrp_bits_ul'].median(),
            'traffic_user_avg': y[y['hh']==int(x.strftime("%H"))]['traffic_user_avg'].median(),
            'traffic_user_max': y[y['hh']==int(x.strftime("%H"))]['traffic_user_max'].median(),
            'hho_interenb_interfr_att': y[y['hh']==int(x.strftime("%H"))]['hho_interenb_interfr_att'].median(),
            'ra_ta_ue_index3': y[y['hh']==int(x.strftime("%H"))]['ra_ta_ue_index3'].median(),
            'traffic_activeuser_dl_qci_1': y[y['hh']==int(x.strftime("%H"))]['traffic_activeuser_dl_qci_1'].median(),
            'traffic_activeuser_dl_qci_8': y[y['hh']==int(x.strftime("%H"))]['traffic_activeuser_dl_qci_8'].median(),
            'traffic_activeuser_dl_qci_9': y[y['hh']==int(x.strftime("%H"))]['traffic_activeuser_dl_qci_9'].median(),
            'traffic_activeuser_dl_avg': y[y['hh']==int(x.strftime("%H"))]['traffic_activeuser_dl_avg'].median(),
            'prb_dl_uti': y[y['hh']==int(x.strftime("%H"))]['prb_dl_uti'].median()
            }


# In[ ]:


def missing(c, y, d):

    dates = list(set(d)-set(y['starttime']))
    df = pd.DataFrame()
    
    for date in dates:
        x = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        df = df.append(fillinblank(x, y, c, date), ignore_index=True)
    
    return df


# In[ ]:


if __name__ == "__main__":
    filled = Parallel(n_jobs=num_cores)(delayed(missing)(cell, seca_hh[seca_hh['cellname']==cell], starttimes)
                                        for cell in tqdm(cells_missing))

    seca_hh = seca_hh.append(filled, ignore_index=True)


# # Scale all cells individually

# In[107]:


def scale(df, d):
    values = np.array(df.users).reshape(-1, 1)
    
    if len(values)==len(d):
        return StandardScaler().fit_transform(values)
    
print("scale() ready.")


# In[108]:


timestr = time.strftime("%Y%m%d_%H%M%S")
users = np.array([])


# In[109]:


if __name__ == "__main__":
    scaled = Parallel(n_jobs=num_cores)(delayed(scale)(seca_hh[seca_hh['cellname']==cell],
                                                       starttimes) for cell in tqdm(full_cells[0:1000]))
    users = np.append(users, scaled)

#np.savetxt("/vha/home/61072380/seca_scaled_"+timestr+".csv", users, delimiter=",")


# In[110]:


cluster_numbers = list(np.arange(2, 34, 2))


# In[111]:


data = np.reshape(np.nan_to_num(users), (-1, len(starttimes), 1))

seed = 0
np.random.seed(seed)
sz = data.shape[1]

output = pd.DataFrame()
output['cellname'] = full_cells[0:1000]

for cluster_number in cluster_numbers:
    print(cluster_number)
    ks = KShape(n_clusters=cluster_number, verbose=True, random_state=seed)
    y_pred = ks.fit_predict(data)
    output[cluster_number] = y_pred


# In[ ]:


try:
    output = pd.read_csv("/vha/home/61072380/seca_hh_clusters_20190930_200800.csv")
    output = output.drop(output.columns[0], axis=1)
except:
    print("No scaled data")

cluster_counts = pd.DataFrame()
cluster_counts['index'] = cluster_numbers
cluster_counts = cluster_counts.set_index('index', drop=True)
cluster_medians = pd.DataFrame()

for cluster_number in cluster_numbers:
    counts = pd.DataFrame(output[str(cluster_number)].value_counts())
    cluster_counts = pd.concat([cluster_counts, counts], axis=1)
    
    try:
        cluster_medians = pd.read_csv("/vha/home/61072380/seca_hh_medians_20190930_200800.csv")
        cluster_medians = cluster_medians.drop(cluster_medians.columns[0], axis=1)
    except:
        for y in range(0, cluster_number):
            cluster_medians[str(cluster_number)+"_"+str(y)] = ks.cluster_centers_[y].ravel()

cluster_counts = cluster_counts.drop(32)
cluster_counts


# In[ ]:


output.head()


# In[ ]:


#cluster_medians.to_csv("/vha/home/61072380/seca_hh_medians_"+timestr+".csv")
starttimes.sort()
cluster_medians['starttime'] = starttimes
cluster_medians.to_csv("/vha/home/61072380/seca_hh_medians_20190930_200800.csv")
cluster_medians


# In[ ]:


starttimes.sort()
cols = [col for col in cluster_medians.columns if '32_13' in col]
cols = ['32_20', '32_22', '32_28', '32_29']
cols = ['32_14', '32_12', '32_18']
index = pd.date_range(starttimes[0], periods=336, freq="h", name="date")
wide_df = pd.DataFrame(np.array(cluster_medians[cols]), index, cols)

plt.figure(figsize=(16, 6))
ax = sns.lineplot(data=wide_df, dashes=False)


# In[ ]:


scatter_df = pd.concat([pd.DataFrame(cluster_medians.median()),
                        pd.DataFrame(cluster_medians.max()),
                        pd.DataFrame(cluster_medians.min()),
                        pd.DataFrame(cluster_medians.mean()),
                        pd.DataFrame(cluster_medians.std()),
                        pd.DataFrame(cluster_medians.quantile(0.9))],
                       axis=1)
scatter_df = scatter_df.reset_index()
scatter_df.columns = ["Cluster", "Median", "Max", "Min", "Mean", "Standard Deviation", "90th Perc"]
scatter_df


# In[ ]:


output.to_csv("/vha/home/61072380/seca_hh_clusters_"+timestr+".csv")


# In[114]:


n=32

plt.figure(figsize=(16, n*4))
for yi in range(n):
    plt.subplot(n, 2, 1 + yi)
    for xx in data[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(ks.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, sz)
    plt.ylim(-5, 5)
    plt.title("Users Cluster %d" % (yi))

plt.tight_layout()
plt.show()


# In[ ]:


clusters = ["0", "3", "4", "5", "6", "7", "15", "24", "27", "30", "31"]
anomaly_cells = list(set(seca_hh[seca_hh['cellname'].isin(list(output[output['32'].isin(clusters)]['cellname']))]['cellname']))


# In[ ]:


pd.DataFrame(anomaly_cells).to_csv("/vha/home/61072380/seca_hh_anomaly_cells.csv")


# In[ ]:


clusters = ["X32_0", "X32_3", "X32_4", "X32_5", "X32_6", "X32_7", "X32_15", "X3_24", "X32_27", "X32_30", "X32_31"]
#to_drop = []

for cluster in clusters:
    tmp = anomalized[anomalized['cluster']==cluster]
    anomalies = list(tmp[tmp['anomaly']=="Yes"]['starttime'])
    anomaly_cells = list(set(seca_hh[seca_hh['cellname'].isin(list(output[output['32']==int(cluster.split("_")[1])]['cellname']))]['cellname']))
    
    todo = len(anomalies)*len(anomaly_cells)
    count = 1
    
    for anomaly in anomalies:
        for anomaly_cell in anomaly_cells:
            to_drop.append(seca_hh.loc[(seca_hh['cellname']==anomaly_cell) &
                                       (seca_hh['starttime']==anomaly)].index[0])
            print(cluster, count, round(100*count/todo, 3))
            count+=1
            
    pd.DataFrame(to_drop).to_csv("/vha/home/61072380/seca_hh_todrop.csv")


# In[ ]:


try:
    to_drop = pd.read_csv("/vha/home/61072380/seca_hh_todrop.csv")
    to_drop = to_drop.drop(to_drop.columns[0], axis=1)
except:
    print("Nothing to drop")
    
seca_hh_norm = seca_hh.drop(list(to_drop['0']))
seca_hh_norm = seca_hh_norm[seca_hh_norm['cellname'].isin(list(output[output['32']!=1]['cellname']))]
seca_hh_norm


# In[ ]:




