#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import requests


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# In[115]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import re


# In[3]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv'


# In[5]:


response = requests.get(data)


# In[6]:


with open('hw_6.csv','wb') as f_file:
    f_file.write(response.content)


# In[56]:


df = pd.read_csv('hw_6.csv')


# In[57]:


df.shape


# In[58]:


df.columns


# In[59]:


df.ocean_proximity.value_counts()


# In[60]:


df = df[(df.ocean_proximity =='<1H OCEAN')|(df.ocean_proximity =='INLAND')]
df.shape


# In[61]:


df.dtypes


# In[62]:


df.isnull().sum()


# In[63]:


df.total_bedrooms = df.total_bedrooms.fillna(0)


# In[64]:


df_full_train,df_test = train_test_split(df,test_size=0.2,random_state=1)
df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)


# In[65]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[66]:


y_train = np.log1p(df_train.median_house_value.values)
y_val = np.log1p(df_val.median_house_value.values)
y_test = np.log1p(df_test.median_house_value.values)


# In[73]:


X_train


# In[72]:


y_train


# In[67]:


del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']


# In[77]:


train_dict = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(train_dict)


# In[79]:


val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


# In[78]:


dt  = DecisionTreeRegressor(max_depth=1)
dt.fit(X_train,y_train)


# In[84]:


print(export_text(dt,feature_names=dv.feature_names_))


# In[88]:


rf = RandomForestRegressor(n_estimators=10,random_state=1,n_jobs=-1)
rf.fit(X_train,y_train)


# In[90]:


y_pred = rf.predict(X_val)
y_pred


# In[93]:


np.sqrt(mean_squared_error(y_val,y_pred))


# In[97]:


scores=[]
for n in range(10,201,10):
    rf = RandomForestRegressor(n_estimators=n,random_state=1,n_jobs=-1)
    rf.fit(X_train,y_train)
    
    y_pred = rf.predict(X_val)
    score = np.sqrt(mean_squared_error(y_val,y_pred))
    
    scores.append((n,score))
df_scores = pd.DataFrame(scores,columns=['n_estimators','rmse'])


# In[98]:


plt.plot(df_scores.n_estimators,df_scores.rmse.round(3));


# In[99]:


scores=[]
for d in [10, 15, 20, 25]:
    for n in range(10,201,10):
        rf = RandomForestRegressor(n_estimators=n,max_depth=d,random_state=1,n_jobs=-1,warm_start=True)
        rf.fit(X_train,y_train)

        y_pred = rf.predict(X_val)
        score = np.sqrt(mean_squared_error(y_val,y_pred))

        scores.append((d,n,score))
df_scores = pd.DataFrame(scores,columns=['max_depth','n_estimators','rmse'])


# In[101]:


df_scores


# In[102]:


for d in [10, 15, 20, 25]:
    df_subset = df_scores[df_scores.max_depth == d]
    plt.plot(df_subset.n_estimators, df_subset.rmse, label=d)

plt.legend()
plt.show()


# In[103]:


rf = RandomForestRegressor(n_estimators=10, max_depth=20, 
                           random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)


# In[104]:


rf.feature_importances_


# In[107]:


df_importance = pd.DataFrame()
df_importance['feature']=dv.feature_names_
df_importance['importance']=rf.feature_importances_
df_importance.sort_values(by='importance', ascending=False).head()


# In[132]:


features = dv.feature_names_
regex = re.compile(r"<", re.IGNORECASE)
features = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in features]
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)


# In[133]:


watchlist = [(dtrain, 'train'), (dval, 'val')]
scores = {}
def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results
     


# In[136]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.3, \n    'max_depth': 6,\n    'min_child_weight': 1,\n\n    'objective': 'reg:squarederror',\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=100,\n                  verbose_eval=5, evals=watchlist)\n")


# In[137]:


parse_xgb_output(output)


# In[138]:


scores['eta=0.3'] = parse_xgb_output(output)


# In[139]:


get_ipython().run_cell_magic('capture', 'output', "\nxgb_params = {\n    'eta': 0.1, \n    'max_depth': 6,\n    'min_child_weight': 1,\n\n    'objective': 'reg:squarederror',\n    'nthread': 8,\n\n    'seed': 1,\n    'verbosity': 1,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=100,\n                  verbose_eval=5, evals=watchlist)\n")


# In[140]:


scores['eta=0.1'] = parse_xgb_output(output)
     


# In[142]:


plt.plot(scores['eta=0.1'].num_iter, scores['eta=0.1'].val_auc,
        label='0.1')
plt.plot(scores['eta=0.3'].num_iter, scores['eta=0.3'].val_auc,
        label='0.3')
plt.legend();


# In[ ]:




