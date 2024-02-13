#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[72]:


titanic_data = pd.read_csv("C:\\Users\\DELL\\Downloads\\archive\\Titanic-Dataset.csv")


# In[73]:


titanic_data


# In[74]:


titanic_data.head()


# In[75]:


titanic_data.describe()


# In[76]:


import seaborn as sns
numeric_data = titanic_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, cmap="YlGnBu")
plt.show()


# In[77]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_indices, test_indices in split.split(titanic_data, titanic_data[["Survived","Pclass","Sex"]]):
    strat_train_set = titanic_data.loc[train_indices]
    strat_test_set = titanic_data.loc[test_indices]


# In[55]:


plt.subplot(1,2,1)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.subplot(1,2,2)
strat_train_set['Survived'].hist()
strat_train_set['Pclass'].hist()

plt.show()


# In[56]:


strat_train_set.info()


# In[57]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
class AgeImputer (BaseEstimator, TransformerMixin):
    def fit(self,x, y=None):
        return self
    def transform(self, X):
        imputer = SimpleImputer(strategy="mean")
        X['Age']=imputer.fit_transform(X[['Age']])
        return X


# In[58]:


from sklearn.preprocessing import OneHotEncoder
class FeatureEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder = OneHotEncoder()
        matrix = encoder.fit_transform(X[['Embarked']]).toarray()
        column_names = ["C","S","Q","N"]
        for i in range (len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        matrix = encoder.fit_transform(X[['Sex']]).toarray()
        column_names = ["Female","Male"]
        for i in range (len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        return X


# In[59]:


class FeatureDropper(BaseEstimator, TransformerMixin):
    def fit (self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(["Embarked","Name","Ticket","Cabin","Sex","N"],axis=1,errors="ignore")


# In[60]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([("ageimputer", AgeImputer()),("featureencoder", FeatureEncoder()),("featurredropper", FeatureDropper())
])


# In[61]:


strat_train_set = pipeline.fit_transform(strat_train_set)


# In[62]:


strat_train_set


# In[63]:


strat_train_set.info()


# In[64]:


from sklearn.preprocessing import StandardScaler
X = strat_train_set.drop(['Survived'], axis=1)
y = strat_train_set['Survived']

scaler = StandardScaler()
X_data = scaler.fit_transform(X)
y_data = y.to_numpy()


# In[65]:


X_data


# In[66]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()

param_grid = [{"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}]

grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data, y_data)


# In[79]:


final_clf = grid_search.best_estimator_


# In[80]:


final_clf


# In[81]:


strat_test_set = pipeline.fit_transform(strat_test_set)


# In[83]:


X_test = strat_test_set.drop(['Survived'], axis=1)
y_test = strat_test_set['Survived']
scaler = StandardScaler()
X_data_test = scaler.fit_transform(X_test)
y_data_test = y_test.to_numpy()


# In[84]:


final_clf.score(X_data_test,y_data_test)


# In[85]:


final_data = pipeline.fit_transform(titanic_data)


# In[86]:


final_data


# In[93]:


X_fainal = final_data.drop(['Survived'], axis=1)
Y_final = final_data['Survived']

scaler = StandardScaler()
X_data_final = scaler.fit_transform(X_test)
y_data_final = y_final.to_numpy()


# In[94]:


prod_clf = RandomForestClassifier()

param_grid = [{"n_estimators": [10, 100, 200, 500], "max_depth": [None, 5, 10], "min_samples_split": [2, 3, 4]}]

grid_search = GridSearchCV(prod_clf, param_grid, cv=3, scoring="accuracy", return_train_score=True)
grid_search.fit(X_data_test, y_data_test)


# In[97]:


prod_final_clf = grid_search.best_estimator_


# In[98]:


prod_final_clf


# In[109]:


titanic_test_data = pd.read_csv("test.csv")


# In[123]:


final_test_data = pipeline.fit_transform(titanic_test_data)


# In[126]:


X_final_test = final_test_data
X_final_test = final_test_data.fillna(method="ffill")

scaler = StandardScaler()
X_data_final_test = scaler.fit_transform(X_final_test)


# In[127]:


predictions = prod_final_clf.predict(X_data_final_test)


# In[129]:


final_df = pd.DataFrame(titanic_test_data['PassengerId'])
final_df['Survived'] = predictions
final_df.to_csv("predictions.csv", index=False)


# In[130]:


final_df


# In[ ]:




