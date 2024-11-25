import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import pickle

output_file = 'model.bin'

data = pd.read_csv('Churn_Modelling.csv')
data.columns = data.columns.str.lower().str.replace(' ','_')

data.value_counts().sum()
data.isnull().sum()

#change dtype of customers to leave bank
data.exited = data.exited.map({
    1:'No', 0: 'Yes'
})
data.hascrcard = data.hascrcard.map({
    1:'Yes', 0:'No'
})
data.isactivemember = data.isactivemember.map({
    1:'Yes', 0:'No'
})

data.exited.value_counts()

data.hascrcard.value_counts()

data.isactivemember.value_counts()

data.head()

data.describe().round(2)

data.dtypes

"""Predicting customer churn targeting exited column"""

data_exited = data.groupby(['customerid'])['exited'].sum().reset_index()

data_exited.values

#spliting data for training
data_train, data_test = train_test_split(data, test_size=0.2,random_state=1)
data_train, data_val = train_test_split(data_train, test_size=0.25, random_state=1)

#x data
data_train = data_train.reset_index(drop=True)
data_test = data_test.reset_index(drop=True)
data_val = data_val.reset_index(drop=True)

#y data
y_train = data_train.exited.values
y_test = data_test.exited.values
y_val = data_val.exited.values

data_train.shape, data_test.shape, data_val.shape

y_train.shape, y_test.shape, y_val.shape

del data_train['exited']
del data_test['exited']
del data_val['exited']

data_train.shape, data_test.shape, data_val.shape

data.nunique()

#Feature Matrix
dv = DictVectorizer(sparse=False)
train_dicts = data_train.to_dict(orient = 'records')
X_train = dv.fit_transform(train_dicts)
X_train

#Validation feature matrix
val_dicts = data_val.to_dict(orient = 'records')
X_val = dv.transform(val_dicts)

#Train using logistic regression model
lr = LogisticRegression(solver='liblinear', max_iter = 1000, random_state=1)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict_proba(X_val)[:,1]
y_pred_lr

roc_auc_score_lr = roc_auc_score(y_val, y_pred_lr)

#Feature Importance
feature_names = dv.get_feature_names_out()
coefficients = lr.coef_[0]
feature_importance = dict(zip(feature_names, coefficients.round(2)))
feature_importance

feature_importance.values()

#Visualise Feature Importance
features = list(feature_importance.keys())
coefficients = list(feature_importance.values())

# training for decision tree model
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(X_train,y_train)
y_pred_dt = dt.predict_proba(X_val)[:,1]
roc_auc_score_dt =roc_auc_score(y_val,y_pred_dt)

#training using randomforest model
rf = RandomForestClassifier(n_estimators=10,max_depth=5, random_state=1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_val)[:,1]
roc_auc_score_rf=roc_auc_score(y_val,y_pred_rf)

#training using XGBoost
X_train.dtype, X_val.dtype
y_train.dtype, y_val.dtype

y_train = (y_train =='Yes').astype(int)
y_val = (y_val=='Yes').astype(int)

feature_names = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names= feature_names)
dval = xgb.DMatrix(X_val,label=y_val, feature_names=feature_names)

watchlist = [(dtrain, 'train'), (dval, 'val')]

#parameters for xgboost model
#Baseline Model

xgb_params = {
    'max_depth': 6,
    'verbosity': 1,
    'eta': 0.3,
    'eval_metric': 'auc',
    'min_child_weight': 1,
    'objective':'binary:logistic',
    'nthread': 8,
    'seed': 1,
}

model = xgb.train(xgb_params, dtrain, evals=watchlist, num_boost_round=200, verbose_eval=2, early_stopping_rounds=10)

y_pred_xgb = model.predict(dval)
roc_auc_score_xgb=roc_auc_score(y_val,y_pred_xgb)
roc_auc_score_xgb

print('Best Iteration', model.best_iteration)
print('Best Score', '%.4f' %(model.best_score))

#Choosing the best model then carry out parameter tuning to improve performance
print('###############')
print('roc_score of models used for training...')
print('LogisticRegression:', roc_auc_score_lr.round(4))
print('DecisionTreeClassifier:', roc_auc_score_dt.round(4))
print('RandomForest:', roc_auc_score_rf.round(4))
print('XGBoost:', roc_auc_score_xgb.round(4))
print('###############')

"""Best Model Used: XGBoost"""

#Hyperparameter tuning

param_grid = {
    'max_depth': [3,5,7],    #range of parameters reduced due to  delay in model fitting...
    'eta': [0.1,0.3,0.5],
    'min_child_weight': [3,5,7],
}

xgb_tune_model = xgb.XGBClassifier(objective='binary:logistic')

gs = GridSearchCV(xgb_tune_model, param_grid, cv=3, scoring='roc_auc', verbose=1)
gs.fit(X_train,y_train)

#gs.best_params_

gs.best_estimator_ #eta = 0.1, max_depth = 3, min_child_weight = 3

#Test model

test_dicts = data_test.to_dict(orient = 'records')
X_test = dv.transform(test_dicts)
model = gs.best_estimator_
y_pred = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test,y_pred)
auc.round(3)

"""Model Deployment"""
model.save_model(output_file)
output_file

file_out = open(output_file, 'wb')
pickle.dump((dv,model), file_out)
file_out.close()

with open(output_file, 'wb') as file_out:
    pickle.dump((dv,model), file_out)
