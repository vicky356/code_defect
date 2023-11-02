#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


df_full = pd.read_csv('train.csv')

numerical = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e',
       'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment',
       'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']

df_full['defects'] = df_full['defects'].astype(int)

df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=1)

y_train = df_train.defects.values
y_test = df_test.defects.values


def train(df, y_train):
    dicts = df[numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    features = dv.get_feature_names_out()
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
    
    xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'min_child_weight':30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1
    }

    model = xgb.train(xgb_params, dtrain, num_boost_round=160)
    
    return dv, model



def predict(df, dv, model):
    dicts = df[numerical].to_dict(orient='records')
    
    X = dv.fit_transform(dicts)

    dpredict = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())

    y_pred = model.predict(dpredict)
    
    return y_pred

dv, model = train(df_train, y_train)
y_pred = predict(df_test, dv, model)
auc = roc_auc_score(y_test, y_pred)

print(f'validation auc score: {auc}')

dv, model = train(df_full, df_full.defects.values)
y_pred = predict(df_full, dv, model)
auc = roc_auc_score(df_full.defects.values, y_pred)

print(f'Final model auc score: {auc}')

output_file = 'model.bin'

# Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is saved to {output_file}')

