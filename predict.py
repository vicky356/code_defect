#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify

with open('dv.bin', 'rb') as f_in:
    dv = pickle.load(f_in)

model = xgb.Booster()
model.load_model('model.bin')

app = Flask('defect')

@app.route('/predict', methods=['POST'])
def defect():
    code_params = request.get_json()
    
    X = dv.transform(code_params)
    X_pred = xgb.DMatrix(X, feature_names=dv.get_feature_names_out().tolist())

    y_pred = np.squeeze(model.predict(X_pred))
    result = {"defect": float(y_pred)}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)




