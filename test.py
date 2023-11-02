#!/usr/bin/env python
# coding: utf-8

import requests

url = "http://localhost:9696/predict"

code_params = {
    "loc": 33.0, 
    "v(g)": 5.0, 
    "ev(g)": 1.00, "iv(g)": 4.00, 
    "n": 144.00, "v": 824.82, "l":0.04, "d":26.96, "i":30.05, "e":22636.74,
    "b":0.27, "t":1257.60, "lOCode":3.00, "lOComment":0.00, "lOBlank":3.00, 
    "locCodeAndComment":0.00, "uniq_Op":21.00, "uniq_Opnd":23.00, "total_Op":87.00, 
    "total_Opnd":57.00, "branchCount":9.00
}
response = requests.post(url, json=code_params).json()

print(response)
