# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:45:50 2020

@author: shalabh.yadu
"""

import os 

#os.chdir(r'C:\Users\shalabh.yadu\webapplication-doc\mlflow_app')


import os
import warnings
import sys
#import argparse
from flask import Flask,render_template,url_for,request,Response
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


np.random.seed(40)

#wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
data = pd.read_csv("wine-quality.csv")
    
    
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]

alpha = 0.3
l1_ratio = 0.01
#mlflow.end_run()

mlflow.start_run()
#mlflow.run("https://0.0.0.0:5000")
lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
lr.fit(train_x, train_y)

#test_x.to_csv("pred.csv", index = False)


predicted_qualities = lr.predict(test_x)

(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
print("  RMSE: %s" % rmse)
print("  MAE: %s" % mae)
print("  R2: %s" % r2)

mlflow.log_param("alpha", alpha)
mlflow.log_param("l1_ratio", l1_ratio)
mlflow.log_metric("rmse", rmse)
mlflow.log_metric("r2", r2)
mlflow.log_metric("mae", mae)

mlflow.sklearn.log_model(lr, "model")


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        predicted_qualities = lr.predict(df)
        predicted_qualities = ",\n".join([str(i) for i in list(predicted_qualities)])
        #(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
        
        return Response(
        predicted_qualities,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=output.csv"})
    return render_template('home.html')


#

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=4000)

#mlflow ui  
#mlflow.end_run("https://0.0.0.0:5000")   
mlflow.end_run()
#mlflow.set_tracking_uri("https://0.0.0.0:5000")
#df = pd.read_csv('pred.csv')
#mlflow ui