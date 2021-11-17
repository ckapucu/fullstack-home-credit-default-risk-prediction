# Author Ceyhun Kapucu
import os
from flask import Flask, request, jsonify, render_template, current_app, send_file
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin

# Flask: to create a Flask app
# request: to get incoming request
# jsonify: to return JSON object
# render_template: to render HTML page

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

load_dotenv()

"""
In Heroku for dealing timeout problem while loading LightGbm model, I deleted --preload parameter in Procfile and 
changed --timeout 1200 to --timeout 120. 
"""

import numpy as np

# to deserializaiton as Pickle
import joblib
# import pickle

from flask_compress import Compress

import time

# from math import exp, log
# from utils import plotPrediction

# create the app
app = Flask(__name__, static_folder='./public/', static_url_path='/')
# Compress(app)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# load the pre-trained model 

# lets define the loaded model as a global variable to prevent loading in every request
global clr

clr = joblib.load("model/model_voting_joblib.pkl")

# clr = pickle.load(open("model/model_cart_joblib.pkl", "rb"))


@app.route('/')
def home():
    # app's entry point in public folder (it is automatically created after building VueJs app)
    return app.send_static_file('index.html')


@app.route('/predict/<data>', methods=['GET', 'POST'])
def predict(data):
    # Prepare data
    data_list = data.split('&')

    # check the size of the incoming data
    # if len(data_list) != 5:
    #    return jsonify(error="wrong data input")

    rawData = np.array([(eval(data_list[0]), eval(data_list[1]), eval(data_list[2]), eval(data_list[3]),
                         eval(data_list[4]), eval(data_list[5]), eval(data_list[6]), eval(data_list[7]),
                         eval(data_list[8]), eval(data_list[9]))], dtype=float)

    # start timer
    s = time.time()

    # parsing the features and using them in feature engineering
    AMT_ANNUITY = rawData[0][0]
    AMT_CREDIT = rawData[0][1]
    DAYS_EMPLOYED = rawData[0][2]
    EXT_SOURCE_1 = rawData[0][3]
    EXT_SOURCE_2 = rawData[0][4]
    EXT_SOURCE_3 = rawData[0][5]
    DAYS_BIRTH = rawData[0][6]
    DAYS_LAST_PHONE_CHANGE = rawData[0][7]
    AMT_GOODS_PRICE = rawData[0][8]
    DAYS_ID_PUBLISH = rawData[0][9]

    # Feature engineering section
    NEW_ANNUITY_OVER_CREDIT = 0

    if AMT_CREDIT > 0:
        NEW_ANNUITY_OVER_CREDIT = AMT_ANNUITY / AMT_CREDIT

    NEW_EXT_1 = 0

    if EXT_SOURCE_3 > 0:
        NEW_EXT_1 = EXT_SOURCE_2 / EXT_SOURCE_3

    NEW_EXT1_TO_BIRTH_RATIO = 0
    NEW_AGE_OVER_WORK = 0

    if DAYS_BIRTH != 0:
        NEW_EXT1_TO_BIRTH_RATIO = EXT_SOURCE_1 / (DAYS_BIRTH / 365)
        NEW_AGE_OVER_WORK = DAYS_EMPLOYED / (DAYS_BIRTH / 365)

    # These features values will come from VueJs front-end.
    inputData = np.array([NEW_ANNUITY_OVER_CREDIT, DAYS_EMPLOYED, NEW_EXT_1, EXT_SOURCE_2, NEW_EXT1_TO_BIRTH_RATIO,
                          DAYS_LAST_PHONE_CHANGE,
                          AMT_GOODS_PRICE, AMT_CREDIT, DAYS_ID_PUBLISH, NEW_AGE_OVER_WORK], dtype=float)

    # to prevent predict_proba function's raising a warning alert
    inputData = inputData.reshape(1, -1)

    # Predict probability for classes
    rawPrediction = clr.predict_proba(inputData)[0]
    predictionIndex = np.argmax(rawPrediction, axis=0)
    predictionAccuracy = rawPrediction[predictionIndex]
    predictionValue = predictionIndex

    # stop timer
    e = time.time()
    predictionTime = e - s

    return jsonify(predicted_case=str(predictionValue), run_time=predictionTime,
                   prediction_probability=str(predictionAccuracy),
                   case0_probability=rawPrediction[0], case1_probability=rawPrediction[1])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
    # app.run()
