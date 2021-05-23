"""
To start flask server, use command 'flask run'
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

from flask import Flask, request
from flask_cors import CORS
import json
import joblib
import pandas as pd
from keras.models import load_model
import numpy as np


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

DL_MODEL = load_model('time_series_our_data.h5')
ML_MODEL = joblib.load('4.sav')

label_map = {
    0: 'Nothing',
    1: 'Standing still',
    2: 'Sitting and relaxing',
    3: 'Lying down',
    4: 'Walking',
    5: 'Climbing stairs',
    6: 'Waist bends forward',
    7: 'Frontal elevation of arms',
    8: 'Knees bending (crouching)',
    9: 'Cycling',
    10: 'Jogging',
    11: 'Running',
    12: 'Jump front & back'
}


@app.route('/')
def index():
    return 'Hello World!'


@app.route('/ml-prediction', methods=['POST'])
def ml_prediction():
    accelerometer_meas_entities = request.form['accelerometerMeasEntities']
    gyroscope_meas_entities = request.form['gyroscopeMeasEntities']
    res_acc = json.loads(accelerometer_meas_entities)
    res_gyro = json.loads(gyroscope_meas_entities)

    pred_data = {
        "acc_x": [res_acc[0]["x"]],
        "acc_y": [res_acc[0]["y"]],
        "acc_z": [res_acc[0]["z"]],
        "gyro_x": [res_gyro[0]["x"]],
        "gyro_y": [res_gyro[0]["y"]],
        "gyro_z": [res_gyro[0]["z"]],
        "measurementId": [1],
        "repetitions": [10]
    }

    df = pd.DataFrame.from_dict(pred_data)
    try:
        model = ML_MODEL
        pred = model.predict(df)
        return pred[0]
    except:
        return "Could not make prediction"


@app.route('/dl-prediction', methods=['POST'])
def dl_prediction():
    accelerometer_meas_entities = request.form['accelerometerMeasEntities']
    gyroscope_meas_entities = request.form['gyroscopeMeasEntities']
    res_acc = json.loads(accelerometer_meas_entities)
    res_gyro = json.loads(gyroscope_meas_entities)

    if len(res_acc) < 100 or len(res_gyro) < 100:
        return "Not enough data provided"

    pred_data = [[res_acc[i]["x"], res_acc[i]["y"], res_acc[i]["z"], res_gyro[i]["x"], res_gyro[i]["y"], res_gyro[i]["z"]] for i in range(len(res_acc))]

    df = np.array(pred_data)
    np.savetxt("data.csv", df, delimiter=",")
    df = df.reshape(df.shape + (1,))
    df = df[None]

    try:
        model = DL_MODEL
        pred = model.predict(df)
        pred = np.argmax(pred, axis=1)

        return label_map[pred[0]]
    except:
        return "Could not make prediction"


if __name__ == '__main__':
    app.run()
