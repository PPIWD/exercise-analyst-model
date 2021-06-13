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

    measurements = {
    "acc": pd.read_json(accelerometer_meas_entities),
    "gyro": pd.read_json(gyroscope_meas_entities)
    }

    for name in ["acc", "gyro"]:
      measurements[name].sort_values(by=['timestampUtc'], ascending=[True], inplace=True)
      measurements[name] = measurements[name].reset_index(drop=True) # update index

    df_resampled = {}
    df_interpolated = {}
    for name in ["acc", "gyro"]:
      # resample
      df_resampled[name] = measurements[name].set_index('timestampUtc').resample('10ms').mean()
      # interpolate
      df_interpolated[name] = df_resampled[name].interpolate('linear')
      # round
      for direction in ['x', 'y', 'z']:
          df_interpolated[name][direction] = df_interpolated[name][direction].round(decimals=6)
      # rename x to gyro_x, acc_x etc
      df_interpolated[name].rename({"x": f"{name}_x", "y": f"{name}_y", "z": f"{name}_z"}, axis='columns', inplace = True, errors='raise')
    # remove duplicate columns
    df_interpolated['gyro'] = df_interpolated['gyro'].drop(columns=['id'])
    df_interpolated['acc'] = df_interpolated['acc'].drop(columns=['id'])
    # merge acc and gyro data by resampled timestamp
    df_interpolated_both = df_interpolated['acc'].merge(df_interpolated['gyro'], how='inner', on='timestampUtc')
    df_interpolated_both.reset_index(inplace = True)

    df_interpolated_both = df_interpolated_both.drop(columns=['measurementId_x'])
    df_interpolated_both = df_interpolated_both.drop(columns=['measurementId_y'])
    df_interpolated_both = df_interpolated_both.drop(columns=['timestampUtc'])

    # df_interpolated_both["measurementId"] = 1
    # df_interpolated_both["repetitions"] = 10

    try:
        model = ML_MODEL
        pred = list(model.predict(df_interpolated_both))
        return str(max(set(pred), key=pred.count))
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