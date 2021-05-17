import pandas as pd
import numpy as np
import json

# get the measurement ids
with open("measurements_id.json") as file:
        id_data = json.load(file)
idx_list = []
id2activity = {}
id2repetitions = {}
for el in id_data:
  idx_list.append(el['id'])
  id2activity[el['id']] = el['activity']
  id2repetitions[el['id']] = el['repetitions']

measurements = {
    "acc": pd.read_json('accelerometer_details.json'),
    "gyro": pd.read_json('gyroscope_details.json')
}

# Sort by timestamp while preserving measurementId
for name in ["acc", "gyro"]:
    measurements[name].sort_values(by=['measurementId', 'timestampUtc'], ascending=[True, True], inplace=True)
    measurements[name] = measurements[name].reset_index(drop=True) # update index

list_to_combine = []
for measurement_id in idx_list:
  df_filtered = {}
  df_resampled = {}
  df_interpolated = {}
  for name in ["acc", "gyro"]:
    # Base
    df_filtered[name] = measurements[name][measurements[name]['measurementId'] == measurement_id]
    # resampling
    df_resampled[name] = df_filtered[name].set_index('timestampUtc').resample('10ms').mean()
    # interpolation
    df_interpolated[name] = df_resampled[name].interpolate('linear')
    # rounding
    for direction in ['x', 'y', 'z']:
        df_interpolated[name][direction] = df_interpolated[name][direction].round(decimals=6)
    # renaming x to gyro_x, acc_x etc
    df_interpolated[name].rename({"x": f"{name}_x", "y": f"{name}_y", "z": f"{name}_z"}, axis='columns', inplace = True, errors='raise')
  # remove duplicate columns
  df_interpolated['gyro'] = df_interpolated['gyro'].drop(columns=['measurementId', 'id'])
  df_interpolated['acc'] = df_interpolated['acc'].drop(columns=['id'])
  # merge acc and gyro data by resampled timestamp
  df_interpolated_both = df_interpolated['acc'].merge(df_interpolated['gyro'], how='inner', on='timestampUtc')
  list_to_combine.append(df_interpolated_both)
combined_df = pd.concat(list_to_combine)
combined_df.reset_index(inplace = True)

# float to int and reorder
combined_df['measurementId'] = combined_df['measurementId'].apply(np.int64)

# measurementId to string value and repetitions
combined_df['activity'] = combined_df['measurementId'].map(id2activity)
combined_df['repetitions'] = combined_df['measurementId'].map(id2repetitions)
combined_df = combined_df[[
  'timestampUtc',
  'acc_x',
  'acc_y',
  'acc_z',
  'gyro_x',
  'gyro_y',
  'gyro_z',
  'measurementId',
  'activity',
  'repetitions'
]]

# save to a csv file
combined_df.to_csv('exercise_data.csv', index=False)
