import pandas as pd
import numpy as np
import json

# get the measurement ids
with open("measurements_id.json") as file:
        id_data = json.load(file)
idx_list = []
for el in id_data:
  idx_list.append(el['id'])

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

# measurementId to string value and repetitions
# #dont_judge_my_code_if_it_works
combined_df['activity'] = combined_df['measurementId'].map({
  1.0: 'Push ups',
  3.0: 'Walking',
  4.0: 'Push ups',
  5.0: 'Push ups',
  6.0: 'Push ups',
  7.0: 'Push ups',
  8.0: 'Squats',
  9.0: 'Frontal elevation of arms',
  11.0: 'Jump front & back',
  13.0: 'Walking',
  14.0: 'Jump front & back',
  15.0: 'Frontal elevation of arms',
  16.0: 'Walking',
  18.0: 'Walking',
  20.0: 'Push ups',
  22.0: 'Walking',
  23.0: 'Push ups',
  24.0: 'Frontal elevation of arms',
  25.0: 'Frontal elevation of arms',
  26.0: 'Push ups',
  27.0: 'Walking',
  28.0: 'Push ups',
})
combined_df['repetitions'] = combined_df['measurementId'].map({
  1.0: 41,
  3.0: 2,
  4.0: 10,
  5.0: 10,
  6.0: 10,
  7.0: 10,
  8.0: 7,
  9.0: 3,
  11.0: 14,
  13.0: 7,
  14.0: 99,
  15.0: 75,
  16.0: 102,
  18.0: 44,
  20.0: 22,
  22.0: 2,
  23.0: 252,
  24.0: 39,
  25.0: 2137, # how did (s)he get exactly 2137 repetitions
  26.0: 1488, # Is this the real life?
  27.0: 9000, # Is this just fantasy?
  28.0: 1234, # i dont believe they are real values
})
# float to int and reorder
combined_df['measurementId'] = combined_df['measurementId'].apply(np.int64)
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
