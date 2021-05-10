import requests
import json
import csv
import pandas as pd

#-----------------------GET THE LIST OF MEASUREMENT IDS-----------------------------#

# get the measurement ids
measurements_id_response = requests.get("http://pawelkob-002-site3.itempurl.com/api/measurements-dev")
json_response = measurements_id_response.json()

json_response = json_response['payload']['measurements']
print(json_response)

# create a list od ids
list_of_ids = []
for l in range(len(json_response)):
    list_of_ids.append(json_response[l]['id'])

# save measurement ids in json file
out_file = open("measurements_id.json", "w")
json.dump(json_response, out_file, indent = 6)
out_file.close()

# covert from json to csv file
df = pd.read_json ('measurements_id.json')
export_csv = df.to_csv ('measurements_id.csv', index = None, header=True)


#-----------------------GET DETAILS FROM MEASUREMENT ID-----------------------------#

# get details from measurement id
for l in list_of_ids:

    measurement_details_response = requests.get("http://pawelkob-002-site3.itempurl.com/api/measurements-dev/" + str(l))
    json_details_response = measurement_details_response.json()

    # save measurement details from one id in temporary json file
    out_file = open("temp_measurement_details.json", "w")
    json.dump(json_details_response, out_file, indent = 6)
    out_file.close()

    with open('temp_measurement_details.json') as json_details_file:
        data = json.load(json_details_file)
    
    # save accelerometer and gyroscope details in two separete json files
    accelerometer_data = data['payload']['accelerometerMeasurements']
    gyroscope_data = data['payload']['gyroscopeMeasurements']

    out_accel_file = open("accelerometer_details.json", "a+")
    json.dump(accelerometer_data, out_accel_file, indent = 6)
    out_accel_file.close()

    out_gyro_file = open("gyroscope_details.json", "a+")
    json.dump(gyroscope_data, out_gyro_file, indent = 6)
    out_gyro_file.close()

    print("Details from measurement id: " + str(l) + " added")

#-----------------------CONVERT TO CSV-----------------------------#

list_of_jsons = ['accelerometer_details.json', 'gyroscope_details.json']

for json_file in list_of_jsons:

    with open(json_file) as f:
        file_data = f.read()

    file_data = file_data.replace('][', ',')

    with open(json_file, 'w') as f:
        f.write(file_data)

    json_data = json.loads(file_data)
    file_name, extension = json_file.split('.')

    df = pd.read_json (json_file)
    export_csv = df.to_csv (file_name + '.csv', index = None, header=True)