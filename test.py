import requests

url = 'http://localhost:8081/predict'

input_data = {
    "rownumber": 9776,
    "customerid": 15744041,
    "surname": "Yobanna",
    "creditscore": 780,
    "geography": "France",
    "gender": "Female",
    "age": 26,
    "tenure": 3,
    "balance": 140356.7,
    "numofproducts": 1,
    "hascrcard": "Yes",
    "isactivemember": "No",
    "estimatedsalary": 117144.15
}

response = requests.post(url, json=input_data).json()
print(response)

if response['exit_status'] == True:
    print('sending email to %s' %('user10'))