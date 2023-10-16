import requests

url = 'http://localhost:9696/predict'

client_id = 'first'
client = {"job": "retired", "duration": 445, "poutcome": "success"}
response = requests.post(url,json=client).json()
print(response)

if response['churn'] == True:
    print('sending promo email to %s' % client_id)
else:
    print('not sending promo email to %s' % client_id)