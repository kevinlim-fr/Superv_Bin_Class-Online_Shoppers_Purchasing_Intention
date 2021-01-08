import requests
import json

url = 'http://0.0.0.0:5000/api/'

with open('model.pickle', 'rb') as f_model:
    model = pickle.load(f_model) # trained classifier model
with open('column_trans.pickle', 'rb') as f_onehotenc:
    ohe = pickle.load(f_onehotenc) # OneHotEncoder object that was used in training
with open('scaler.pickle', 'rb') as f_scaler:
    scaler = pickle.load(f_scaler) # OneHotEncoder object that was used in training    

data = [[0,0,0,0,1,0,0.2,0.2,0,0,Feb,1,1,1,1,Returning_Visitor,FALSE]]
data = ohe.transform(data)
data = scaler.transform(data)

j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)