import requests
import json
import pickle as p
import pandas as pd

url = 'http://0.0.0.0:5000/api/'

with open('model.pickle', 'rb') as f_model:
    model = p.load(f_model) # trained classifier model
with open('column_trans.pickle', 'rb') as f_onehotenc:
    ohe = p.load(f_onehotenc) # OneHotEncoder object that was used in training
with open('scaler.pickle', 'rb') as f_scaler:
    scaler = p.load(f_scaler) # OneHotEncoder object that was used in training

data = [[0,0,0,0,1,0,0.2,0.2,0,0,"Feb",1,1,1,1,"Returning_Visitor","FALSE"]]
df = pd.DataFrame(data=data)
df = ohe.transform(df)
df = scaler.transform(df)

j_data = json.dumps(data)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)
