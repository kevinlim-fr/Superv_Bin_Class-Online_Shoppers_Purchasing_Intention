import requests
import json
import pickle as p
import pandas as pd
import csv
url = 'http://127.0.0.1:5000/api/'

with open('model.pickle', 'rb') as f_model:
    model = p.load(f_model) # trained classifier model
with open('column_trans.pickle', 'rb') as f_onehotenc:
    ohe = p.load(f_onehotenc) # OneHotEncoder object that was used in training
with open('scaler.pickle', 'rb') as f_scaler:
    scaler = p.load(f_scaler) # OneHotEncoder object that was used in training

a ="Administrative,Administrative_Duration,Informational,Informational_Duration,ProductRelated,ProductRelated_Duration,BounceRates,ExitRates,PageValues,SpecialDay,Month,OperatingSystems,Browser,Region,TrafficType,VisitorType,Weekend"
columns = a.split(",")

b = "0,0,0,0,1,0,0.2,0.2,0,0,Feb,1,1,1,1,Returning_Visitor,FALSE"
val = b.split(",")

if val[16] == "FALSE":
    val[16] = False
else: 
    val[16] = True

df = pd.DataFrame([val],columns=columns)

for col in df.columns[:6]:
    df[col] = df[col].astype('int64')
for col in df.columns[6:8]:
    df[col] = df[col].astype('float64')
for col in df.columns[8:10]:
    df[col] = df[col].astype('int64')
for col in df.columns[11:15]:
    df[col] = df[col].astype('int64')

df = ohe.transform(df)
df = scaler.transform(df)
df = df.tolist()

j_data = json.dumps(df)
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)