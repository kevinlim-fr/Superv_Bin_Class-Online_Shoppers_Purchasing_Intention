from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import pickle as p
import json
import pandas as pd

app = Flask(__name__)
api = Api(app)


with open('model.pickle', 'rb') as f_model:
    model = p.load(f_model) # trained classifier model
with open('column_trans.pickle', 'rb') as f_onehotenc:
    ohe = p.load(f_onehotenc) # OneHotEncoder object that was used in training
with open('scaler.pickle', 'rb') as f_scaler:
    scaler = p.load(f_scaler) # OneHotEncoder object that was used in training

a ="Administrative,Administrative_Duration,Informational,Informational_Duration,ProductRelated,ProductRelated_Duration,BounceRates,ExitRates,PageValues,SpecialDay,Month,OperatingSystems,Browser,Region,TrafficType,VisitorType,Weekend"
columns = a.split(",")

# argument parsing

class PredictPurchase(Resource):

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('query',type=str)
        args = parser.parse_args()
        # use parser and find the user's query
        arg = args['query']
        val = arg.split(",")

        if val[16] == "FALSE":
            val[16] = False
        else:
            val[16] = True

        df = pd.DataFrame([val],columns=columns)

        for col in df.columns[:6]:
            df[col] = df[col].astype('float64')
        for col in df.columns[6:8]:
            df[col] = df[col].astype('float64')
        for col in df.columns[8:10]:
            df[col] = df[col].astype('float64')
        for col in df.columns[11:15]:
            df[col] = df[col].astype('float64')

        df = ohe.transform(df)
        df = scaler.transform(df)
        df = df.tolist()

        prediction = model.predict(df)

        # Output 'Negative' or 'Positive' along with the score
        print(prediction)
        if prediction == False:
            pred_text = 'Not purchased'
        else:
            pred_text = 'Purchased'

        return pred_text

api.add_resource(PredictPurchase, '/')


if __name__ == '__main__':
    app.run(debug=True)
