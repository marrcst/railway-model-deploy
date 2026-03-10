#Start with imports needed
import os
import json
import pickle
import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    Model, IntegerField, FloatField,
    TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict
from playhouse.db_url import connect


#Initialize Database ##################################
DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):
    observation_id = IntegerField(unique = True)
    observation = TextField()
    proba = FloatField()
    true_class = IntegerField(null = True)

    class Meta:
        database = DB

DB.create_tables([Prediction], safe=True)
#######################################################


#Deserializa col names, col dtypes and pipeline #######
with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)

#######################################################

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    obs_dict = request.get_json()

    _id = obs_dict['id']
    observation = obs_dict['observation']

    try:
        obs = pd.DataFrame([observation], columns = columns).astype(dtypes)
    except Exception:
        #If any of the columns has an incorrect type, an error is raised
        return jsonify({"error": "Observation is invalid!"})

    proba = pipeline.predict_proba(obs)[0, 1]
    response = {'proba': proba}

    p = Prediction(
        observation_id = _id,
        proba = proba,
        observation = request.data,
        true_class = None
    )

    try:
        p.save()

    except IntegrityError:
        error_msg = f'Observation ID {_id} already exists'
        response['error'] = error_msg
        print(error_msg)
        # DB.rollback()

    return jsonify(response)


@app.route("/update", methods=['POST'])
def update():
    obs_dict = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs_dict['id'])
        p.true_class = obs_dict['true_class']
        p.save()
        return jsonify(model_to_dict(p))

    except Prediction.DoesNotExist:
        error_msg = f"Observation ID {obs_dict['id']} does not exist"
        return jsonify({'error': error_msg})


# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)
