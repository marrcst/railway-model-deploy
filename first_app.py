from flask import Flask, request, jsonify
import joblib
import json
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():

    prediction = 0.5
    return jsonify({
        'prediction': prediction
    })
    
if __name__ == "__main__":
    app.run(debug=True)
