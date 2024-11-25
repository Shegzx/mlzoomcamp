import pickle
import numpy as np
from flask import Flask, request, jsonify

#Load the model
with open('model.bin', 'rb') as file_in:
    dv,model = pickle.load(file_in)

app = Flask('exit_status')

@app.route('/')
def home():
    return "Model API"

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    X = dv.transform([input_data])
    y_pred = model.predict(X)[0,1]

    return jsonify({'prediction': float(y_pred[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8081)
