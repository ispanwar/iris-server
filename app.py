from flask import Flask,request,jsonify
import joblib
import numpy as np
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

model = joblib.load('iris_model.pkl')


@app.route('/')
def home():
    return 'Welcome to the iris predictor'


@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']
    features = np.array(features).reshape(1,-1)
    prediction = model.predict(features)
    return jsonify({"prediction":int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
