from flask import Flask, render_template, request, redirect
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

MLapp = Flask(__name__)
model = pickle.load(open('model.pkl', "rb")) 

@MLapp.route('/')
def home():
    return render_template("index.html")

@MLapp.route('/predict', methods=['POST'])
def predict():
    float_feature = [float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The predicted crop is: {}".format(prediction))

if __name__ ==  '__main__':
    MLapp.run(debug=True)
