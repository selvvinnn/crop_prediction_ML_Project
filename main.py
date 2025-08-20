from flask import Flask, render_template, request, redirect
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

MLapp = Flask(__name__)

@MLapp.route('/predict')
def predict():
    pass


if __name__ ==  '__main__':
    MLapp.run(debug=True)