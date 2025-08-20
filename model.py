from flask import Flask, render_template, request, redirect
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

import pandas as pd


data = pd.read_csv('Crop_recommendation.csv')

x = data.iloc[:,:-1].values #features
y = data.iloc[:,-1].values #target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

model = RandomForestClassifier()

model.fit(x_train,y_train)

#Make predictions on test model
#predictions = model.predict(x_test)

#Accuracy of the model
#model_accuracy = model.score(x_test, y_test)
#print(f'Model Accuracy: {model_accuracy}')

pickle.dump(model, open('model.pkl', 'wb'))