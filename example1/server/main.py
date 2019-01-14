import pickle
import flask
from flask import Flask, request, jsonify

app = flask.Flask(__name__)

#getting our trained model from a file we created earlier
model = pickle.load(open("../train_model/model.pkl","rb"))

@app.route('/hello', methods=['POST'])
def index():
    #grabs the data tagged as 'name'
    name = request.get_json()['name']
    
    #sending a hello back to the requester
    #return jsonify(request.json)
    return "Hello " + name

@app.route('/predict', methods=['POST'])
def predict():
    #grabbing a set of wine features from the request's body
    feature_array = request.get_json()['feature_array']
    
    #our model rates the wine based on the input array
    prediction = model.predict([feature_array]).tolist()
    
    #preparing a response object and storing the model's predictions
    response = {}
    response['predictions'] = prediction
    
    #sending our response object back as json
    return flask.jsonify(response)