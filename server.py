#from crypt import methods
from flask import Flask, request, jsonify
import flask
from joblib import load
from PIL import Image
import json
from werkzeug.utils import secure_filename
#from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import time

app = Flask('Identity')

# instantiate the class with the ML models 
from photo_inferencing import inferencing
photo_match = inferencing()


# instantiate the class with the scoring functionality 
from score_service import similarityScore
scoring = similarityScore()


# endpoint for API health check 
# the "ping" endpoint is one that is required by AWS 
@app.route("/ping", methods=['GET'])
def health():
    
    results =  {"API Status": 200}
    resultjson = json.dumps(results)

    return flask.Response(response=resultjson, status=200, mimetype='application/json')
    

# endpoint for matches from two photos
@app.route("/identity", methods=['POST'])
def embeddings():
    
    # retrieve reference photo
    ref_file = request.files['reference']
    ref_img = Image.open(ref_file.stream)

    # retrieve sample photo 
    sample_file = request.files['sample']
    sample_img = Image.open(sample_file.stream)

    # generate pair of tensors 
    start = time.time()

    ref_tensor, sample_tensor = photo_match.identity_verify(ref_img, sample_img)
    
    end = time.time()

    latency = end - start

    # generate score 
    score, status = generate_score(ref_tensor, sample_tensor)

    # return data 
    results = {"Match Status": status,
               "Score": score,
               "Inferencing Latency": latency}


    resultjson = json.dumps(results)

    return flask.Response(response=resultjson, status=200, mimetype='application/json')


# endpoint for presenting a pre-processed/cached tensor and a sample photo 
@app.route("/cached_data", methods=['POST'])
def cached():

    ref = request.files['reference']
    cached_tensor = torch.load(ref)

    # retrieve sample photo 
    sample_file = request.files['sample']
    sample_img = Image.open(sample_file.stream)

    # generate embeddings for sample photo 
    start = time.time()

    sample_tensor = photo_match.cached_reference(sample_img)

    end = time.time()

    latency = end - start

    # generate match score 
    score, status = generate_score(cached_tensor, sample_tensor)

    # return json data 
    results = {"Match Status": status,
               "Score": score,
               "Inferencing Latency": latency}


    resultjson = json.dumps(results)

    return flask.Response(response=resultjson, status=200, mimetype='application/json')


# method for generating a match score 
# update this to have a variable for routing to the proper score type
# add a variable for passing the desired threshold 
def generate_score(reference_tensor, sample_tensor):
    
    score, status = scoring.cosine_score(reference_tensor, sample_tensor)

    return score, status
