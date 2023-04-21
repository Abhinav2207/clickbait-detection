from flask import Flask, request, jsonify
from keras.models import load_model
from src.models.convnets import ConvolutionalNet
from src.preprocessors.preprocess_text import clean
from keras.utils import pad_sequences
from flask_cors import CORS
import re
import json

MATCH_MULTIPLE_SPACES = re.compile("\ {2,}")
SEQUENCE_LENGTH = 20
EMBEDDING_DIMENSION = 30

UNK = "<UNK>"
PAD = "<PAD>"

vocabulary = open("data/vocabulary.txt").read().split("\n")
inverse_vocabulary = dict((word, i) for i, word in enumerate(vocabulary))

def words_to_indices(inverse_vocabulary, words):
    return [inverse_vocabulary.get(word, inverse_vocabulary[UNK]) for word in words]

class Predictor (object):
    def __init__(self, model_path):
        model = ConvolutionalNet(vocabulary_size=len(vocabulary), embedding_dimension=EMBEDDING_DIMENSION, input_length=SEQUENCE_LENGTH)
        model.load_weights(model_path)
        self.model = model
    
    def predict (self, headline):
        headline = headline.encode("utf-8").decode("ascii", "ignore")
        inputs = pad_sequences([words_to_indices(inverse_vocabulary, clean(headline).lower().split())], maxlen=SEQUENCE_LENGTH)
        clickbaitiness = self.model.predict(inputs)[0, 0]
        return clickbaitiness
predictor = Predictor("models/detector.h5")

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])

def predict():
    response = request.get_json(force=True)
    headline=response['data']
    output=[]
    for i in headline:
        output.append({i:predictor.predict(i) * 100})
    return jsonify(output)