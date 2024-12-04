from xml.dom.minidom import Document

from flask import Flask, jsonify, request, flash
from flask_cors import CORS, cross_origin
from transformers import pipeline
from werkzeug.utils import secure_filename

from tryloadmodel  import  main_t
from tryloadmodel import AttentionLayer
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from keras.layers import Bidirectional
import os
from keras.models import model_from_json
from keras.layers import Bidirectional, Embedding, LSTM


app = Flask(__name__)
CORS(app, supports_credentials=True)
encoder=None
decoder=None
PipTransletionToEnglish = None
PipTransletionToEbrowe = None




@app.route("/translate", methods=['GET'])
@cross_origin(supports_credentials=True)
def getSummeraztion():
    st = request.args.get('st')

    transStrEn = translationToEn(st) #english sentence
    sumeraztiontext=main_t(transStrEn,encoder,decoder)

    sumeraztiontext=str(sumeraztiontext)
    sumeraztiontext=translationToEb(sumeraztiontext)
    return  jsonify({"translation": sumeraztiontext})


def translationToEn(st):

    return (PipTransletionToEnglish(st))


def translationToEb(st):
    return (PipTransletionToEbrowe(st))


def load_model(model_filename, model_weights_filename):
    f = open(model_filename, 'r', encoding='utf8')
    model = model_from_json(f.read())
    model.load_weights(model_weights_filename)
    return model


def loadallmodels():

    loadEncoder_Decoder()
    global PipTransletionToEnglish
    global  PipTransletionToEbrowe
    PipTransletionToEnglish = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-he-en")
    PipTransletionToEbrowe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-he")
def load_dec(model_filename, model_weights_filename):
    with open(model_filename, "r") as json_file:
        model_json = json_file.read()

    # Create a new model from the JSON content
    new_model = tf.keras.models.model_from_json(model_json , custom_objects={'AttentionLayer': AttentionLayer})

    # Load the model weights from the H5 file
    new_model.load_weights(model_weights_filename)
    json_file.close()
    return new_model



def load_enc(model_filename, model_weights_filename):

    # שלב 1: טעינת מבנה המודל מה-JSON
    with open(model_filename, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)

    # שלב 2: טעינת המשקלים מה-H5
    loaded_model.load_weights(model_weights_filename)
    json_file.close()
    return loaded_model

def loadEncoder_Decoder():

    tf.keras.utils.get_custom_objects()['AttentionLayer'] = AttentionLayer
    global encoder
    global decoder

    decoder = load_dec("moduls/new_model/decoder_bilstm2.json",
                           'moduls/new_model/dec_W3.h5')
    encoder = load_enc("moduls/new_model/encoder_bilstm.json",
                         'moduls/new_model/enc_W3.h5')









if __name__ == "__main__":
    app.secret_key = 'super secret key'
    # app.config['SESSION_TYPE'] ='filesystem'
    loadallmodels()
    app.run()
