import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import model_from_json
import pickle
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
from keras.preprocessing.sequence import pad_sequences
import numpy as np
# from tensorflow import keras
from keras.layers import TimeDistributed
from keras.models import Model

from flask import Flask, jsonify, request, flash
from flask_cors import CORS, cross_origin
from transformers import pipeline
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json
from keras.layers import Bidirectional
import os
from keras.models import model_from_json
from keras.layers import Bidirectional, Embedding, LSTM



logger = tf.get_logger()

#CLASS של שיכבת ה ATTENTION!
class AttentionLayer(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs

        logger.debug(f"encoder_out_seq.shape = {encoder_out_seq.shape}")
        logger.debug(f"decoder_out_seq.shape = {decoder_out_seq.shape}")

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state
            inputs: (batchsize * 1 * de_in_dim)
            states: (batchsize * 1 * de_latent_dim)
            """

            logger.debug("Running energy computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch size * en_seq_len * latent_dim
            W_a_dot_s = K.dot(encoder_full_seq, self.W_a)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim

            logger.debug(f"U_a_dot_h.shape = {U_a_dot_h.shape}")

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            Ws_plus_Uh = K.tanh(W_a_dot_s + U_a_dot_h)

            logger.debug(f"Ws_plus_Uh.shape = {Ws_plus_Uh.shape}")

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.squeeze(K.dot(Ws_plus_Uh, self.V_a), axis=-1)
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            logger.debug(f"ei.shape = {e_i.shape}")

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """

            logger.debug("Running attention vector computation step")

            if not isinstance(states, (list, tuple)):
                raise TypeError(f"States must be an iterable. Got {states} of type {type(states)}")

            encoder_full_seq = states[-1]

            # <= batch_size, hidden_size
            c_i = K.sum(encoder_full_seq * K.expand_dims(inputs, -1), axis=1)

            logger.debug(f"ci.shape = {c_i.shape}")

            return c_i, [c_i]

        # we don't maintain states between steps when computing attention
        # attention is stateless, so we're passing a fake state for RNN step function
        fake_state_c = K.sum(encoder_out_seq, axis=1)
        fake_state_e = K.sum(encoder_out_seq, axis=2)  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e], constants=[encoder_out_seq]
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c], constants=[encoder_out_seq]
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]







def tryloadmodel(model_filename, model_weights_filename):
    with open(model_filename, "r") as json_file:
        model_json = json_file.read()

    # Create a new model from the JSON content
    new_model = tf.keras.models.model_from_json(model_json, custom_objects={"AttentionLayer": AttentionLayer})

    # Load the model weights from the H5 file
    new_model.load_weights(model_weights_filename)
    return new_model


def load_model(model_filename, model_weights_filename):
    f = open(model_filename, 'r', encoding='utf8')
    model = model_from_json(f.read())
    model.load_weights(model_weights_filename)

    return model


def text_cleaner(text):
    contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                           "could've": "could have", "couldn't": "could not",

                           "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                           "hasn't": "has not", "haven't": "have not",

                           "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                           "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                           "I'm": "I am", "I've": "I have", "i'd": "i would",

                           "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                           "i've": "i have", "isn't": "is not", "it'd": "it would",

                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                           "let's": "let us", "ma'am": "madam",

                           "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                           "mightn't've": "might not have", "must've": "must have",

                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                           "needn't've": "need not have", "o'clock": "of the clock",

                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                           "sha'n't": "shall not", "shan't've": "shall not have",

                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                           "she'll've": "she will have", "she's": "she is",

                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                           "so've": "so have", "so's": "so as",

                           "this's": "this is", "that'd": "that would", "that'd've": "that would have",
                           "that's": "that is",
                           "there'd": "there would",

                           "there'd've": "there would have", "there's": "there is", "here's": "here is",
                           "they'd": "they would", "they'd've": "they would have",

                           "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                           "they've": "they have", "to've": "to have",

                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                           "we'll've": "we will have", "we're": "we are",

                           "we've": "we have", "weren't": "were not", "what'll": "what will",
                           "what'll've": "what will have", "what're": "what are",

                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                           "where'd": "where did", "where's": "where is",

                           "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                           "who's": "who is",
                           "who've": "who have",

                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                           "won't've": "will not have",

                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                           "y'all": "you all",

                           "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                           "y'all've": "you all have",

                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                           "you'll've": "you will have",

                           "you're": "you are", "you've": "you have"}

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    newString = text.lower()  # הופך את כל הטקסט לאותיות קטנות
    newString = BeautifulSoup(newString, "html.parser").text  # מוריד תגי HTML
    newString = re.sub(r'\([^)]*\)', '', newString)  # מיפוי התכווצות
    newString = re.sub('"', '', newString)  # הסרת S
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])  #
    newString = re.sub(r"'s\b", "", newString)
    newString = re.sub("[^a-zA-Z]", " ", newString)
    tokens = [w for w in newString.split() if not w in stop_words]
    long_words = []
    for i in tokens:
        if len(i) >= 3:  # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip()




def decode_sequence(input_seq,target_word_index,reverse_target_word_index,encoder,decoder,max_len_summary):
    print('hello iam here!')
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']
    print(target_seq)

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

         output_tokens, h, c = decoder.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
         sampled_token_index = np.argmax(output_tokens[0, -1, :])


         sampled_token = reverse_target_word_index[sampled_token_index]



         if(sampled_token!='end'):
             decoded_sentence += ' '+sampled_token
             print(sampled_token)


            # Exit condition: either hit max length or find stop word.
         if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
             stop_condition = True



        # Update the target sequence (of length 1).
         target_seq = np.zeros((1,1))
         target_seq[0, 0] = sampled_token_index

        # Update internal states
         e_h, e_c = h, c

    return decoded_sentence






# str1, encoder, decoder
def main_t(str1, encoder, decoder):

 #הגדרת פרמטרים
    max_len_text = 80
    max_len_summary = 10
#הצהרה של שימוש בCUSTOM ֹOBJECTS
    tf.keras.utils.get_custom_objects()['AttentionLayer'] = AttentionLayer

 #פתיחה של הTOKENS
    with open('moduls/old_models/x_tokenizer2.pickle', 'rb') as handle:
        loaded_x_tokenizer = pickle.load(handle)

    with open('moduls/old_models/y_tokenizer2.pickle', 'rb') as handle:
        loaded_y_tokenizer = pickle.load(handle)



    reverse_target_word_index = loaded_y_tokenizer.index_word
    # reverse_source_word_index = loaded_x_tokenizer.index_word
    target_word_index = loaded_y_tokenizer.word_index
   #ניקוי הטקסט שהתקבל על ידי הפונקצייה
    clein_str = text_cleaner(str(str1))


    new_text_sequence = loaded_x_tokenizer.texts_to_sequences([clein_str])
    new_text_padded = pad_sequences(new_text_sequence, maxlen=max_len_text, padding='post')

    return ('p:', decode_sequence(new_text_padded[0].reshape(1, max_len_text),target_word_index,reverse_target_word_index,encoder,decoder,max_len_summary))





