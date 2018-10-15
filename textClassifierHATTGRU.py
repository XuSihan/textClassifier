#! encoding=utf-8
print 'running textclassifierHATTGRU.py'
import numpy as np
import pandas as pd
import cPickle
import yaml
from collections import defaultdict
import re
import json
from bs4 import BeautifulSoup

import sys
import os

#os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer, text_to_word_sequence, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Concatenate
from keras.models import Model
from keras.utils import to_categorical
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
import tensorflow as tf
# set run environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)

MAX_SENT_LENGTH = 100#每句最多单词数
MAX_SENTS = 15#每个code fragments最多句数
MAX_CODE_TOKENS = 20000#code fragments中处理的频率高的最大单词数
MAX_ALL_WORDS = 30000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
MAX_COM_WORDS = 200#comment中最多的单词数
def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

from nltk import tokenize

codes = []
comments = []
code_texts = []

input_file = 'comment_datasets/test.json'
with open (input_file,'r') as f:
    print ('load data...')
    unicode_data = json.load(f) # read files
    str_data = json.dumps(unicode_data) # convert into str
    all_methods = yaml.safe_load(str_data) # safely load (remove 'u')
    for method in all_methods:
        m_comment = method['comment']
        m_code = method['code']
        if len(m_comment) == 0 or len(m_code) == 0:
            continue
        # preprocess
        comment = BeautifulSoup(m_comment)
        comment = clean_str(comment.get_text().encode('ascii','ignore'))
        comment = "<S> " + comment + " <E>"
        comments.append(comment)

        code = ''
        sents = []
        for sentence in m_code:
            sentence = BeautifulSoup(sentence)
            sentence = clean_str(sentence.get_text().encode('ascii','ignore'))
            sents.append(sentence)
            code = code + sentence + ' '
        code_texts.append(code)
        codes.append(sents)
print('Total %s code fragments' % len(code_texts))
# input to numbers
tokenizer = Tokenizer(num_words=MAX_CODE_TOKENS)
tokenizer.fit_on_texts(code_texts)
# 没出现的就是0
data = np.zeros((len(code_texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(codes):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS: #每个code fragments的前MAX_SENTS句话
            #分词
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_CODE_TOKENS: #每句话的前MAX_SENT_LENGTH个单词
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1
print('Total %s unique tokens in code.' % (len(tokenizer.word_index)+1))
num_encoder_tokens = len(tokenizer.word_index) + 1 # 默认为0

# target to numbers
all_texts = code_texts + comments
print('Total %s comment and code' % len(all_texts))
tokenizer2 = Tokenizer(num_words=MAX_ALL_WORDS)
tokenizer2.fit_on_texts(all_texts)
print('Total %s unique tokens in comment and code.' % len(tokenizer2.word_index))

com_data = np.zeros((len(comments), MAX_COM_WORDS), dtype='int32')
for i, com in enumerate(comments):
    wordTokens = text_to_word_sequence(com)
    j=0
    for _, word in enumerate(wordTokens):
        if j<MAX_COM_WORDS and tokenizer2.word_index[word]<MAX_ALL_WORDS:
            com_data[i,j] = tokenizer2.word_index[word]
            j=j+1
num_decoder_tokens = len(tokenizer2.word_index) + 1 #默认为0

print('Shape of code tensor:', data.shape)
print('Shape of comment tensor:', com_data.shape)
#one_hot
#com_one_hot = to_categorical(com_data)

#shuffle
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
com_data = com_data[indices]

# split
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

# encoder_input_data(int sequence)
encoder_input_data = data[:-nb_validation_samples]
encoder_input_data2 = data[-nb_validation_samples:]

# decoder_inputs为decoder_target的错一位，从而teacher force
# decoder_input_data(int sequence, decoder_target_data + 1 step)
decoder_input = np.zeros((len(com_data),MAX_COM_WORDS), dtype='int32')
decoder_target = np.zeros((len(com_data),MAX_COM_WORDS), dtype='int32')
for i,com in enumerate(com_data):
    for j in range(0,len(com)):
        decoder_input[i][j] = com_data[i][j]
        if j > 0:
            decoder_target[i,j-1]=com_data[i,j]
print 'decoder_input[0]: ' , decoder_input[0]
print 'com_data[0]: ' , com_data[0]
decoder_input_data = decoder_input[:-nb_validation_samples]
decoder_input_data2 = decoder_input[-nb_validation_samples:]

# decoder_target_data(int sequence)
decoder_target = np.expand_dims(decoder_target,-1)
decoder_target_data = decoder_target[:-nb_validation_samples]
decoder_target_data2 = decoder_target[-nb_validation_samples:]


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


embedding_layer = Embedding(num_encoder_tokens,
                            EMBEDDING_DIM,
                            input_length=MAX_SENT_LENGTH,
                            mask_zero=True,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
att1 = AttLayer(100)
l_att = att1(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent,forward_h, backward_h = Bidirectional(GRU(100,return_state=True, return_sequences=True))(review_encoder)
#state_h = Concatenate()([forward_h, backward_h])
att2 = AttLayer(100)
l_att_sent = att2(l_lstm_sent)
encoder_states = l_att_sent

# add decoder here
decoder_inputs = Input(shape=(None,))
embedding_layer2 = Embedding(num_decoder_tokens, EMBEDDING_DIM)
x = embedding_layer2(decoder_inputs)
decoder_lstm = GRU(100, return_sequences=True)
x = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(x)

model = Model([review_input, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
print("model fitting - Hierachical LSTM")
#model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=50, epochs=20, validation_split=0.2)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, validation_data=([encoder_input_data2, decoder_input_data2], decoder_target_data2),batch_size=50, epochs=10, validation_split=0.2)

# inference
encoder_model = Model(review_input, encoder_states)

decoder_state_h = Input(shape=(100,))
decoder_states = decoder_state_h

y = embedding_layer2(decoder_inputs)
y = decoder_lstm(y, initial_state=decoder_states)
decoder_outputs = decoder_dense(y)
decoder_model = Model( [decoder_inputs] + decoder_states, decoder_outputs)

index2token = {}
for token in tokenizer2.word_index:
    index2token[tokenizer2.word_index[token]] = token


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,MAX_COM_WORDS))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer2.word_index['<S>']
    # Sampling loop for a batch of sequences # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    k = 0
    while not stop_condition:
        output_tokens = decoder_model.predict([target_seq] + states_value)
        print('output_tokens',output_tokens)
        print('output_tokens.shape',output_tokens.shape)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        decoded_sentence += index2token[sampled_token_index] + ' '
        k += 1
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token_index == tokenizer2.word_index['<E>'] or k >= MAX_COM_WORDS):
            stop_condition = True

        # Add the sampled token to the sequence
        target_seq[0,k] = sampled_token_index
    return decoded_sentence

for i, input_seq in enumerate(encoder_input_data2):
    input_now = np.zeros((1,MAX_SENTS, MAX_SENT_LENGTH))
    input_now[0] = input_seq
    print 'Input: ' + input_now[0]
    decoded_sentence = decode_sequence(input_now)
    print 'Output: ' + decoded_sentence
