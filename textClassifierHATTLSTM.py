#! encoding=utf-8
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
from nltk.translate.bleu_score import sentence_bleu
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
MAX_CODE_TOKENS = 50000#code fragments中处理的频率高的最大单词数
MAX_ALL_WORDS = 100000
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

input_file = 'comment_datasets/train.json'
with open (input_file,'r') as f:
    print ('load data...')
    unicode_data = json.load(f) # read files
    #str_data = json.dumps(unicode_data) # convert into str
    #all_methods = yaml.safe_load(str_data) # safely load (remove 'u')
    #for method in all_methods:
    for method in unicode_data:
        m_comment = method['comment']
        m_code = method['code']
        if len(m_comment) == 0 or len(m_code) == 0:
            continue
        # preprocess
        comment = BeautifulSoup(m_comment)
        comment = clean_str(comment.get_text().encode('ascii','ignore'))
        comment = " commentstart " + comment + " commentend "
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
all_texts = comments + code_texts
print('Total %s comment and code' % len(all_texts))
tokenizer2 = Tokenizer(num_words=MAX_ALL_WORDS)
tokenizer2.fit_on_texts(all_texts)
print('keys: ', tokenizer2.word_index.keys)
print('commentstart: ', tokenizer2.word_index['commentstart'])
print('commentend: ', tokenizer2.word_index['commentend'])
print('Total %s unique tokens in comment and code.' % (len(tokenizer2.word_index)+1))

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
print('decoder_input[0]: ' , decoder_input[0])
print('decoder_target[0]: ' , decoder_target[0])
decoder_input_data = decoder_input[:-nb_validation_samples]
decoder_input_data2 = decoder_input[-nb_validation_samples:]

# decoder_target_data(int sequence)
decoder_target = np.expand_dims(decoder_target,-1)
decoder_target_data = decoder_target[:-nb_validation_samples]
decoder_target_data2 = decoder_target[-nb_validation_samples:]

embedding_layer = Embedding(num_encoder_tokens,
                            EMBEDDING_DIM,
                            input_length=MAX_SENT_LENGTH,
                            mask_zero=True,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent,forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(100,return_state=True))(review_encoder)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

encoder_states = [state_h, state_c]

# add decoder here
decoder_inputs = Input(shape=(None,))
embedding_layer2 = Embedding(num_decoder_tokens, EMBEDDING_DIM)
x = embedding_layer2(decoder_inputs)
decoder_lstm = LSTM(200, return_sequences=True, return_state=True)
x, _, _  = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(x)

model = Model([review_input, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
print("model fitting - Hierachical LSTM")
#model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=50, epochs=20, validation_split=0.2)
model.fit([encoder_input_data, decoder_input_data], decoder_target_data, validation_data=([encoder_input_data2, decoder_input_data2], decoder_target_data2),batch_size=50, epochs=2, validation_split=0.2)

# Save model
model.save('lstm.h5')

# inference
encoder_model = Model(review_input, encoder_states)

decoder_state_input_h = Input(shape=(200,))
decoder_state_input_c = Input(shape=(200,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

y = embedding_layer2(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(y, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model( [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

index2token = {}
index2token[0] = 'UNK'
for token in tokenizer2.word_index:
    index2token[tokenizer2.word_index[token]] = token


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,MAX_COM_WORDS))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = tokenizer2.word_index['commentstart']
    # Sampling loop for a batch of sequences # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_tokens = []
    k = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        decoded_tokens.append(index2token[sampled_token_index])
        k += 1
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_token_index == tokenizer2.word_index['commentend'] or k >= MAX_COM_WORDS):
            stop_condition = True

        # Add the sampled token to the sequence
        target_seq = np.zeros((1,MAX_COM_WORDS))
        target_seq[0,0] = sampled_token_index
        # Update states
        states_value = [h, c]
    return decoded_tokens

score = 0.0
for i, input_seq in enumerate(encoder_input_data2):
    input_now = np.zeros((1,MAX_SENTS, MAX_SENT_LENGTH))
    input_now[0] = input_seq
    print('Input: ')
    for _, sents in enumerate(input_seq):
        if sents[0] == 0:
            print ' '
            break
        for _, token in enumerate(sents):
            if token == 0:
                break
            print index2token[token], ' ',
    decoded_tokens = decode_sequence(input_now)
    print('Output: ', decoded_tokens)
    # validate
    expected_output = decoder_target_data2[i,:,-1]
    expected_tokens = []
    for t in expected_output:
        expected_tokens.append(index2token[t])
        if t == 'commentend':
           break
    print('Expected Output:', expected_tokens)
    score += sentence_bleu([expected_tokens], decoded_tokens)
average_bleu = score / len(encoder_input_data2)
print('average_bleu: ', average_bleu)

'''

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = AttLayer()(l_dense)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttLayer()(l_dense_sent)
# add decoder here
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical attention network")
model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=10, batch_size=50)
'''
