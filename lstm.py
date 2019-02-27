#!/usr/bin/env python3

import os
import sys
import re
import codecs
import numpy
import time
import random
import pandas
from optparse import OptionParser
from collections import defaultdict
import CONLLParser
import CustomLabelEncoder
import utils
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, Reshape, Concatenate
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import keras.backend
import configparser

def getTokenIdDict(conlltokens):

    sidtid2ct = defaultdict(CONLLParser.CustomCONLLToken)
    for ct in conlltokens:
        sidtid2ct[(ct.sid, ct.stid)] = ct
    return sidtid2ct


def getFeatureMatrix(conlltokens):

    sidtid2ct = getTokenIdDict(conlltokens)
    matrix = []
    #headers = ['word', 'func', 'relative_position_in_sentence', 'orig_lang', 'class_label'] # mem heavy version
    memheavy = False # to leave some features with a lot of different values out 
    for cti, ct in enumerate(conlltokens):
        if cti % 1000 == 0:
            sys.stderr.write('INFO: Processed %i of %i...\n' % (cti, len(conlltokens)))
        row = []
        word = ct.token
        dist2par = 0
        if type(ct.head) == int:
            dist2par = ct.stid - ct.head # could experiment with absolute distance here
        parent_func = None
        if (ct.sid, ct.head) in sidtid2ct:
            parent_func = sidtid2ct[(ct.sid, ct.head)].deprel
        func = ct.deprel
        parent_pos_coarse = None
        if (ct.sid, ct.head) in sidtid2ct:
            parent_pos_coarse = sidtid2ct[(ct.sid, ct.head)].pos_coarse
        poscoarse = ct.pos_coarse
        word_upper = False
        if ct.token[0].isupper():
            word_upper = True
        relative_position = 0
        if not len(ct.fullSentence) == 0:
            relative_position = utils.categorizeFractal(ct.stid / len(ct.fullSentence), 5)
        

        row.append(word)
        if not memheavy:
            row.append(dist2par)
            row.append(parent_func)
        row.append(func)
        if not memheavy:
            row.append(parent_pos_coarse)
        if not memheavy:
            row.append(poscoarse)
        if not memheavy:
            row.append(word_upper)
        row.append(relative_position)
        row.append(ct.orig_lang)

        label = ct.segmentStarter
        row.append(label)

        
        matrix.append(row)
        
    return matrix


def createEmbeddingMatrix(matrix, headers, le, embd):

    df = pandas.DataFrame(matrix, columns=headers)
    Y = df.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])

    f2ohvpos = defaultdict(lambda : defaultdict(int))
    f2 = defaultdict(set)
    f2ohvlen = defaultdict()
    rowsize = 0
    for row in matrix:
        rowsize = len(row)
        for pos, val in enumerate(row):
            f2[pos].add(val)
    for i in f2:
        f2ohvlen[i] = len(f2[i])
        for c, i2 in enumerate(f2[i]):
            f2ohvpos[i][i2] = c

    dim = 300
    
    X1 = []
    X2 = []
    input_dim = 0
    for row in matrix:
        token = row[0]
        orig_lang = row[-2]
        
        nrow = []
        embrow = []
        if embd:
            if token in embd:
                for item in embd[token]:
                    embrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    embrow.append(item)
            X1.append(embrow)
        else:
            X1.append([le.w2i[row[0]]])
            
        for index, val in enumerate(row[1:-1]):
            nextrows = [0] * f2ohvlen[index+1]
            nextrows[f2ohvpos[index+1][val]] = 1
            nrow += nextrows
        input_dim = len(nrow)
        X2.append(nrow)

    X1 = numpy.array(X1)
    X2 = numpy.array(X2)
        
    return X1, X2, Y, f2ohvlen, f2ohvpos, input_dim


def getFeaturesForTestData(matrix, headers, le, f2ohvlen, f2ohvpos, embd):

    df = pandas.DataFrame(matrix, columns=headers)
    Y = df.class_label
    labels = list(set(Y))
    Y = numpy.array([labels.index(x) for x in Y])

    dim = 300
    X1 = []
    X2 = []
    for row in matrix:
        token = row[0]
        orig_lang = row[-2]

        nrow = []
        embrow = []
        if embd:
            if token in embd:
                for item in embd[token]:
                    embrow.append(item)
            else:
                for item in numpy.ndarray.flatten(numpy.random.random((1, dim))):
                    embrow.append(item)
            X1.append(embrow)
        else:
            if row[0] in le.w2i:
                X1.append([le.w2i[row[0]]])
            else:
                X1.append([0]) # unknown words get a 0
            
        for index, val in enumerate(row[1:-1]):
            nextrows = [0] * f2ohvlen[index+1]
            if val in f2ohvpos[index+1]:
                nextrows[f2ohvpos[index+1][val]] = 1
            else: # feature not seen during training
                nextrows[nextrows[random.randint(0, len(nextrows)-1)]] = 1
            nrow += nextrows
        X2.append(nrow)

    X1 = numpy.array(X1)
    X2 = numpy.array(X2)

    return X1, X2, Y


def prepareDataInBatchSize(X, bsize, step_size):

    n = []
    for i in range(0, len(X), step_size):
        if len(X[i:i+bsize]) == bsize:
            n.append(numpy.array(X[i:i+bsize]))
    n = numpy.stack(n)

    return n


def extembs(train, test, headers, epochs, embd):

    le = CustomLabelEncoder.CustomLabelEncoder()
    le.vocabIndex(train)

    X1, X2, Y, f2ohvlen, f2ohvpos, dim2 = createEmbeddingMatrix(train, headers, le, embd)
    dim = 300 # static, depends on pretrained embs
    X1, X2 = padInputs(X1, X2, dim, dim2)
    X1_test, X2_test, Y_test = getFeaturesForTestData(test, headers, le, f2ohvlen, f2ohvpos, embd)
    X1_test, X2_test = padInputs(X1_test, X2_test, dim, dim2)
    dim_1 = numpy.shape(X1)[1]
    dim_2 = numpy.shape(X2)[1]
    nr_test_items = len(X1_test)
    binary_gold = Y_test
    bsize = 20
    step_size = 5
    X1 = prepareDataInBatchSize(X1, bsize, step_size)
    X2 = prepareDataInBatchSize(X2, bsize, step_size)
    Y = prepareDataInBatchSize(Y, bsize, step_size)
    Y = numpy.expand_dims(Y, -1)

    model = LSTM_extembs(max(dim, dim2), bsize)
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit({'i1': X2, 'i2': X1}, {'output_layer' :Y }, batch_size=128,epochs=epochs,validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.000001)])
    X1_test = prepareDataInBatchSize(X1_test, bsize, step_size)
    X2_test = prepareDataInBatchSize(X2_test, bsize, step_size)
    Y_test = prepareDataInBatchSize(Y_test, bsize, step_size)
    Y_test = numpy.expand_dims(Y_test, -1)
    results = model.predict([X2_test, X1_test])

    binary_pred = interpret_keras_results(results, step_size, bsize, nr_test_items)

    return binary_pred
    
def loadExternalEmbeddings(lang):

    starttime = time.time()
    sys.stderr.write('INFO: Loading external embeddings...\n')
    ed = defaultdict()
    config = configparser.ConfigParser()
    config.read('config.ini')
    ud = {'deu':config['embeddings']['deu'],
          'nld':config['embeddings']['nld'],
          'por':config['embeddings']['por'],
          'spa':config['embeddings']['spa'],
          'eng':config['embeddings']['eng'],
          'rus':config['embeddings']['rus'],
          'fra':config['embeddings']['fra'],
          'zho':config['embeddings']['zho'],
          'eus':config['embeddings']['eus']}
    with open(ud[lang], 'r') as f:
        for line in f.readlines():
            line = line.strip()
            values = line.split()
            ed[values[0]] = numpy.array([float(x) for x in values[1:]])

    endtime = time.time()
    sys.stderr.write('INFO: Done loading embeddings. Took %s seconds.\n' % (str(endtime - starttime)))
    return ed


def intembs(train, test, headers, epochs):

    le = CustomLabelEncoder.CustomLabelEncoder()
    le.vocabIndex(train)
    X1, X2, Y, f2ohvlen, f2ohvpos, dim = createEmbeddingMatrix(train, headers, le, False)
    X1_test, X2_test, Y_test = getFeaturesForTestData(test, headers, le, f2ohvlen, f2ohvpos, False)

    nr_test_items = len(X1_test)
    binary_gold = Y_test

    # NOTE: X1 are the words, X2 are the one-hot encoded features

    voc_size = (X1.max()+1).astype('int64')
    bsize = 20
    step_size = 5
    epochs = 10

    X1 = prepareDataInBatchSize(X1, bsize, step_size)
    X2 = prepareDataInBatchSize(X2, bsize, step_size)
    Y = prepareDataInBatchSize(Y, bsize, step_size)
    Y = numpy.expand_dims(Y, -1)

    inputs1 = Input(name='i1',shape=[bsize,dim])
    inputs2 = Input(name='i2',shape=[bsize,1])

    edim = 20
    emblayer = Embedding(voc_size,edim)(inputs2)
    emblayer = Reshape((bsize,edim))(emblayer)

    model = LSTM_intembs(dim, voc_size, bsize)

    model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    
    model.fit({'i1': X2, 'i2': X1}, {'output_layer' :Y }, batch_size=256,epochs=epochs,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.00001)])
    #model.fit({'i1': X2, 'i2': X1}, {'output_layer' :Y }, batch_size=256,epochs=epochs,validation_split=0.1)

    X1_test = prepareDataInBatchSize(X1_test, bsize, step_size)
    X2_test = prepareDataInBatchSize(X2_test, bsize, step_size)
    Y_test = prepareDataInBatchSize(Y_test, bsize, step_size)
    Y_test = numpy.expand_dims(Y_test, -1)
    results = model.predict([X2_test, X1_test])

    binary_pred = interpret_keras_results(results, step_size, bsize, nr_test_items)

    return binary_pred

def LSTM_extembs(dim, bsize):

    inputs1 = Input(name='i1',shape=[bsize,dim])
    inputs2 = Input(name='i2',shape=[bsize,dim])
    layer = Concatenate(axis=-1)([inputs1, inputs2])
    layer = LSTM(2048,return_sequences=True)(layer)
    layer = Dense(1024,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(256,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(128,name='FC3')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid', name='output_layer')(layer)
    
    model = Model(inputs=[inputs1, inputs2],outputs=layer)
    return model

def LSTM_intembs(dim, voc_size, bsize):

    inputs1 = Input(name='i1',shape=[bsize,dim])
    inputs2 = Input(name='i2',shape=[bsize,1])
    emblayer = Embedding(voc_size,50)(inputs2)
    emblayer = Reshape((bsize,50))(emblayer)
    layer = Concatenate()([inputs1, emblayer])
    layer = LSTM(2048,return_sequences=True)(layer)
    layer = Dense(1024,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2)(layer)
    layer = Dense(256,name='FC2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(128,name='FC3')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1)(layer)
    layer = Activation('sigmoid', name='output_layer')(layer)

    model = Model(inputs=[inputs1, inputs2],outputs=layer)
    return model



def interpret_keras_results(results, step_size, bsize, nr_test_items):

    # from 20 onward, each has 5 figures, average these (nr 15 has 4, 10 has 3, 5 has 2, 0-4 have 1, and len(Y)-15 have 4 again, etc.)
    pos2vals = defaultdict(list)
    offset = 0
    for i in results:
        for p, j in enumerate(i):
            pos2vals[p+offset].append(j)
        offset += step_size
    float_vals = []
    for pos in pos2vals:
        float_vals.append(sum(pos2vals[pos]) / len(pos2vals[pos]))
    binary_results = [0 if x  < 0.5 else 1 for x in float_vals]
    for jk in range(len(binary_results), nr_test_items):
        binary_results.append(0)

    return binary_results

def padInputs(X1, X2, dim, dim2):

    target = max(dim, dim2)
    
    diff = target - min(dim, dim2)
    X1_padded = []
    X2_padded = []
    if numpy.shape(X1)[1] == target:
        for row in X2:
            row = numpy.concatenate([row,numpy.zeros(diff, dtype=float)])
            X2_padded.append(row)
        X1_padded = X1
        X2_padded = numpy.array(X2_padded)
    elif numpy.shape(X2)[1] == target:
        for row in X1:
            row = numpy.concatenate([row,numpy.zeros(diff, dtype=float)])
            X1_padded.append(row)
        X2_padded = X2
        X1_padded = numpy.array(X1_padded)

    return X1_padded, X2_padded


def printConllOut(conllin, labels, i, suffix):

    outname = conllin + '.%s_%s.predicted' % (suffix, str(i))
    conllout = codecs.open(outname, 'w')
    c = 0
    for line in codecs.open(conllin).readlines():
        if re.search('\t', line) and not line.startswith('#'):
            lastelem = line.strip().split('\t')[-1]
            if labels[c] == 1:
                if re.search('SpaceAfter', lastelem):
                    conllout.write('\t'.join(line.strip().split('\t')[:-1]) + '\tBeginSeg=Yes|%s\n' % lastelem.split('|')[-1])
                else:
                    conllout.write('\t'.join(line.strip().split('\t')[:-1]) + '\tBeginSeg=Yes\n')
            else:
                if re.search('SpaceAfter', lastelem) and not re.search('BeginSeg', lastelem):
                    conllout.write(line)
                else:
                    conllout.write('\t'.join(line.strip().split('\t')[:-1]) + '\t_\n')
            c += 1
        else:
            conllout.write(line)
    conllout.close()
    sys.stderr.write('INFO: output written to %s\n' % outname)


if __name__ == '__main__':

    parser = OptionParser('Usage: %prog -options')
    parser.add_option('--train', dest='trainconll', help='train file in CONLL format')
    parser.add_option('--dev', dest='devconll', help='dev file in CONLL format')
    parser.add_option('--test', dest='testconll', help='test file in CONLL format')
    parser.add_option('--mode', '-m', dest='mode', help='specify either "int" for corpus-internal embeddings or "ext" for pre-trained embeddings (location specified in config.ini)')
    
    options, args = parser.parse_args()

    if not options.trainconll or not options.devconll or not options.testconll:
        parser.print_help(sys.stderr)
        sys.exit(1)
    if not options.mode or not options.mode in ['int', 'ext']:
        sys.stderr.write('Please specify a mode, either "int" for corpus-internal embeddings or "ext" for pre-trained embeddings (location specified in config.ini)\n')
        sys.exit(1)

    language = os.path.basename(options.trainconll)[:3]
    
    traintokens = CONLLParser.parse(options.trainconll, language)
    devtokens = CONLLParser.parse(options.devconll, language)
    testtokens = CONLLParser.parse(options.testconll, language)

    traintokens.extend(devtokens) # throw train and dev together

    trainmatrix = getFeatureMatrix(traintokens)
    testmatrix = getFeatureMatrix(testtokens)

    headers = ['word', 'dist2par', 'parent_func', 'func',
               'parent_pos_coarse', 'pos_coarse',
               'word_upper', 'relative_position_in_sentence',
               'orig_lang',
               'class_label']

    epochs = 10
    """
    for i in range(10):
        labels = intembs(trainmatrix, testmatrix, headers, epochs)
        printConllOut(options.testconll, labels, i, 'lstm_intembs')
    """
    """
    embd = loadExternalEmbeddings(language)
    #for i in range(10):
    i = 0
    labels = extembs(trainmatrix, testmatrix, headers, epochs, embd)
    printConllOut(options.testconll, labels, i, 'lstm_extembs')
    """
    if options.mode == 'int':
        labels = intembs(trainmatrix, testmatrix, headers, epochs)
        printConllOut(options.testconll, labels, 0, 'lstm_intembs')

    elif options.mode == 'ext':
        labels = extembs(trainmatrix, testmatrix, headers, epochs, embd)
        printConllOut(options.testconll, labels, 0, 'lstm_extembs')
