import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
from keras.utils import to_categorical
from itertools import chain
import matplotlib.pyplot as plt
import csv
import seaborn as sn
from sklearn.metrics import confusion_matrix


def labels_2_one_hots(labels=''):
    words = []
    new_word_id = 0
    dictionary = {}
    labels = labels[:, np.newaxis]
    # sort by alphabetical order
    labelsSorted = np.sort(labels, axis=0)

    for word in labelsSorted:
        if word[0] not in dictionary:
            dictionary[word[0]] = new_word_id
            new_word_id += 1

    for word in labels:
        words.append(dictionary[word[0]])

    one_hots = to_categorical(words)

    dictionary_inv = {dictionary[k]: k for k in dictionary}

    return dictionary, one_hots, dictionary_inv

def get_train_test_data():
    # Get current working directory of the process
    cwd = os.getcwd()

    # '\\' is Windows style. Doesn't work on Linux
    path = cwd + '\\train\\ASL\\Etienne\\data\\Word'
    files = []

    # Returns a list containing names of the entries in the directory given by path
    for file in os.listdir(path): 
        if file.endswith("_100.csv"): # Returns True if string ends with specified suffix
            files.append(file)
    files = np.array(files)
    labelsOutput = []
    InputAll = []

    for i in range(0, int(files.shape[0])):
        dfInput = pd.read_csv(path + '\\' + files[i], sep=',')
        dfInput.dropna(how='any') # Drop missing values
        dfInput.drop(['Unnamed: 0', 'Time'], axis=1, inplace=True)
        print(files[i] + ' open')
        label = str.lower(files[i][:-4])
        InputNames = dfInput.columns

        Input = dfInput.as_matrix() # Convert the frame to its Numpy-array representation
        nCut = int(Input.shape[0] / 100) # Why 100? 
        Input = np.array(np.split(Input, nCut, axis=0))
        Input = Input.transpose([1, 2, 0])


        labelsOutput.append([label for j in range(0, nCut)])

        if InputAll == []:
            InputAll = Input
        else:
            InputAll = np.concatenate([InputAll, Input], axis=2)

    labelsOutput = np.array(list(chain.from_iterable(labelsOutput))) # chain('ABC', 'DEF') --> A B C D E F

    p = np.random.permutation(InputAll.shape[2])

    InputAll = InputAll[:, :, p]
    labelsOutput = labelsOutput[p]

    dictionary, one_hots, dictionary_inv = labels_2_one_hots(labelsOutput)

    return InputAll, dictionary, one_hots, InputNames, dictionary_inv

def standardize_data(data):
    std = []
    mean = []
    dataOut = data.copy()

    std.append(np.std(data, ddof=1))
    mean.append(np.mean(data))
    dataOut = (data - mean) / std

    dataOut[np.isnan(dataOut)] = 0

    return dataOut, mean, std

class RNN:
    def __init__(RNN):

        RNN.hm_epochs = 10000

    def set_Data(RNN, Input, Output, InputNames, LabelsDict, dictionary_inv):

        RNN.InputNames = InputNames
        RNN.OutputNames = sorted(LabelsDict, key = LabelsDict.get, reverse = False)
        RNN.dictionary = LabelsDict
        RNN.dictionary_inv = dictionary_inv

        RNN.n_samples = Input.shape[2]
        RNN.training_size = int(0.8 * RNN.n_samples)
        RNN.testing_size = int(0.2 * RNN.n_samples)

        Input_N, RNN.meanInput, RNN.stdInput = standardize_data(Input)

        RNN.train_x = np.array(Input[:, :, :RNN.training_size]).astype('float32')
        RNN.train_x = np.transpose(RNN.train_x, [2, 0, 1])
        RNN.train_y = np.array(Output[:RNN.training_size])

        RNN.test_x = np.array(Input[:, :, -RNN.testing_size:]).astype('float32')
        RNN.test_x = np.transpose(RNN.test_x, [2, 0, 1])
        RNN.test_y = np.array(Output[-RNN.testing_size:])

        RNN.train_x_N = np.array(Input_N[:, :, :RNN.training_size]).astype('float32')
        RNN.train_x_N = np.transpose(RNN.train_x_N, [2, 0, 1])
        RNN.test_x_N = np.array(Input_N[:, :, -RNN.testing_size:]).astype('float32')
        RNN.test_x_N = np.transpose(RNN.test_x_N, [2, 0, 1])

    def train_neural_network(RNN, InputAll, one_hots):

        errorHistory =[]

        tf.reset_default_graph()

        sess = tf.Session()

        frames = 100
        n_Joint = 162

        CellSize = 10
        OutputSize = 20

        RNN.x = tf.placeholder('float32',
                           [None, frames, n_Joint])  # TensorShape([Dimension(None), Dimension(28), Dimension(28)])
        RNN.y = tf.placeholder('float32', [None, OutputSize])


        layer = {'weights': tf.Variable(tf.random_normal([CellSize, OutputSize])),
                 'biases': tf.Variable(tf.random_normal([OutputSize]))}

        x = tf.transpose(RNN.x, [1, 0, 2])  # TensorShape([Dimension(28), Dimension(None), Dimension(28)])
        x = tf.reshape(x, [-1, n_Joint])
        x = tf.split(x, frames, 0)  # len(x) = 100

        lstm_cell = rnn.BasicRNNCell(CellSize)

        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

        RNN.y_ = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

        learningRate = 0.01
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=RNN.y_, labels=RNN.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

        correct = tf.equal(tf.argmax(RNN.y_, 1), tf.argmax(RNN.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        sess.run(tf.global_variables_initializer())

        for epoch in range(RNN.hm_epochs):

            _, c = sess.run([optimizer, cost], feed_dict={RNN.x:RNN.train_x_N , RNN.y: RNN.train_y})
            errorHistory.append(c)
            if epoch > 100:
                if epoch % 100 == 0:
                    if abs(errorHistory[epoch] - errorHistory[epoch-100]) < 0.01:
                        break
            print('Epoch', epoch, 'completed out of', RNN.hm_epochs, 'loss:', c)

        test_data = {RNN.x: RNN.test_x_N, RNN.y: RNN.test_y}

        RNN.accuracy = sess.run(accuracy, feed_dict = test_data)

        input_data = {RNN.x: RNN.test_x_N}

        RNN.output_rnn = sess.run(RNN.y_, feed_dict=input_data)

        RNN.output_probabilities = tf.nn.softmax(RNN.output_rnn)
        RNN.output_probabilities = sess.run(RNN.output_probabilities)
        RNN.output_probabilities = np.array(RNN.output_probabilities)

        RNN.output_OneHot = (RNN.output_probabilities == RNN.output_probabilities.max(axis = 1, keepdims = True)).astype(int)

        RNN.output_label = []

        for i in range(0, len(RNN.dictionary)):
            RNN.output_label_temp = RNN.dictionary_inv.get(np.argmax(RNN.output_OneHot[i]))
            RNN.output_label = np.append(RNN.output_label, RNN.output_label_temp)
            # RNN.output_label = np.core.defchararray.add(RNN.output_label, RNN.output_label_temp)

        print('Accuracy:', RNN.accuracy)
        plt.figure()
        plt.plot(errorHistory)
        plt.title('Loss_value')
        plt.savefig('Loss' + RNN.accuracy + '.png')

        f = open('loss' + RNN.accuracy + '.csv', 'w')
        writer = csv.writer(f, lineterminator='/n')
        writer.writerow(errorHistory)
        f.close()

    def draw_confusion_matrix(RNN):

        RNN.test_label = []

        for i in range(0, len(RNN.dictionary)):
            RNN.test_label_temp = RNN.dictionary_inv.get(np.argmax(RNN.test_y[i]))
            RNN.test_label = np.append(RNN.test_label, RNN.test_label_temp)

        labels = sorted(list(set(RNN.test_label)))
        cmx_data = confusion_matrix(RNN.test_label, RNN.output_label, labels=labels)
        with np.errstate(divide='ignore', invalid='ignore'):
            cmx_data_N = np.true_divide(cmx_data, cmx_data.astype(np.float).sum(axis=0))
            cmx_data_N = np.nan_to_num(cmx_data_N)

        df_cmx = pd.DataFrame(cmx_data.T, index=labels, columns=labels)
        df_cmx_N = pd.DataFrame(cmx_data_N.T, index=labels, columns=labels)

        np.savetxt('100.txt', cmx_data_N)

        # plt.figure()
        # sn.heatmap(df_cmx, vmax=1, vmin=0, annot=True, robust=True)
        # plt.show()

        plt.figure(figsize=(30,30))
        sn.heatmap(df_cmx_N, vmax=1, vmin=0, annot=True)
        plt.show()
        plt.savefig('cmx.png')






InputAll, dictionary, one_hots, InputNames, dictionary_inv = get_train_test_data()

RNN = RNN()
RNN.set_Data(Input=InputAll, Output=one_hots, InputNames=InputNames, LabelsDict=dictionary, dictionary_inv=dictionary_inv)
print('-----done.')
RNN.train_neural_network(InputAll, one_hots)
RNN.draw_confusion_matrix()
