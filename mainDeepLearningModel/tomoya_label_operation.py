# -*-# coding: utf-8 -*-

"""
This modele provides eneric loading, showing and saving of input data.

This includes some uncompleted function such as ...
"""

import numpy as np
from keras.utils import to_categorical

class Labeling_data:

    def __init__(label, motionNames):

        label.motionnames = motionNames
        label.new_word_id = 0
        label.words = []
        label.dictionary = {}

    def labeling(label):

        label.text = label.motionnames.lower().replace("\n", " ")
        for label.word in label.text.split():
            if label.word not in label.dictionary:
                label.dictionary[label.word] = label.new_word_id
                label.new_word_id += 1
            label.words.append(label.dictionary[label.word])
        label.vocabulary_size = label.new_word_id
        label.label = to_categorical(label.words)

        # label.dictionary_inv = {label.dictionary[k] : for k in label.dictionary}

new_word_id = 0
words = []
dictionary = {}
print('start')

train = 'We are the member of gv lab are of'
text = train.lower().replace("\n", " ")
for word in text.split():
    if word not in dictionary:
        dictionary[word] = new_word_id
        new_word_id += 1
    words.append(dictionary[word])
vocabulary_size = new_word_id
label = to_categorical(words)


dictionary_inv = {dictionary[k] : k for k in dictionary}
for i in range(0, 7):
    print(label[i], dictionary_inv.get(i))

a = np.argmax([ 0.,  0.,  0.,  0.,  0.8,  0.,  0.])
b = dictionary_inv.get(a)
print(b)

print('finish')

