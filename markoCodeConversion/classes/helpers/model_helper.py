#  general imports
import json
import pickle
from sys import platform as _platform

# Keras
from keras.models import Sequential, model_from_json
from keras.layers.recurrent import GRU
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop

# custom classes
from config import config


class ModelHelper:

    def _cloneWeights(self, sourceModel, destinModel):
        # set weights from the original model to the new destination model layer by layer
        for sourceL, destinL in zip(sourceModel.layers, destinModel.layers):
            destinL.set_weights(sourceL.get_weights())
        return destinModel

    def saveModel(self, model, name, path=None):

        if path is None:
            path = config.Config().root_dir + config.settings.models.path

        # DEBUG:
        print 'saving model ', name

        # model to JSON
        jsonModel = model.to_json()

        fname = path + '/' + name + '.json'
        with open(fname, 'w+') as f:
            json.dump(jsonModel, f)

        # weights to txt
        w = []
        for l in model.layers:
            w.append(l.get_weights())

        fname = path + '/' + name + '_weights.txt'
        with open(fname, "wb+") as f:
            pickle.dump(w, f)

    def loadModel(self, name, path=None):

        # DEBUG:
        # print 'loading model...'

        if path is None:
            path = config.Config().root_dir + config.settings.models.path
        elif not path.startswith(config.Config().root_dir + config.settings.models.path):
            path = config.Config().root_dir + config.settings.models.path + path

        # model from JSON
        fname = path + '/' + name + '.json'

        with open(fname) as f:
            json_model = json.load(f)

        # rebuild the model
        model = model_from_json(json_model)

        # model weights from txt
        fname = path + '/' + name + '_weights.txt'
        with open(fname, 'rb') as f:
            w = pickle.load(f)

        # set the weights
        for l, layer_w in zip(model.layers, w):
            l.set_weights(layer_w)

        if model is not None:
            print 'Model ', name, ' loaded'
            return model
        else:
            print 'Loading model ', name, ' failed!'
        return False

    def makeModelFromParam(self, inpSize, nLabels, architecture, DOValue, lr, init, activ, momentum=0, stateful=False, sliceSize=10):
        '''Creates model based on the architecture formatted as generated from build_architectures()
        other parameters:
            DOvalue - value of dropout between the layers
        '''

        print 'Building Keras model...'

        model = Sequential()
        if stateful:
            model.add(GRU(architecture[0],  batch_input_shape=(1, sliceSize, inpSize), return_sequences=True,
                          init=init,
                          activation=activ))
        else:
            model.add(GRU(architecture[0], input_dim=inpSize, return_sequences=True, init=init, activation=activ))
        model.add(Dropout(DOValue))

        for i in range(1, len(architecture)):
            model.add(GRU(architecture[i], return_sequences=True, init=init, activation=activ))
            model.add(Dropout(DOValue))

        model.add(TimeDistributed(Dense(nLabels, activation='softmax')))

        model = self.compileModel(model, lr=lr, momentum=momentum)

        # print model.summary()

        return model

    def compileModel(self, model, momentum=0.1, lr=0.1):

        # setting the optimiser parameters
        optim = RMSprop(lr=lr)

        # model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer=optim,
                      metrics=['accuracy'])

        return model

    def makeGSModelPad(self, inpSize, nLabels, nHiddenLayers, layerSize, DOValues):
        '''Creates model comprised of n hiddenLayers, with sizes layernSizes
        other parameters:
            DOvalue - value of dropout between the layers
            batchSize - batch size
        '''

        model = Sequential()

        model.add(GRU(layerSize, input_dim=inpSize, return_sequences=True))
        model.add(Dropout(DOValues))

        for i in range(nHiddenLayers):
            model.add(GRU(layerSize, return_sequences=True))
            model.add(Dropout(DOValues))

        model.add(TimeDistributed(Dense(nLabels, activation='softmax')))

        model.summary()
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        return model

    def plotModel(self, model):
        from keras.utils.visualize_util import plot
        plot(model, to_file='model.png', show_shapes=True)
