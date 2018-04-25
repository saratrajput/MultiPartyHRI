# general imports
import numpy as np
from numpy import linalg

# custom modules
from classifier import Classifier
from classes.helpers import classifier_helper


class Touch(Classifier):
    def __init__(self, name):
        super(Touch, self).__init__(name)
        cs = classifier_helper.ClassifierSettings(name)
        self.threshold = cs.threshold

    def update_data(self, raw_data):
        data = self._take_features(raw_data)
        # take the norm of the hand vectors
        proc_data = np.asarray([
            linalg.norm(data[:3]),
            linalg.norm(data[3:])
                                ])
        # DEBUG
        # print proc_data

        super(Touch, self).update_data(proc_data)

    def update_decision(self, decision=False):

        # DEBUG
        # print [self.buffer[:, 0].sum()/self.buffer_length,
        #        self.buffer[:, 1].sum()/self.buffer_length]

        if self.buffer[:, 0].sum()/self.buffer.shape[0] < self.threshold or \
            self.buffer[:, 1].sum()/self.buffer.shape[0] < self.threshold:
            decision = self.label_list[1]
        else:
            decision = self.label_list[0]
        super(Touch, self).update_decision(decision)
