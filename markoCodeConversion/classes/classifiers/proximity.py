import math
import numpy as np

from classifier import Classifier
from classes.helpers import classifier_helper


class Proximity(Classifier):
    def __init__(self, name):
        super(Proximity, self).__init__(name)
        cs = classifier_helper.ClassifierSettings(name)
        self.threshold = cs.threshold

    def update_data(self, raw_data):
        data = self._take_features(raw_data)
        proc_data = math.sqrt(data[0] ** 2 + data[1] ** 2)
        super(Proximity, self).update_data(np.asarray([proc_data]))

    def update_decision(self, decision=False):
        # DEBUG:
        # print self.buffer.sum() / self.buffer.shape[0]

        if self.buffer.sum() / self.buffer.shape[0] < self.threshold:
            decision = self.label_list[1]
        else:
            decision = self.label_list[0]
        super(Proximity, self).update_decision(decision)
