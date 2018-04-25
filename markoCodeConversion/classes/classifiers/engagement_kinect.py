import numpy as np

from classifier import Classifier
from classes.helpers import classifier_helper


class Engagement(Classifier):
    def __init__(self, name):
        cs = classifier_helper.ClassifierSettings(name)
        self.threshold = cs.threshold

    def update_data(self, raw_data):
        data = self._take_features(raw_data)
        super(Engagement, self).update_data(np.asarray([data]))

    def update_decision(self, decision=False):
        # DEBUG
        # print self.buffer.sum() / self.buffer.shape[0]

        if self.buffer.sum() / self.buffer.shape[0] > self.threshold:
            decision = self.label_list[1]
        else:
            decision = self.label_list[0]
        super(Engagement, self).update_decision(decision)
