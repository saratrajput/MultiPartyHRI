import numpy as np

from classifier import Classifier
from classes.helpers import classifier_helper


class Engagement(Classifier):
    def __init__(self, name):
        super(Engagement, self).__init__(name)
        cs = classifier_helper.ClassifierSettings(name)
        self.threshold = cs.threshold
        self.hysteresis = 30
        self.hyster_ct = 0


    def update_data(self, raw_data):
        raw_data = self._take_features(raw_data)
        data = self.preprocess(raw_data)
        super(Engagement, self).update_data(data)

    def update_decision(self, decision=False):

        # DEBUG
        # print self.buffer.sum() / self.buffer.shape[0]

        if self.buffer.sum() / self.buffer.shape[0] > self.threshold:
            decision = self.label_list[1]
        else:
            decision = self.label_list[0]

        if self.decision != decision and self.hyster_ct == 0:
            super(Engagement, self).update_decision(decision)
            self.hyster_ct = self.hysteresis
        elif self.hyster_ct > 0:
            self.hyster_ct -= 1

    def preprocess(self, data):
        # take dot product of normalized head position and orientation vector
        head_pos_vect = data[0:3]
        head_orinet_vect = data[3:]

        # DEBUG
        # print np.dot(
        #     head_pos_vect/np.linalg.norm(head_pos_vect),
        #     head_orinet_vect / np.linalg.norm(head_orinet_vect)
        # )

        return [np.dot(
            head_pos_vect/np.linalg.norm(head_pos_vect),
            head_orinet_vect / np.linalg.norm(head_orinet_vect)
        )]
