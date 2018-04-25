from classifier import Classifier
from classes.helpers import classifier_helper


class Presence(Classifier):
    def __init__(self, name):
        super(Presence, self).__init__(name)
        cs = classifier_helper.ClassifierSettings(name)
        self.dead_steps = cs.dead_steps
        self.threshold = cs.threshold

    def update_data(self, raw_data):
        data = self._take_features(raw_data)
        super(Presence, self).update_data(data)

    def update_decision(self, decision=False):
        # DEBUG:
        # print self.buffer.shape
        # print self.buffer[:, 0].std()
        # print self.buffer[:, 1].std()

        if self.buffer.shape[0] > self.dead_steps:
            # print self.buffer[:, 0].std(), self.buffer[:, 1].std()
            if self.buffer[:, 0].std() < self.threshold and self.buffer[:, 1].std() < self.threshold:
                decision = self.label_list[1]
            else:
                decision = self.label_list[0]
            super(Presence, self).update_decision(decision)
