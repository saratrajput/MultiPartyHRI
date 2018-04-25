from classifier import Classifier


class Tracking(Classifier):
    def __init__(self, name):
        Classifier.__init__(self, name)
        self.counter = 0

    def update_data(self, raw_data):
        self.counter += 1

    def clean_buffer(self):
        self.counter = 0

    def update_decision(self, decision=False):
        if self.counter > 0:
            self.decision = self.label_list[1]
        else:
            self.decision = self.label_list[0]
