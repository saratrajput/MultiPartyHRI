from classifier import Classifier


class Speech(Classifier):
    def __init__(self, name):
        Classifier.__init__(self, name)

    def update_data(self, raw_data):
        data = self._take_features(raw_data)
        super(Speech, self).update_data(data)

        # DEBUG
        # sp = '%010.3f' % self.buffer[-1, 0]
        # mo = '%010.3f' % self.buffer[-1, 1]
        # lo = '%010.3f' % self.buffer[-1, 2]
        # print sp, ' | ', mo, ' | ', lo

    def update_decision(self, decision=False):

        aggregate_mouth_move = self.buffer[:, 0].sum() / self.buffer.shape[0]
        aggregate_look = self.buffer[:, 1].sum() / self.buffer.shape[0]
        aggregate_speak_conf = self.buffer[:-5, 2].sum() / 5

        # DEBUG
        # sp = '%010.3f' % aggregate_speak_conf
        # mo = '%010.3f' % aggregate_mouth_move
        # lo = '%010.3f' % aggregate_look
        # print sp, ' | ', mo, ' | ', lo

        # custom empirical ugly law of decision
        # TODO: make sure it works with all other features!
        if aggregate_speak_conf > 0.98 and \
           aggregate_mouth_move > 1.01: # and \
           # 0.01 < aggregate_look < 2.9:
           decision = self.label_list[1]
        else:
            decision = self.label_list[0]
        super(Speech, self).update_decision(decision)
