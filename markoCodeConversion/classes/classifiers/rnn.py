import numpy as np
import time

import matplotlib.pyplot as plt

# custom classes
from config import config
from classifier import Classifier
from classes.helpers import model_helper, data_processor_helper, classifier_helper


class Rnn(Classifier):
    def __init__(self, name):

        super(Rnn, self).__init__(name)

        cs = classifier_helper.ClassifierSettings(name)

        # load just the needed parameters to keep the classifier fast
        self.label_list = [item[0] if isinstance(item, list) else item for item in self.label_list]
        self.dead_steps = cs.dead_steps
        self.add_step_indexes = cs.add_step_indexes
        self.running_window = cs.running_window
        self.smallest_predict_buf = cs.smallest_predict_buf
        self.prediction_threshold = cs.prediction_threshold
        self.parts_to_normalize = cs.parts_to_normalize
        self.shoulder_orientation = cs.shoulder_orientation
        self.slice_size = cs.slice_size
        if self.slice_size:
            self.slice_list_buf = []
        self.norm_scale = cs.normalize
        if self.norm_scale:
            self.norm_scaler = self._load_scaler(config.settings.proc.norm_scaler_suffix)
        self.min_max_scale = cs.min_max_scale
        if self.min_max_scale:
            self.mm_scaler = self._load_scaler(config.settings.proc.mm_scaler_suffix)

        # load the model
        m = model_helper.ModelHelper()
        keras_model = m.loadModel(self.name)
        self.keras_model = m.compileModel(keras_model)

        # some initialisations
        self.decision = self.label_list[0]
        if self.add_step_indexes:
            self.scaled_step_idx = self.normalize(np.asarray(
                [range(self.buffer_length)]*(len(self.feature_indexes)+1), dtype=np.float32).T)[:, -1]

        # # predict once with model to avoid lag caused by "cold run" TODO: make work with slicing too
        # foo_data = np.random.rand(5, len(cs.feature_list))
        # if self.add_step_indexes:
        #     foo_data = np.random.rand(5, len(cs.feature_list) + 1)
        # self.keras_model.predict(foo_data[np.newaxis, :], verbose=0)

    def update_data(self, raw_data):
        feat_trckr = data_processor_helper.FeatureTracker(self.name, verbose=False)
        data = self._take_features(raw_data)
        proc_data = self.preprocess(data, feat_trckr)

        if self.slice_size:
            # keep buffering the slice until it reaches desired size, then process and append to the overall buffer
            if len(self.slice_list_buf) < self.slice_size:
                self.slice_list_buf.append(proc_data)
            else:
                featured_slice_buf = self.process_slice(np.asarray(self.slice_list_buf), feat_trckr)
                norm_slice_data = self.normalize(np.nan_to_num(featured_slice_buf.get_data()))
                super(Rnn, self).update_data(norm_slice_data)
                self.slice_list_buf = []

                # # DEBUG
                # if self.buffer.shape[0] == self.buffer_length:
                #     labl_data_helper.plot_sequence_features(self.buffer)
                #     plt.show()

        else:
            norm_data = self.normalize(proc_data)
            super(Rnn, self).update_data(norm_data)

        # add the sequence length
        if self.add_step_indexes:
            self.buffer[:, -1] = self.scaled_step_idx[:self.buffer.shape[0]]

    def update_decision(self, decision=False):

        decision = self.decision

        # if self.name == 'basicEngag':
        #     print self.dead_steps, self.buffer.shape[0], decision

        # predict only after dead steps and 1/2 frame rate (for now, computational reasons)
        if self.buffer is not None:
            if self.buffer.shape[0] > self.dead_steps and (self.buffer.shape[0] % 2) == 0:
                # st = time.time()
                predictions = self.keras_model.predict(self.buffer[np.newaxis, :],
                                                       verbose=0)

                # DEBUG
                # e = time.time()
                # print 'prediction of size ', self.buffer.shape[0], ' took ', e - st, 's.'

                nr_last = min([self.buffer.shape[0], self.running_window])
                # ensure running window with smallest prediction buffer size
                if nr_last < self.smallest_predict_buf:
                    nr_last = self.smallest_predict_buf

                average_prediction = np.average(np.squeeze(predictions)[-nr_last:], axis=0)

                if np.max(average_prediction) > self.prediction_threshold:
                    label = np.where(average_prediction == np.max(average_prediction))[0][0]
                    decision = self.label_list[label]

            # print '%04.3f' % average_prediction[0],\
            #         '%04.3f' % average_prediction[1],\
            #         '%04.3f' % average_prediction[2],\
            #         self.decision

        # DEBUG
        # print self.name
        # print self.label_list
        # print decision
        super(Rnn, self).update_decision(decision)

    def preprocess(self, data, feat_trckr):

        data = data_processor_helper.normalize_body_parts(data,
                                                          self.parts_to_normalize,
                                                          feat_trckr)

        if self.shoulder_orientation:
            data = data_processor_helper.shoulders_to_orientation(data, feat_trckr)

        # make the vector one longer for the sequence length
        if self.add_step_indexes:
            data = np.resize(data, (data.shape[1] + 1))[np.newaxis, :]

        return data[0]      # TODO: change the normalizer to return a simple list?

    def normalize(self, data):

        # get the right shape for the scalers
        data = np.squeeze(data)[np.newaxis]

        if self.min_max_scale:
            data = self.mm_scaler.transform(data)

        if self.normalize:
            data = self.norm_scaler.transform(data)

        return np.squeeze(data)

    def process_slice(self, data, feat_trckr):

        return data_processor_helper.process_slice_for_classifier(data, feat_trckr.current_feature_list)
