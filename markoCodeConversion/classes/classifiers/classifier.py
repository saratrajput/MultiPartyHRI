# general imports
import numpy as np
# For serializing and de-serializing python object structure
import pickle
import logging

# custom modules
from classes.helpers import kinect_data_helper, classifier_helper
from config import config


class Classifier(object):
    def __init__(self, name):

        # parameters
        self.name = name
        cs = classifier_helper.ClassifierSettings(name)
        self.feature_indexes = kinect_data_helper.KinectDataHelper(version=3).get_features_indexes(
            cs.feature_list)
        self.label_list = cs.label_list
        self.buffer_length = cs.buffer_length

        # initialisation
        self.buffer = None
        self.decision = None
        self.norm_scaler = None

        # DEBUG:
        self.last_decision = None
        # print 'classifier ', self.name, ' is up!'
        logging.basicConfig(filename=config.Config().root_dir + config.settings.log.path + '/decision_log.log',
                            level=logging.DEBUG)

    def update_data(self, dat):
        # stacks the data into a buffer in a new dimension
        if self.buffer is None:
            self.buffer = np.asarray(dat)[np.newaxis, :]
        elif self.buffer.shape[0] >= self.buffer_length:
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1, :] = np.asarray(dat)
        else:
            self.buffer = np.concatenate((self.buffer, np.asarray(dat)[np.newaxis, :]), axis=0)

        # DEBUG:
        # if self.buffer is not None:
        #     print 'classifier', self.name, ', data shape: ', self.buffer.shape

    def clean_buffer(self):
        self.buffer = None
        self.decision = None
        # if self.name == 'basicEngag':
        #     print 'cleaning -------------------'

    def update_decision(self, decision=False):
        if decision is not None:
            self.decision = decision

        # DEBUG:
        # print self.name, ' decision: ', self.decision

    def get_decision(self):
        return self.decision

    def _take_features(self, data):
        return np.asarray(data).take(self.feature_indexes, axis=0)

    def _load_scaler(self, suffix):

        # load the pickled scaler
        fname = config.Config().root_dir + config.settings.proc.path + '/' + self.name + suffix

        with open(fname, 'rb') as f:
            return pickle.load(f)

    def log_decision_change(self, id, frame_nr, counter):

        # TODO: mmake it nicer... more automatic, not needing to call it all the time
        if self.decision != self.last_decision:
            self.last_decision = self.decision
            try:
                frame = str(int(frame_nr)).zfill(6)
            except:
                frame = '0'
            logging.info('@ ' + str(frame) + ' P  ' + str(id) + ' :: ' + str(self.decision))

            # if self.name == 'basicEngag':
            #     print '******:', counter, '\t, Buf Size ',  self.buffer.shape[0], self.decision
