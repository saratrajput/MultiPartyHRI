# GENERAL IMPORTS
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from tqdm import *
import random

# sciKitLearn
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import preprocessing

# Keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

# CUSTOM IMPORTS
from classes.helpers import common, data_processor_helper, classifier_helper, kinect_data_helper
from config import config, dataConfig, classifConfig


class MetaData:

    # ----------------------------------------------------------------
    # filename;    origin;    IDs;    fromIndex;    toIndex;    labels
    # ----------------------------------------------------------------

    def __init__(self, string=None):
        if string is not None:
            self.from_string(string=string)
        else:
            self.npz_f_name = None
            self.origin_f_name = None
            self.ids = None
            self.from_idx = None
            self.to_idx = None
            self.labels = None

    def print_metadata(self):
        print self.to_string(warn=False)

    def to_string(self, warn=True):
        if warn:
            empty_items = ['Items empty!' for a in dir(self) if not a.startswith('__') and a is None]
            if len(empty_items) > 0:
                print 'Warning meta data being turned into string with some fields not initialized!'
                self.print_metadata()
                raw_input()

        return str(self.npz_f_name) + ';' + \
               str(self.origin_f_name) + ';' + \
               ','.join([str(i) for i in self.ids]) + ';' + \
               str(self.from_idx) + ';' + \
               str(self.to_idx) + ';' + \
               ','.join(self.labels)

    def from_string(self, string):
        # clean string
        string = string.replace(" ", "")
        string = string.rstrip('\n')
        string = string.rstrip('\r')

        # divide subcomponents
        meta_list = string.split(';')

        # read each
        self.npz_f_name = meta_list[0]
        self.origin_f_name = meta_list[1]
        self.ids = [int(item) for item in meta_list[2].split(',')]
        self.from_idx = int(meta_list[3])
        self.to_idx = int(meta_list[4])
        self.labels = meta_list[5].split(',')


class Data(object):
    # class that reads all holds all the read data as properties
    def __init__(self, classifier):

        self.cs = classifier_helper.ClassifierSettings(classifier)
        self.ft = data_processor_helper.FeatureTracker(classifier)

        self._data_list = None
        self.time_data_list = None
        self.seq_lens = None
        self._data_padded_np = None

        self.targets_raw = None
        self.targets_list = None
        self.targets_padded_np = None

        self.nr_samples = None
        self.nr_labels = None
        self.nr_features = None

    # properties (getters)          -------------------
    @property
    def data_list(self): return self._data_list

    @property
    def data_padded_np(self): return self._data_padded_np

    # setters                       -------------------
    @data_list.setter
    def data_list(self, value):
        self.data_list_change(value)

    @data_padded_np.setter
    def data_padded_np(self, value):
        self.data_padded_np_change(value)

    # callbacks                     -------------------
    def data_list_change(self, value):

        check_nan(value)

        new_seq_lens = get_sequence_lengths(value)
        pad_len = min(self.cs.sequence_trim, max(new_seq_lens))

        if self.seq_lens != new_seq_lens:
            self.seq_lens = new_seq_lens
            if self._data_list is not None:
                self.targets_list = targets_raw_to_list(self.targets_raw, self.seq_lens)
                self.targets_padded_np, pad_len = keras_pad_list(self.targets_list,
                                                                 padding=self.cs.padding_type,
                                                                 max_len=pad_len)
            if self.nr_samples != len(new_seq_lens):
                print 'Warning! Number of samples changed, verify consistency with targets!'
                self.nr_samples = len(new_seq_lens)

        self._data_padded_np, pad_len = keras_pad_list(value,
                                                       padding=self.cs.padding_type,
                                                       max_len=pad_len)

        self.nr_features = value[0].shape[-1]

        self._data_list = value

    def data_padded_np_change(self, value):
        if self._data_padded_np is not None:
            print 'Warning! data_padded_np change may have caused inconsistency with data_list!'
            self._data_list = None
        self._data_padded_np = value

    # functions                     -------------------
    def load(self, data_list=None):

        if data_list is None:
            self.data_list, self.targets_raw, target_dict, self.time_data_list =\
                read_dir_to_list(classifier=self.cs.classifier_name, directory=config.settings.label.current_dir)
        else:
            self.targets_raw = [0]
            if data_list[0].shape[-1] != len(self.cs.feature_list):
                raise ValueError('Trying to manually load data with different number of features as the classifier'
                                 'contains!')
            else:
                self.data_list = data_list
        # self.data_list = remove_feature_from_list(data_raw, 0)

        self.nr_labels = max(self.targets_raw) + 1
        self.targets_list = targets_raw_to_list(self.targets_raw, self.seq_lens)
        self.targets_padded_np, pad_len = keras_pad_list(self.targets_list,
                                                         padding=self.cs.padding_type,
                                                         max_len=self.cs.sequence_trim)

    def preprocess(self, trim=True):
        # TODO: trim setting is misleading, only affects the data_list, not the np array!!!
        if self.cs.limit_instances is not None:
            print 'limiting instances...'
            for key in self.cs.limit_instances:
                self.data_list, targets_raw = limit_class_instances(self.data_list, self.targets_raw, key,
                                                                    self.cs.limit_instances[key])

        if self.cs.normalize_body_parts:
            print 'normalizing w.r.t. spine...'
            self.data_list = normalize_body_parts(self.data_list,
                                                  self.cs.parts_to_normalize,
                                                  self.ft)

        if self.cs.shoulder_orientation:
            print 'computing shoulder orientation...'
            self.data_list = shoulders_to_orientation(self.ft, self.data_list)

        if self.cs.add_step_indexes:
            print 'adding step indexes...'
            self.data_list = add_timestep_nrs(self.data_list)

        if self.cs.sequence_trim is not None and trim:
            print 'trimming sequences...'
            self.data_list = crop_data_list_front(self.data_list, self.cs.sequence_trim)

    def normalize(self, save_scalers=False, use_scalers=False):

        if self.cs.min_max_scale:
            print 'scaling data...'
            self.data_list = min_max_scale(self.data_list, classifier=self.cs.classifier_name, save=save_scalers, use_scaler=use_scalers)

        if self.cs.normalize:
            print 'normalizing data...'
            self.data_list = normalize_data(self.data_list, classifier=self.cs.classifier_name, save=save_scalers, use_scaler=use_scalers)

    def slice_process(self):

        # only perform if the slice size is defined
        if self.cs.slice_size:
            print 'creating and processing slices...'
            sliced_data_list = []

            for sequence in tqdm(self.data_list):
                sliced_sequence = []
                sequence_slices = slice_sequence(sequence, self.cs.slice_size)
                for seq_slice in sequence_slices:
                    feat_buf = data_processor_helper.process_slice_for_classifier(seq_slice, self.ft.current_feature_list)
                    sliced_sequence.append(feat_buf.get_data())

                # DEBUG
                # plot_sequence_features(np.asarray(sliced_sequence), feature_names=feat_buf.ft.current_feature_list), plt.show()

                sliced_data_list.append(np.asarray(sliced_sequence))

            # update data
            # TODO: update all not only data list!!!
            self.ft = feat_buf.ft
            self.data_list = sliced_data_list

    def print_details(self):
        # TODO: complete, nice feature for debug...

        print 'Dataset details:'
        print '------------------------------------------------'
        print 'number of samples: ', self.nr_samples
        print 'number of labels: ', self.nr_labels
        print 'number of features: ', self.nr_features
        print 'np data shape: ', self.data_padded_np.shape
        print 'np targets shape: ', self.targets_padded_np.shape
        print '------------------------------------------------'


# -------------------------------------------------------------------------------------------------------------- reading


def get_f_names_in_dir(directory=False, full_path=False):
    # TODO: use the one from common and check for beginnings
    directory = resolve_dir(directory)

    # read all files in the corresponding directory, either full path or name only
    if full_path:
        labeled_data_files = [''.join([root, name])
                              for root, dirs, files in os.walk(directory)
                              for name in files
                              if (name.startswith(("#")) and name.endswith((".npz")) and (
                              not (root.endswith(("test")))))]
    else:
        labeled_data_files = [name
                              for root, dirs, files in os.walk(directory)
                              for name in files
                              if (name.startswith(("#")) and name.endswith((".npz")) and (
                              not (root.endswith(("test")))))]
    return labeled_data_files


def read_dir_to_list(directory=False, labels=False, features=False, persons=1, classifier=False, version=4):

    '''

    ...

    FORMAT:
        data:
            list of [ndarray(recording_length x Kinect_message_length), ndarray(rec... ...]
        readLabels:
            list (of lists) of labels
    '''

    # TODO: multilabel vs singlelabel data!!!!!!!!!!!

    print 'Started reading the files...'

    kdh = kinect_data_helper.KinectDataHelper(version=version)
    cs = classifier_helper.ClassifierSettings(classifier)

    # if the classifier is specified, load data from the classifConfig, otherwise manually
    if classifier:
        labels = cs.source_label_list
        features = kdh.get_features_indexes(cs.feature_list)
        persons = cs.persons

    target_d = dict(zip(range(len(labels)), labels))
    target_d_inv = dict(zip(labels, range(len(labels))))

    # return init
    data = []
    t_data = []
    targets = []

    # get all filenames
    directory = resolve_dir(directory)
    data_f_names = get_f_names_in_dir(directory)

    used_files = []
    # loop through files finding data and labels
    for fileName in tqdm(data_f_names[:]):
        # print fileName
        single_data, found_labels, time_data = read_labl_file(fileName,
                                                              directory=directory, labels=labels,
                                                              features=features, persons=persons)

        if found_labels:
            targets.append(target_d_inv[found_labels[0]])
            data.append(single_data)
            t_data.append(time_data)
            used_files.append(fileName)
            # sys.stdout.write('.')
    # print ''

    # merge the classes (just the targets in fact)
    targets = merge_classes(classifier, targets)

    merged = cs.source_label_list

    print len(targets), '/', len(data_f_names), ' files were used.'
    print 'persons: ' + str(persons)
    print 'labels:'
    for label, i in zip(cs.source_label_list, range(len(cs.source_label_list))):
        print ' - ', targets.count(i), '\t', label
        print ' - ', targets.count(i), '\r', label
        try:
            pr = merged[label]
            print '\t', pr
        except:
            pass
    print 'features:'
    print features
    print '--------'

    # debug features
    # for labeledDataFileName, sequence, target, i in zip(used_files, data, targets, range(len(data))):
    #     self.plot_2_features_from_buf(plt.figure(1, figsize=(8, 6)), data[i:i+1], targets[i:i+1],
    #                           classifier, title=labeledDataFileName, feat = [1, 2]), plt.show()

    return data, targets, target_d, t_data


def read_labl_file(f_name, directory=False, labels=False, features=False, check_corrupt=True, persons=1, report=False):
    # TODO: refactor!!!
    '''
    Reads an .npz file and returns the data + list of labels of the data
    directory - default according to config
    labels - single or a list of desired labels ( TODO: multiple labels!!!!!!) in their full string form
    features - single or a list of features to extract from the Kinect Message, in number form
    persons - number of persons in the specific data that will be taken into consideration

    FORMAT:
        data:
            ndarray(recording_length x Kinect_message_length)
        readLabels:
            list (of lists) of labels

    '''

    conf = dataConfig.dataConfig()

    # initialize variables
    readLabels = []
    validData = False

    # get the proper name
    directory = resolve_dir(directory)
    full_f_name = directory + f_name
    if not (f_name.endswith('.npz')):
        full_f_name += '.npz'

    # read the file
    npzfile = np.load(full_f_name)
    npzfile.files

    # check if there is an empty data timestamp in the beginning or end of the data
    if check_corrupt == True:
        check_corrupt_crop(npzfile['data'], f_name)

    check_nan(npzfile['data'], data_id=f_name)

    # check if the number of persons corresponds to the requirement, otherwise return false, false
    sh = npzfile['data'].shape
    # TODO: find a way to deal with the past wrong decision of storing multiple persons, the current is very fragile

    if not round(float(sh[1]) / conf.KinectMessageLength[0]) == persons:
        # print persons
        # print 'refused', round(float(sh[1]) / conf.KinectMessageLength)
        return False, False, False

    # based on argument look for any or specific labels
    if labels is False or labels == 'default':
        checkedLabels = conf.labelsDict.keys()
    else:
        checkedLabels = labels

    # load data and search for labels in the .npz data
    for lab in checkedLabels:
        datAvailable = False
        try:
            labelPresent = npzfile[lab]
            if labelPresent == False:
                pass
            else:
                datAvailable = True
                validData = True
        except:
            pass
        if datAvailable:
            readLabels.append(lab)

    # based on argument look for any or specific labels
    if not (features == False or features == 'default'):
        data = npzfile['data'].take(features, axis=1)
        for i in range(1, persons):
            print 'processing multiple!!!!!!'
            print data.shape
            next_pick = [idx + 1 for idx in features]
            data = np.concatenate((data, npzfile['data'].take(next_pick, axis=1)), axis=1)
            print data.shape
    else:
        data = npzfile['data']

    time_data = npzfile['data'][:, 0]

    if report:
        print 'File ', full_f_name, ' read successfully.'

    # check if both "and" labels are present
    allLablPresent = True
    if checkedLabels[0] == 'and':
        for i in range(1, len(labels)):
            if not (readLabels[i] in readLabels):
                allLablPresent = False
        if allLablPresent:
            return data, readLabels
        else:
            return False, False, False

    # check if the single label was found and return
    else:
        if validData:
            return data, readLabels, time_data
        else:
            return False, False, False

# -------------------------------------------------------------------------------------------------------- *manipulation


def crop_data_list(data_list, md):

    i = 0
    for person_id in md.ids:
        full_id_axis = np.asarray(data_list[person_id])
        curr_data = full_id_axis[md.from_idx:md.to_idx, :]
        if i > 0:
            data = np.hstack((data, curr_data))
        else:
            data = curr_data
        i += 1

    return data


def slice_sequence(sequence, slice_size):

    seq_len = sequence.shape[0]
    nr_slices = int(math.ceil(seq_len / slice_size))

    return [sequence[ptr:ptr + slice_size] if ((ptr + slice_size) < seq_len) else sequence[ptr:]
            for ptr in [i*5 for i in xrange(nr_slices)]]


def crop_data_list_front(data_list, length):
    for sequence, i in zip(data_list, range(len(data_list))):
        if sequence.shape[0] > length:
            data_list[i] = sequence[-length:]
    return data_list


def remove_feature_from_list(data_list, feature_idx):
    data, orig_sh_indexes = data_list_to_feat_np(data_list)

    index_list = common.listify(feature_idx)
    trimmed_data = np.delete(data, index_list, axis=1)

    return feature_np_to_data_list(trimmed_data, orig_sh_indexes)


def data_list_to_feat_np(data_list):
    '''
    Args:
        data_list: data in the standard dataBuf fromat

    Returns:
        np array of shape (sequences*theirLengths)*features
    '''

    ptr = 0
    orig_shape_idx = [ptr]
    for sequence, i in zip(data_list, range(len(data_list))):
        if i == 0:
            reshapedData = sequence
            # origShInd = [0, int(sequence.shape[0])]

        else:
            reshapedData = np.concatenate((reshapedData, sequence), axis=0)
            # origShInd.append(origShInd[i-1] + int(sequence.shape[0]))

        ptr += int(sequence.shape[0])
        orig_shape_idx.append(ptr)

    return reshapedData, orig_shape_idx


def feature_np_to_data_list(data, orig_shape_idx):
    data_list = []
    orig_shape_idx = orig_shape_idx
    for i in range(len(orig_shape_idx) - 1):
        data_list.append(data[orig_shape_idx[i]:orig_shape_idx[i + 1], :])

    return data_list


def limit_class_instances(data_list, targets_raw, class_nr, limit):
    class_idx = np.where(np.asarray(targets_raw, dtype=np.int32) == class_nr)[0]
    class_samples = class_idx.shape[0]
    if class_samples > limit:
        sample_idx_to_remove = random.sample(class_idx, class_samples - limit)

        print 'Will remove ', len(sample_idx_to_remove), ' instances of class ', class_nr
        remaining_idx = list(set(range(len(data_list))) - set(sample_idx_to_remove))

        data_list = common.list_take(data_list, remaining_idx)
        targets_raw = common.list_take(targets_raw, remaining_idx)

    return data_list, targets_raw

# -------------------------------------------------------------------------------------------- treatment & feature engin


def add_timestep_nrs(data_list):
    # adds a vector of timestep order to every sequence
    for sequence, i in zip(data_list, range(len(data_list))):
        data_list[i] = np.hstack((sequence, np.arange(sequence.shape[0])[:, np.newaxis]))
    return data_list


def shoulders_to_orientation(feature_tracker, data_list):

    data, orig_sh_indexes = data_list_to_feat_np(data_list)

    data = data_processor_helper.shoulders_to_orientation(data, feature_tracker)

    return feature_np_to_data_list(data, orig_sh_indexes)


def normalize_data(data_list, classifier=False, save=False, use_scaler=False):
    '''
    normalizes the dataBuf: zero mean, unit variance
    saves the normalisation details (scipy saler) so that they can be used for online normalisation of the streams
    returns the normalised dataBuf
    '''

    if use_scaler:
        # load the pickled scaler
        fname = config.Config().root_dir + config.settings.proc.path + '/' +\
                classifier + config.settings.proc.norm_scaler_suffix

        with open(fname, 'rb') as f:
            scaler = pickle.load(f)

    else:
        reshaped_data, orig_sh_indexes = data_list_to_feat_np(data_list)
        scaler = preprocessing.StandardScaler().fit(reshaped_data)

    # DEBUG
    # print scaler
    # print 'mean:\t ' + str(scaler.mean_)se
    # print 'scale:\t ' + str(scaler.scale_)
    # debug ----------------------

    for sequence, i in zip(data_list, range(len(data_list))):
        normalized_seq = scaler.transform(sequence)
        if i == 0:
            normalized_list = [normalized_seq]
        else:
            normalized_list.append(normalized_seq)

    # if set so, save the normalisation vector for online processing
    if save is True and classifier is not False:
        normalization_f_name = config.Config().root_dir + config.settings.proc.path + '/' + \
                               str(classifier) + config.settings.proc.norm_scaler_suffix
        print 'Saving norm scaler: ' + str(scaler)
        with open(normalization_f_name, 'wb+') as f:
            pickle.dump(scaler, f)

            # DEBUG
            # with open(normalisationFileName, 'rb') as f:
            #     scaler = pickle.load(f)
            # print 'loaded: ' + str(scaler)

    return normalized_list


def min_max_scale(data_list, classifier=False, save=False, use_scaler=False):

    data, orig_sh_indexes = data_list_to_feat_np(data_list)

    if use_scaler:
        # load the pickled scaler
        fname = config.Config().root_dir + config.settings.proc.path + '/' +\
                classifier + config.settings.proc.mm_scaler_suffix

        with open(fname, 'rb') as f:
            scaler = pickle.load(f)

    else:
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(data)

    scaled_data = scaler.transform(data)

    if save is True and classifier is not False:
        normalization_f_name = config.Config().root_dir + config.settings.proc.path + '/' + \
                               str(classifier) + config.settings.proc.mm_scaler_suffix
        print 'Saving minmax scaler: ' + str(scaler)
        with open(normalization_f_name, 'wb+') as f:
            pickle.dump(scaler, f)

    return feature_np_to_data_list(scaled_data, orig_sh_indexes)


def normalize_body_parts(data_list, parts_feat, feature_tracker, base_feat=None):
    # Normalizes bodyParts w.r.t. the spine base by subtracting the vector

    data, orig_sh_indexes = data_list_to_feat_np(data_list)

    norm_data = data_processor_helper.normalize_body_parts(data, parts_feat, feature_tracker, base_feat=base_feat)

    return feature_np_to_data_list(norm_data, orig_sh_indexes)


def keras_pad_list(data_list, padding='pre', max_len=None):
    # turns data into ndarray(samples, dim_1, dim_1) and targets into ndarray(samples)

    if max_len is None:
        lens = []
        for i in range(len(data_list)):
            lens.append(data_list[i].shape[0])
        max_len = max(lens)

    return pad_sequences(data_list, maxlen=max_len, dtype='float32', padding=padding), max_len


def apply_pca(data, components=False):
    # TODO: refactor and make an option in data loader
    # implment pca based on the article

    print 'Applying PCA...'

    # set the number of components to keep after PCA
    if components == False:
        components = data.shape[2]

    sh = data.shape
    data = np.reshape(data, (sh[0]*sh[1], sh[2]))

    sklearn_pca = sklearnPCA(n_components=components)

    PCAfited = sklearn_pca.fit(data)
    # The amount of variance that each PC explainss
    print PCAfited.explained_variance_ratio_

    # Cumulative Variance explains
    print np.cumsum(np.round(PCAfited.explained_variance_ratio_, decimals=4) * 100)

    PCAtransformed = sklearn_pca.fit_transform(data)
    if components < 1:
        print PCAtransformed.shape
        components = -1

    outData = np.reshape(PCAtransformed, (sh[0], sh[1], components))

    print 'New data shape after PCA: ' + str(data.shape)
    return outData

# ---------------------------------------------------------------------------------------------------------------- utils


def resolve_dir(directory):
    # either sets default directory or leaves the variable unchanged
    if directory is False or directory == 'default':
        directory = config.settings.label.current_dir
    # else:
    #     directory = config.settings.label.path + '/' + directory
    return directory + '/'

# ----------------------------------------------------------------------------------------------------------------- misc


def merge_classes(classifier, targets):
    cs = classifier_helper.ClassifierSettings(classifier)

    source_labels_d_i = common.dictify(cs.source_label_list, inv=True)
    labels_d_i = common.dictify(cs.label_list, inv=True)
    labels_d = common.dictify(cs.label_list, inv=False)
    to_merge_d = cs.dict_to_merge

    # for all new labels (there might be a simpler way, this has historical reasons):
    for i in range(len(labels_d_i)):
        # if the label is a merged one, for all merged targets replace with the value of the merged label
        if labels_d[i] in to_merge_d:
            for source_label in to_merge_d[labels_d[i]]:
                targets = [labels_d_i[labels_d[i]] if x == source_labels_d_i[source_label] else x for x in targets]
        # in the other case just do it for each single instance
        else:
            targets = [labels_d_i[labels_d[i]] if x == source_labels_d_i[labels_d[i]] else x for x in targets]
    return targets


def check_corrupt_crop(data, fileName):
    # TODO make it work for multiperson data as well!!!!!!!!!!!!
    # checks whether when cropping, there has been some zero-time steps included in the sequence in front or back
    frontEmpty = True
    backEmpty = True
    frontCt = 0
    backCt = -1

    while frontEmpty:
        # print data[frontCt:].shape
        if np.count_nonzero(data[frontCt, :]) == 1:
            print ''
            print 'file ' + fileName + ' crop problem at front'
            print data[frontCt, :]
        else:
            frontEmpty = False
        frontCt += 1

    while backEmpty:
        # print data[:backCt].shape
        if np.count_nonzero(data[backCt, :]) == 1:
            print ''
            print 'file ' + fileName + ' crop problem at back'
            print data[backCt, :]
        else:
            backEmpty = False
        backCt -= 1
    return 0


def check_nan(data, data_id=None, replace=True):
    '''

    Informs about np.nan in data of any format - data_list, np_dataset or single_np
    Replaces the np.nan with a 0.0

    Args:
        data: data to be checked for np.nan
        data_id: identifier of the data in the case of single sequence

    '''
    if isinstance(data, list):
        for sequence, i in zip(data, range(len(data))):
            if np.isnan(sequence).any():
                print 'np.nan found in sequence ', i, 'with shape', sequence.shape
                print np.where(np.isnan(sequence) == True)
                if replace:
                    print 'replacing np.nan...'
                    data[i] = np.nan_to_num(sequence)

    elif data.ndim > 2:
        for i in range(data.shape[0]):
            if np.isnan(data[i]).any():
                print 'np.nan found in sequence ', i, 'with shape', data[i].shape
                print np.where(np.isnan(data[i]) == True)
                if replace:
                    print 'replacing np.nan...'
                    data[i] = np.nan_to_num([i])

    elif data.ndim == 2:
        if np.isnan(data).any():
            if data_id is not None:
                print 'np.nan found in ', data_id, 'with shape', data.shape
            else:
                print 'np.nan found in sequence with shape', data.shape
            print np.where(np.isnan(data) == True)
            if replace:
                print 'replacing np.nan...'
                data = np.nan_to_num(data)

    elif isinstance(data, np.ndarray):
        if np.isnan(data).any():
            print 'np.nan found in the data'
            print data
            if replace:
                print 'replacing np.nan...'
                data = np.nan_to_num(data)

    return data


def targets_raw_to_list(targets_raw, seq_lengths):
    # converts an array of raw targets [0, 3, ...] into list of arrays in shape of nr_labels x seq_len in
    # categorical form [1, 0, 0, 0] [0, 0, 0, 1]..., ...
    targets_categ = to_categorical(targets_raw)
    targets_list = []
    for i in range(len(seq_lengths)):
        time_step_distributed = np.asarray(np.tile(targets_categ[i], (seq_lengths[i], 1)))
        targets_list.append(time_step_distributed)

    return targets_list


def targets_distributed_to_raw(targets):
    out_list = []

    if isinstance(targets, list):
        targets = list_to_np(targets)

    for i in xrange(targets.shape[0]):
        out_list.append(int(np.where(targets[i][-1] == 1.0)[0]))
    return out_list


def list_to_np(data_list):
    i = 0
    for sequence in data_list:
        if i == 0:
            out_data = np.expand_dims(sequence, axis=0)
        else:
            out_data = np.vstack((out_data, np.expand_dims(sequence, axis=0)))
        i += 1
    return out_data


def get_sequence_lengths(data_list):
    lens = []
    for item in data_list:
        lens.append(item.shape[0])
    return lens

# ------------------------------------------------------------------------------------------------------------- plotting


def plot_2_features(fig, data, title=None, feat=[0, 1]):

    nr_classes = data.nr_labels

    labelNames = []
    for i in range(nr_classes):
        txt = 'C' + str(i+1)
        labelNames.append(txt)

    fig.suptitle('Subject trajectories in robot frame', fontsize=14)

    ax = fig.add_subplot(111)
    # fig.subplots_adjust(top=0.85)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')

    ax.text(0, 0, 'ROBOT', fontsize=13, bbox={'facecolor': 'grey', 'alpha':0.5, 'pad':5}, verticalalignment='center', horizontalalignment='center')

    colors = plt.cm.rainbow(np.linspace(0, 1, nr_classes))

    t = [0] * nr_classes

    for sequence, target in zip(data.data_list, data.targets_raw):
        lab = 'C' + str(target+1)
        plt.plot(sequence[:, feat[0]], sequence[:, feat[1]], color=colors[target], label=lab if t[target] == 0 else "")
        t[target] += 1

    plt.legend(loc=1, borderaxespad=0., fontsize=10) # bbox_to_anchor=(1.05, 1),
    plt.axis('equal')
    plt.title(title)


def plot_2_features_from_buf(fig, data_dict, targets, classifier, title=None, feat=[0, 1]):

    nr_classes = len(classifier_helper.get_classifier_setting(classifier, 'label_list'))

    labelNames = []
    for i in range(nr_classes):
        txt = 'C' + str(i+1)
        labelNames.append(txt)

    fig.suptitle('Subject trajectories in robot frame', fontsize=14)

    ax = fig.add_subplot(111)
    # fig.subplots_adjust(top=0.85)
    ax.set_xlabel('x[m]')
    ax.set_ylabel('y[m]')

    ax.text(0, 0, 'ROBOT', fontsize=13, bbox={'facecolor':'grey', 'alpha':0.5, 'pad':5}, verticalalignment='center', horizontalalignment='center')

    colors = plt.cm.rainbow(np.linspace(0, 1, nr_classes))

    t = [0] * nr_classes

    for sequence, target in zip(data_dict, targets):
        lab = 'C' + str(target+1)
        plt.plot(sequence[:, feat[0]], sequence[:, feat[1]], color=colors[target], label=lab if t[target] == 0 else "")
        t[target] += 1

    plt.legend(loc=1, borderaxespad=0., fontsize=10) # bbox_to_anchor=(1.05, 1),
    plt.axis('equal')
    plt.title(title)


def plot_sequence_features(sequence, features=None, feature_names=None, title=None):

    if not features:
        features = range(sequence.shape[1])
    if not feature_names:
        feature_names = features

    sh = sequence.shape
    print 'Printing sequence data for features ' + str(features) + ' out of ' + str(sh[1])

    i = 0
    for feature in features:
        plt.subplot(len(features), 1, i + 1)
        plt.plot(range(sh[0]), sequence[:, feature])

        plt.title(feature_names[i])
        i += 1

    # plot settings
    plt.autoscale(enable=True, axis='both', tight=None)
    if title is not None:
        plt.suptitle(title)


# ----------------------------------------------------------------------------------------------------------------- misc


def interpolate_data_list(data_list, time_data, samples=200):

    print 'interpolating and resampling data to ' + str(samples) + ' samples...'
    out_data = []

    for dat, t_vect in zip(data_list, time_data):
        dat = data_processor_helper.interpolate_np(dat, t_vect, samples)
        out_data.append(dat)

    return out_data

# ----------------------------------------------------------------------------------------------------------------------
# TODO: get rid of as much as possible...
# ----------------------------------------------------------------------------------------------------------------------


class LabeledDataHelper:
    # TODO: take out whats needed, eliminate the class if possible
    def __init__(self):
        conf = dataConfig.dataConfig()
        self.labelsDict = conf.labelsDict
        self.labels = conf.labelsDict.keys()
        self.fileNameLength = 5

# ---------------------------------------------------------------------------------------------------------------- debug

    def print_statistics(self, directory=False, listFiles=False):

        print ' '
        print '------ Labeled data statistics -------'
        print ' '

        if directory == False or directory == 'default':
            conf = dataConfig.dataConfig()
            directory = conf.labelPath

        # get the actual datafiles by searching the default(optional directories), excluding test
        labeledDataFiles = ['/'.join([root, name])
             for root, dirs, files in os.walk(directory)
             for name in files
             if (name.startswith(("#")) and name.endswith((".npz")) and (not(root.endswith(("test")))))]

        filesCt = len(labeledDataFiles)

        # get the dataFileas according to the metadatalist
        labeledMetaDataFiles = []
        SingleLabelCounterDict = {el:0 for el in self.labels}
        MultiLabelCounterDict = {el:0 for el in self.labels}
        fullMetaFile = conf.labelPath + conf.labelsMetadataFileName + '.txt'

        with open(fullMetaFile) as f:
            lineList = f.readlines()

            # read line by line
            for line in lineList:
                md = MetaData(line)

                # collect the filename
                labeledMetaDataFiles.append(md.origin_f_name)

                # update the corresponding statistics
                if len(md.ids) == 1:
                    try:
                        SingleLabelCounterDict[md.labels] += 1
                    except:
                        for label in md.labels:
                            SingleLabelCounterDict[label] += 1
                else:
                    try:
                        MultiLabelCounterDict[md.labels] += 1
                    except:
                        for label in md.labels:
                            MultiLabelCounterDict[label] += 1

        # prints
        print 'Number of files: ' + str(filesCt)
        print 'Sinlgle person interaction counts:'
        print(json.dumps(SingleLabelCounterDict, indent=4))

        print 'Multiple person interaction counts:'
        print(json.dumps(MultiLabelCounterDict, indent=4))

        # TODO: check consistency of meta and actual, print list of matches, print statistics over the labels

        filesCt = len(labeledDataFiles)
        metaFilesCt = len(labeledMetaDataFiles)

        # list all the files found
        if listFiles:
            print ''
            print 'List of files:'
            for i in xrange(len(labeledDataFiles)):
                print labeledDataFiles[i]

        print ' '
        print '------ Labeled data statistics -------'
        print ' '

# --------------------------------------------------------------------------------------------------- argument resolving

    def resolveDir(self, directory):

        # either sets default directory or leaves the variable unchanged
        if directory == False or directory == 'default':
            directory = directory = config.settings.label.current_dir + '/'
        return directory

    def resolveFeatures(self, features):
        # if needed (features are given in string form), uses the config dict to convert features to integer form
        if type(features[0]) == type(1):
            return features

        elif type(features[0]) == type('a'):
            dataDict = kinect_data_helper.KinectDataHelper().get_data_dict()
            newFeatures = []
            i = 0
            for feature in features:
                newFeatures.append(dataDict[feature])
                i += 1

            # extend features with the time value!!!!
            if newFeatures[0] != 0:
                newFeatures = [0] + newFeatures

            return newFeatures

        else:
            print 'Invalid list of features?'
            print features
            return False

    def resolveClassifFeatures(self, features, classifier):
        if type(features[0]) == type(1):
            return features

        elif type(features[0]) == type('a'):
            cC = classifConfig.classifConfig()
            featDict = cC.getDictFromList(cC.getFeatList(classifier))

            newFeatures = []
            for feature in features:
                newFeatures.append(featDict[feature])
            return newFeatures

        else:
            print 'Invalid list of features?'
            print features
            return False

    def resolveLabels(self):
        # TODO: if needed (labels are given in string form), uses the config dict to convert labels to integer form
        pass

    def resolveSliceParam(self, sliceSize, overlap):
        # TODO: broken, does not work for intentional overlap = 0!
        if sliceSize == False:
            cC = classifConfig.classifConfig()
            sliceSize = cC.sliceSize
        if overlap == False:
            cC = classifConfig.classifConfig()
            overlap = cC.overlap
        return sliceSize, overlap

# ---------------------------------------------------------------------------------------------------- file manipulation

    def get_f_names_in_dir(self, directory=False, fullPath=False):

        directory = self.resolveDir(directory)

        # read all files in the corresponding directory, either full path or name only
        if fullPath:
            labeled_data_files = [''.join([root, name])
                 for root, dirs, files in os.walk(directory)
                 for name in files
                 if (name.startswith(("#")) and name.endswith((".npz")) and (not(root.endswith(("test")))))]
        else:
            labeled_data_files = [name
                 for root, dirs, files in os.walk(directory)
                 for name in files
                 if (name.startswith(("#")) and name.endswith((".npz")) and (not(root.endswith(("test")))))]
        return labeled_data_files

    def merge_classes(self, classifier, targets):

        cs = classifier_helper.ClassifierSettings(classifier)

        source_labels_d_i = common.dictify(cs.source_label_list, inv=True)
        labels_d_i = common.dictify(cs.label_list, inv=True)
        labels_d = common.dictify(cs.label_list, inv=False)
        to_merge_d = cs.dict_to_merge

        # for all new labels (there might be a simpler way, this has historical reasons):
        for i in range(len(labels_d_i)):
            # if the label is a merged one, for all merged targets replace with the value of the merged label
            if labels_d[i] in to_merge_d:
                for source_label in to_merge_d[labels_d[i]]:
                    targets = [labels_d_i[labels_d[i]] if x == source_labels_d_i[source_label] else x for x in targets]
            # in the other case just do it for each single instance
            else:
                targets = [labels_d_i[labels_d[i]] if x == source_labels_d_i[labels_d[i]] else x for x in targets]
        return targets

    def create_labelled_file(self, data, meta_data, directory_arg=None, file_name_arg=None, meta_f_name_arg=None):
        '''
        Creates a .npz file with the following structure:
        keywords:   ['data',self.labels]
        values:     [data, True, False, True ... dependin on the dataLabels passed to the function]

        example symbolic print of the file content for dataLabels = ['PassingUninterested', 'PassingInterested']:
                    ['data', 'PassingUninterested', 'PassingPossiblyInterested', 'PassingInterested', 'Engaged', 'Disengaging', 'LeavingUninterested']
                    [array([[ 2000.   ,  2000.   ,  2000.   , ...,  2000.   ,  2000.   ,  2000.   ],
                       [ 2000.001,  2000.001,  2000.001, ...,  2000.001,  2000.001,
                         2000.001],
                       [ 2000.002,  2000.002,  2000.002, ...,  2000.002,  2000.002,
                         2000.002],
                       ...,
                       [ 2000.997,  2000.997,  2000.997, ...,  2000.997,  2000.997,
                         2000.997],
                       [ 2000.998,  2000.998,  2000.998, ...,  2000.998,  2000.998,
                         2000.998],
                       [ 2000.999,  2000.999,  2000.999, ...,  2000.999,  2000.999,
                         2000.999]]), True, False, True, False, False, False]

        calls handleMetadata for keeping track of the files created
        '''

        if file_name_arg:
            file_name = file_name_arg
        else:
            md = self.read_last_metadata()
            integer_name = int(md.npz_f_name[1:])
            next_integer_name = integer_name + 1
            file_name = '#' + str(next_integer_name).zfill(self.fileNameLength)
        meta_data.npz_f_name = file_name

        directory = config.settings.label.current_dir + '/'
        if directory_arg is not None:
            directory = config.settings.label.path + '/' + directory_arg + '/'

        full_f_name = directory + file_name + '.npz'

        labelArray = ['data'] + self.labels
        labelVals = [False for x in range(len(self.labels))]

        for lab in meta_data.labels:
            labelVals[self.labelsDict[lab]] = True
        dataArray = [data] + labelVals

        saved_md_file = meta_f_name_arg
        if meta_f_name_arg != 'no':
            saved_md_file = self.add_meta_data_entry(meta_data, meta_f_name=meta_f_name_arg)

        np.savez(full_f_name, **{name: value for name, value in zip(labelArray, dataArray)})

        return full_f_name, meta_data, saved_md_file

    def add_meta_data_entry(self, meta_data, meta_f_name=None):
        # add meta data entry into specified meta file or the current one

        if meta_f_name is None:
            meta_f_name = config.settings.label.current_meta_f
        full_f_name = config.settings.label.path + '/' + meta_f_name + '.txt'

        meta_data = '\n' + meta_data.to_string()

        with open(full_f_name, 'a') as outfile:
            outfile.write(meta_data)

        return full_f_name

    def read_last_metadata(self):
        # will read the last metadata entry of file in the list
        conf = dataConfig.dataConfig()
        full_f_name = conf.labelPath + conf.labelsMetadataFileName + '.txt'
        with open(full_f_name) as f:
            line_list = f.readlines()
            i = 0
            while 1:
                line = line_list[-1-i]
                if line[0] == '#':
                    return MetaData(line)
                else:
                    i += 1

                    
# ---------------------------------------------------------------------------------------------- data index manipulation

    def getRandomTrainTestIndexes(self, targets, percRatio):
        # creates a list of lists of randomly shuffled indexes of the training and testing samples, with specific ratio

        labelInd = self.sort_idx_by_labl(targets)

        # randomly shuffle the indexes
        for i in xrange(len(labelInd)):
            np.random.shuffle(labelInd[i])

        # divide the indexes in the desired ratio
        for i in range(len(labelInd)):
            labelInd[i] = self.split_list(labelInd[i], percRatio)

        # join the indexes into list of training list and test index lists
        for i in range(len(labelInd)):
            if i == 0:
                train = labelInd[i][0]
                test = labelInd[i][1]
            else:
                train += labelInd[i][0]
                test += labelInd[i][1]

        print 'Setting ', percRatio*100, '\% of the samples for training ( ', len(train), ' / ', len(test), ' )'
        return zip([train], [test])

    def sort_idx_by_labl(self, targets):
        # sorts the indexes into a (label-) list of lists of sample indexes
        labels = set(targets)
        labelInd = range(len(labels))
        i = 0
        for label in labels:
            arr = np.where(np.array(targets) == label)[0]
            labelInd[i] = arr.tolist()
            i += 1
        return labelInd

    def split_list(self, a_list, ratio):
        if ratio > 1:
            ratio = ratio / 100.0
        splitInd = int(len(a_list) * ratio)
        return [a_list[:splitInd], a_list[splitInd:]]

# ------------------------------------------------------------------------------------------------------- data treatment


# --------------------------------------------------------------------------------------- potentially useful derivatives

    # def takeFeaturesDerivatives(self, data, tVect, features):
    #     '''
    #     will take derivatives of features, which must be subset of the selected feature list in conofig
    #     derivatives are concatenated to the feature dimension in the form (0, dx/dt1,..) due to n-1 dimension nature
    #     '''
    #
    #     cC = classifConfig.classifConfig()
    #     allFeatures = cC.basicEngagFeatList
    #     i=0
    #     featDict = {k:v for k,v in zip(allFeatures,range(len(allFeatures)))}
    #     for feature in features:
    #
    #         # get the sequence corresponding to the feature
    #         featureSequenec = data[:, feature]
    #
    #         # differentiation
    #         derivatives = np.diff(featureSequenec)/np.diff(tVect)
    #         derivatives = np.insert(derivatives, 0, 0)
    #
    #         # build the matrix of derivatives
    #         if i == 0:
    #             derivData = derivatives
    #         else:
    #             derivData = np.vstack((derivData, derivatives))
    #         i += 1
    #
    #     # reshape to the form (n_timesteps, n_features)
    #     derivData = derivData.T
    #     return derivData
    #
    # def takeFeaturesDerivatives_n4N7(self, data, tVect, features):
    #     '''
    #     will take derivatives of features, which must be subset of the selected feature list in conofig
    #     derivatives are concatenated to the feature dimension in the form (0, dx/dt1,..) due to n-1 dimension nature
    #     '''
    #
    #     pD = data_processor_helper.procData()
    #     cC = classifConfig.classifConfig()
    #     allFeatures = cC.basicEngagFeatList
    #     i=0
    #     featDict = {k:v for k,v in zip(allFeatures, range(len(allFeatures)))}
    #     for feature in features:
    #
    #         # get the sequence corresponding to the feature
    #         featureSequence = data[:, feature]
    #
    #         # differentiation
    #         dx = []
    #         for k in range(len(featureSequence) - 6):
    #             dx.append(pD.diffDenoise_n4N7(featureSequence[k:k+7], tVect[k:k+7]))
    #
    #         # compensate for the central trimming by floor(7/2)
    #         dx = [0,0,0] + dx + [0,0,0]
    #
    #         # build the matrix of derivatives
    #         if i == 0:
    #             derivData = np.asarray(dx)
    #         else:
    #             derivData = np.vstack((derivData, np.asarray(dx)))
    #         i += 1
    #
    #     # reshape to the form (n_timesteps, n_features)
    #     derivData = derivData.T
    #     return derivData
    #
    # def takeFeaturesDerivatives_n4N11(self, data, tVect, features):
    #     '''
    #     will take derivatives of features, which must be subset of the selected feature list in conofig
    #     derivatives are concatenated to the feature dimension in the form (0, dx/dt1,..) due to n-1 dimension nature
    #     '''
    #
    #     # features = self.resolveClassifFeatures(features)
    #
    #     pD = data_processor_helper.procData()
    #     cC = classifConfig.classifConfig()
    #
    #     i = 0
    #     for feature in features:
    #
    #         # get the sequence corresponding to the feature
    #         featureSequence = data[:, feature]
    #
    #         # differentiation
    #         dx = []
    #         for k in range(len(featureSequence) - 10):
    #             dx.append(pD.diffDenoise_n4N11(featureSequence[k:k+11], tVect[k:k+11]))
    #
    #         # compensate for the central trimming by floor(7/2)
    #         dx = [0,0,0,0,0] + dx + [0,0,0,0,0]
    #
    #         # build the matrix of derivatives
    #         if i == 0:
    #             derivData = np.asarray(dx)
    #         else:
    #             derivData = np.vstack((derivData, np.asarray(dx)))
    #         i += 1
    #
    #     # reshape to the form (n_timesteps, n_features)
    #     derivData = derivData.T
    #     return derivData
    #
    # def addDerivatives(self, dataBuf, tData, features, classifier):
    #     '''
    #     for all sequences in dataBuf extends the feature space by the derivatives of corresponding features from arg
    #     '''
    #
    #     print 'Adding derivatives of features: ' + str(features)
    #
    #     features = self.resolveClassifFeatures(features, classifier)
    #
    #     i = 0
    #     for dat,tDat in zip(dataBuf,tData):
    #
    #         derivData = self.takeFeaturesDerivatives_n4N11(dat, tDat, features)
    #
    #         dat = np.concatenate((dat, derivData), axis=1)
    #         dataBuf[i] = dat
    #         i += 1
    #
    #     return dataBuf
    #
