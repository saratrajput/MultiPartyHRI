import numpy as np
import math
import pickle
from numpy import diff
from scipy.interpolate import UnivariateSpline

# custom classes
from config import dataConfig, classifConfig
from classes.helpers import common, classifier_helper


class FeatureTracker:
    """
    Simple class, which offers basic functionality on a list of features, making it easier to keep track of what is
    happening with the features.
    When initialized with clasifier, the classifier feature list from config will become the source and current
    feature list, otherwise can be directly initialized with a feature list

    Args:
        init_arg: classifier name string or a list of initial features
    """
    def __init__(self, init_arg, verbose=True):

        if isinstance(init_arg, list):
            self.orig_feature_list = init_arg
        elif isinstance(init_arg, str):
            self.orig_feature_list = classifier_helper.get_classifier_setting(init_arg, 'feature_list')

        self.current_feature_list = self.orig_feature_list
        self.verbose = verbose

    def add_features(self, features, position=None):
        # TODO: make some general adder/replacer into common?
        if position is None or position == len(self.current_feature_list) or position == -1:
            self.current_feature_list += common.listify(features)
        else:
            self.current_feature_list = self.current_feature_list[:position] + common.listify(features) +\
                                        self.current_feature_list[position + 1:]

    def features_from_to(self, feat_from, feat_to, position=None):
        """
        updates the current feature list for sake of keeping track of the features used throughout the processing

        Args:
            feat_from: features to be replaced
            feat_to: the new features

        """
        feat_before = self.current_feature_list

        from_idx = []
        for feat in feat_from:
            from_idx.append(self.current_feature_list.index(feat))

        if len(from_idx) == 0:
            print 'Current features: ', self.current_feature_list
            print 'Features to process: ', feat_to
            raise ValueError('Tried to do operation on data, not containing the features!')

        from_idx.sort()
        self.current_feature_list = self.current_feature_list[:from_idx[0]] + common.listify(feat_to) +\
                                    self.current_feature_list[from_idx[0] + len(from_idx):]
        if self.verbose:
            print 'Features changed from ', feat_before, ' to ', self.current_feature_list

    def get_current_feat_idx(self, features):
        """
        Args:
            features: the features of question

        Returns:
            list of indexes of the requested features in the current feature list
        """
        return common.match_lists_to_idx(self.current_feature_list, features)

    def reset_current_features(self):
        """
        when one processing cycle is done, this allows for tracking features again by reseting them
        """
        self.current_feature_list = self.feature_list


class FeaturedBuffer:
    """
    class which holds np data array with feature tracker
    """
    def __init__(self, data=None, features=[]):
        if (data is None and features != []) or (features == [] and data is not None):
            raise ValueError('FeaturedBuffer must be initialized either empty or with bot arguments data and features')
        elif data is not None and data.shape[-1] != len(features):
            raise ValueError('Data shape does not correspond to the number of features')
        self.__buffer = data
        self.ft = FeatureTracker(features)

    def add(self, data, features, position=None):
        if data.shape[-1] != len(features):
            raise ValueError('Data shape does not correspond to the number of features')
        if self.__buffer is None:
            self.__buffer = data

        else:
            if position is None:
                position = -1
            self.__buffer = insert_data(self.__buffer, data, to_idx=position)

        self.ft.add_features(features, position)

    def delete(self, feature):
        # TODO: implement if needed
        pass

    def get_data(self):
        return self.__buffer

    def print_details(self):
        print 'Feature buffer details:'
        print '------------------------------------------------'
        print 'data shape: ', self.__buffer.shape
        print 'features: ', self.ft.current_feature_list
        print '------------------------------------------------'


# ---------------------------------------------------------------------------------------------------- general functions


def interpolate_np(data, time_data, samples=200):
    """
    Args:
        data: np array of shape (nr_time_steps x features)
        time_data: sequence of time step
        samples: the number of desired samples of the new inteporlated array

    Returns:
        interpolated 2 dimensional array along the first axis
    """

    interp_data = []

    # interpolate based on the time diension
    sh = data.shape
    for i in xrange(sh[1]):

        new_x = np.linspace(time_data.min(), time_data.max(), samples, endpoint=True)

        # # cubic
        # interp_vect = interp1d(time, data[:,i], kind='cubic')(new_x)

        # spline
        spl = UnivariateSpline(time_data, data[:, i])
        spl.set_smoothing_factor(0.05)
        interp_vect = spl(new_x)

        interp_data.append(interp_vect)

    return np.asarray(interp_data).T


def insert_data(data, proc_data, source_idx=None, to_idx=None):
    """
        stacks data twice - data up to from_idx with new, and these with data from from_idx on
    Args:
        data: the entire np data array to be updated
        source_idx: indexes of the source data features, representing lines to be replaced
        proc_data: the data to replace the source data
        to_idx: optional index specifying where to insert the data

    Returns:
        updated np data

    """

    if source_idx is not None:
        source_idx = common.listify(source_idx)
        data = np.delete(data, source_idx, axis=1)
        source_idx.sort()

    if source_idx is None and to_idx is not None:
        if to_idx == -1:
            to_idx = data.shape[-1] + 1
        to_idx = common.listify(to_idx)
        source_idx = to_idx

    if source_idx is None and to_idx is None:
        raise ValueError('to insert, either source_idx or to_ids has to be specified')

    if data.ndim > 1:
        return np.hstack((np.hstack((data[:, :source_idx[0]], proc_data)), data[:, (source_idx[0]):]))
    else:
        return np.hstack((np.hstack((data[:source_idx[0]], proc_data)), data[(source_idx[0]):]))


def extract_features(data, current_feat_list, features, flatten=False):
    if flatten:
        return data.take(common.match_lists_to_idx(current_feat_list, features), axis=-1).flatten()
    else:
        return data.take(common.match_lists_to_idx(current_feat_list, features), axis=-1)


def lin_reg_2_feat(x1, x2):
    """
    Args:
        x1: x component of the data
        x2: y component of the data

    Returns:
        returns parameters of the equation x2 = w[0]x1 + w[1]
    """
    a = np.vstack((x1, np.ones(x1.shape)))
    return np.linalg.lstsq(a.T, x2)[0]  # obtaining the parameters


def get_mean_slope(data):
    """
    Args:
        data: np array of shape whatever x 2

    Returns:
        returns mean and the slope parameter only (regressed fit of subtracted mean => b = 0 in y = ax + b) converted to
        degrees
    """
    x1_mu = data[:, 0].mean()
    x1 = data[:, 0] - x1_mu
    x2_mu = data[:, 1].mean()
    x2 = data[:, 1] - x2_mu

    return np.asarray([x1_mu, x2_mu, math.degrees(math.atan(lin_reg_2_feat(x1, x2)[0]))])


# ------------------------------------------------------------------------------------------------- specific processings
# TODO: could be also implemened as Operation class... there is a lot of repetition, same scheme
# class Operation:
#     def __init__(self, name, ):
#          reads from, to features,
#
#     def apply_on_arary(self, overwrite, insert_where):
#
#     def get_outcome(self):


def normalize_body_parts(data, parts_feat, feature_tracker, base_feat=None):

    if base_feat is None:
        base_feat = ['SpineBasePx', 'SpineBasePy']

    # deal with single vectors
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)

    base_idx = feature_tracker.get_current_feat_idx(base_feat)
    feat_idx = feature_tracker.get_current_feat_idx(parts_feat)

    for i in xrange(data.shape[0]):
        norm_vect = subtract_body_parts(data[i, :], base_idx, feat_idx)
        data[i, :] = norm_vect

    return data


def subtract_body_parts(feat_vect, base_idx, feat_idx):
    """
    Normalizes bodyParts w.r.t. the base by subtracting the vector
    Args:
        feat_vect: the entire feature vector of values
        base_idx: index of data in the feat_vect corresponding to the base parts
        feat_idx: analogous

    Returns:
        The feat_vect vector with subtracted body parts
    """

    base_vect_len = len(base_idx)
    parts_nr = len(feat_idx) / base_vect_len

    base_vect = feat_vect[min(base_idx):max(base_idx) + 1]

    for i in range(parts_nr):
        feat_vect[feat_idx[i * base_vect_len]: feat_idx[(i + 1) * base_vect_len - 1] + 1] -= base_vect

    return feat_vect


def shoulders_to_orientation(data, feature_tracker):
    # TODO: seems to sometimes give NaN, find out why
    feat_from = [
        'ShoulderLeftPx',  # 4
        'ShoulderLeftPy',  # 5
        'ShoulderRightPx',  # 6
        'ShoulderRightPy',  # 7
    ]

    feat_to = [
        'ShoulderVectX',
        'ShoulderVectY'
    ]

    from_idx = feature_tracker.get_current_feat_idx(feat_from)

    # actual processing ---------------------------

    x_ = data.take([from_idx[0], from_idx[2]], axis=1)
    y_ = data.take([from_idx[1], from_idx[3]], axis=1)

    # create shoulder vector array
    shoulder_vector = np.vstack((x_[:, 0] - x_[:, 1], y_[:, 0] - y_[:, 1])).T

    # normalize
    norm_vectors = []
    for i in xrange(shoulder_vector.shape[0]):
        norm_vectors.append(shoulder_vector[i, :]/np.linalg.norm(shoulder_vector[i, :]))

    proc_data = np.asarray(norm_vectors)

    # actual processing ---------------------------

    feature_tracker.features_from_to(feat_from, feat_to)
    return insert_data(data, proc_data, source_idx=from_idx)


# ---------------------------------------------------------------------------------------------------------------- slice


def get_trajectory_len(data, current_feat_list):

    feat_from = [
        'SpineBasePx',  # 4
        'SpineBasePy',  # 5
    ]

    feat_to = [
        'AbsLenTraj'
    ]

    from_idx = common.match_lists_to_idx(current_feat_list, feat_from)

    # actual processing ---------------------------

    xy = data.take(from_idx, axis=1)

    proc_data = np.asarray([np.linalg.norm(xy[0, :] - xy[-1, :])])

    # actual processing ---------------------------

    return proc_data, feat_to


def get_traj_mean_slope(data, current_feat_list):

    feat_from = [
        'SpineBasePx',  # 4
        'SpineBasePy',  # 5
    ]

    feat_to = [
        'MeanSpinePx',
        'MeanSpinePx',
        'TrajSlopeDeg'
    ]

    from_idx = common.match_lists_to_idx(current_feat_list, feat_from)

    # actual processing ---------------------------

    xy = data.take(from_idx, axis=1)
    proc_data = get_mean_slope(xy)

    # actual processing ---------------------------

    return proc_data, feat_to


def get_feat_mean(data, current_feat_list, features):

    from_idx = common.match_lists_to_idx(current_feat_list, features)
    return data.take(from_idx, axis=1).mean(axis=0)


def get_traj_quadrant(xy_data):
    """
    Args:
        xy_data: numpy array of [x, y] points (e.g. trajectory)

    Returns:
        number of quadrant 1-4 depending on the direction of the vector [x, y](last) - [x, y](first)
    """
    quadrant = 1

    x = xy_data[-1, 0] - xy_data[0, 0]
    y = xy_data[-1, 1] - xy_data[0, 1]

    if x > 0 and y > 0:
        quadrant = 1
    elif x < 0 and y > 0:
        quadrant = 2
    elif x < 0 and y < 0:
        quadrant = 3
    elif x > 0 and y < 0:
        quadrant = 4

    return quadrant

# ------------------------------------------------------------------------------------------------ high level operations


def process_slice_for_classifier(slice_data, current_feat_list):

    proc_buf = None
    del proc_buf
    # TODO: make it conditioned just like in the data preparation in labl_data_helper
    proc_buf = FeaturedBuffer(data=None, features=[])

    # trajectory mean, slope # TODO: std?
    data, features = get_traj_mean_slope(slice_data, current_feat_list)
    proc_buf.add(data, features)

    # len
    data, features = get_trajectory_len(slice_data, current_feat_list)
    proc_buf.add(data, features)

    # mean shoulders
    shoulder_orient_f = ['ShoulderVectX', 'ShoulderVectY']
    data, features = get_feat_mean(slice_data, current_feat_list, shoulder_orient_f), shoulder_orient_f
    proc_buf.add(data, shoulder_orient_f)

    # TODO: make more generic, no time...
    yaw = ['faceY']
    data = extract_features(slice_data, current_feat_list, yaw, flatten=True)
    # print 'before', data

    new_feat_names = [yaw[0] + str(i) for i in xrange(data.shape[-1])]

    if data.any() == 0.0:

        slope = float(extract_features(proc_buf.get_data(), proc_buf.ft.current_feature_list, ['TrajSlopeDeg']))

        xy_data = extract_features(slice_data, current_feat_list, [
            'SpineBasePx',  # 4
            'SpineBasePy',  # 5
        ])

        quadrant = get_traj_quadrant(xy_data)

        new_yaw_vals = []
        for yaw_val in data.tolist():
            if yaw_val == 0.0:
                if quadrant == 1 or quadrant == 4:
                    val = 90.0 + slope
                    new_yaw_vals.append(val)

                elif quadrant == 2 or quadrant == 3:
                    val = 90.0 - slope
                    new_yaw_vals.append(val)

            else:
                new_yaw_vals.append(yaw_val)

    else:
        new_yaw_vals = data

    proc_buf.add(np.asarray(new_yaw_vals), new_feat_names)

    return proc_buf


# --------------------------------------------------------------------------------------- potentially useful derivatives


# def diffSimple(X, t):
#     '''
#     Takes two consecutive values of X and t, computes derivative
#
#     Args:
#         X: ...
#         t: ...
#
#     Returns:
#         the time derivative
#     '''
#     # np way
#     dX = diff(X)/(t[1] - t[0])
#
#     # manual way
#     dX = (X[1] - X[0])/(t[1] - t[0])
#
#     return dX
#
# def diffDenoise_n4N7(X, t):
#     '''
#     Implement higher order derivative for noise reduction n=2, N=7
#
#     Args:
#         X: ...
#         t: ...
#
#     Returns:
#         time derivative according to: http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
#     '''
#     i = [3, 4, 5, 6, 0, 1, 2]
#     h = sum(t)/len(t)
#     dx = (39*(X[i[1]] - X[i[-1]]) + 12*(X[i[2]] - X[i[-2]]) + 5*(X[i[3]] - X[i[-3]]))/(96*h)
#     return dx
#
# def diffDenoise_n4N11(X, t):
#     '''
#     Implement higher order derivative for noise reduction n=2, N=11
#
#     Args:
#         X: ...
#         t: ...
#
#     Returns:
#         time derivative according to: http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators/
#     '''
#     i = [5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
#     dt = np.diff(t)
#     h = sum(dt)/len(dt)
#     dx = (322*(X[i[1]] - X[i[-1]]) + 256*(X[i[2]] - X[i[-2]]) + 39*(X[i[3]] - X[i[-3]]) + 32*(X[i[4]] - X[i[-4]]) +
#           11*(X[i[5]] - X[i[-5]]))/(1536*h)
#     return dx