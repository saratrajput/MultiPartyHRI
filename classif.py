# -------------------------------------------------------------------------------------------------------------- imports
#  general imports

# Provides a portable way of using OS dependent functionality
import os

# Makes it easy to write user-friendly command-line interfaces
import argparse

# Plotting library which produces publication quality figures in various formats
import matplotlib.pyplot as plt

import matplotlib
# Fundamental package for scientific computing with Python
import numpy as np

from operator import add
# For displaying a 'fancy' progress bar
from tqdm import *
# For serializing and de-serializing a Python object structure
import pickle
# For time-related functions
import time
# For implementing pseudo-random number generators
import random

# SciKit Learn
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split, StratifiedKFold, StratifiedShuffleSplit

# custom classes
from classes.helpers import common, model_helper, labl_data_helper, classifier_helper, plotly_functions
from config import config

matplotlib.rc('font', family='Times New Roman')
# ---------------------------------------------------------------------------------------------------------------- debug


def get_toy_dataset():
    # 1D
    # toySeq = [np.arrange(10, 2), np.asarray(range(10, 25)), np.asarray(range(25, 43))]
    # targets = [0, 1, 0]

    # 2D
    X1, Y = np.mgrid[0:10, 0:2]
    X2, Y = np.mgrid[10:25, 0:2]
    X3, Y = np.mgrid[25:32, 0:2]
    toySeq = [X1, X2, X3]
    targets = [0, 1, 0]

    return (toySeq, targets)


def plot_seq_histogram(data_list, targets, nr_labels):
    lens = []
    for i in range(len(data_list)):
        lens.append(config.settings.kin_frame_dur * data_list[i].shape[0])

    lengths = []
    [lengths.append([]) for i in range(nr_labels)]

    print('Min length: ' + str(min(lens)) + ', max length: ' + str(max(lens)))

    [lengths[i-1].append(val) for i, val in zip(targets, lens)]

    try:
        import plotlyX
        plotly_functions.plot_histogram(lengths)
    except ImportError:

        fig = plt.figure(figsize=config.settings.fig.fig_size_2)
        ax = fig.add_subplot(111)

        colors = config.settings.fig.colors_set_1

        # multiple-histogram of data-sets with different length
        # ax3.hist(lengths, 20, histtype='bar', color=fig)
        ax.hist(lengths, 30, histtype='step', color=colors,
                label=['no interest', 'paying attention', 'approaches'])
        ax.legend(prop={'size': 12})
        plt.xlabel('[seconds]')
        # ax.set_title('Sequence length histogram')
        plt.tight_layout()
        plt.show()

        # simple histogram
        # plt.hist(lens)
        # plt.title("Sequence length histogram")

# ------------------------------------------------------------------------------------------------ scikit classification


def skl_multi_classif(classifier=False):

    # TODO: check if it works as expected!

    d = labl_data_helper.Data(classifier)
    d.load()

    data = labl_data_helper.interpolate_data_list(d.data_list, d.time_data_list)

    # ------------------------------------------------------------------------

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression

    names = ["Nearest Neighbors", "Linear SVM", "LogReg", "RBF SVM", "Decision Tree",
             "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
             "Quadratic Discriminant Analysis"]
    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        LogisticRegression(C=1e5),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis()]

    # load custom dataset into X,y, preprocess dataset, split into training and test part
    X, y = data, d.targets_raw
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

    print('Score for the classifiers:')
    print('\tName\t\t\tmean\t\tstd')
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        scores = cross_validation.cross_val_score(clf, X, y, cv=10)
        print(' :: ' + name.ljust(22, ' ') + '\t' + str(scores.mean()) + '\t' + str(scores.std()))

# -------------------------------------------------------------------------------------------------------------  helpers


class Evaluator:
    # TODO: batch of accuracies and holding the confusion matrix are... confusing, refine to intuitive way.
    # TODO: make use of a Metrics class instead of just bling indexing of some outputs
    def __init__(self, nr_classes, batch_size=32, verbose=1):
        self.confusion_m = np.zeros((nr_classes, nr_classes))
        self.nr_of_classes = nr_classes

        self.batch = batch_size
        self.verbose = verbose

        self.batch_accuracies = []

    def evaluate_complete_next(self, model, x, y, x_orig=None, y_orig=None):

        self.reset_cm()

        if x_orig is None or y_orig is None:
            x_orig = x
            y_orig = y

        l, a = model.evaluate(x, y, batch_size=self.batch, verbose=self.verbose)
        self.evaluate_custom(model, x_orig, y_orig)

        self.batch_accuracies.append(a)

        return [l, a, self.get_accuracy_from_cm(), self.get_cm()]

    def get_batch_variance(self):
        return np.std(np.asarray(self.batch_accuracies))

    def reset(self):
        self.reset_cm()
        self.batch_accuracies = []

    def reset_cm(self):
        self.confusion_m = np.zeros((self.nr_of_classes, self.nr_of_classes))

    def evaluate_complete_once(self, model, x, y, x_orig=None, y_orig=None):
        return self.evaluate_complete_next(model, x, y, x_orig, y_orig)

    def evaluate_custom(self, model, X, y):
        if isinstance(X, (list, tuple, type(None))):
            for seq, trg in zip(X, y):
                history = model.predict(seq[np.newaxis])
                history = history.squeeze()
                self.update_cm(np.asarray(history[-1]).argmax(), trg[-1].argmax())
        else:
            for i in range(X.shape[0]):
                history = model.predict(np.expand_dims(X[i], axis=0))
                history = history.squeeze()
                self.update_cm(np.asarray(history[-1]).argmax(), y[i, -1].argmax())

        return np.asarray(history[-1]).argmax()

    def update_cm(self, prediction, label):
        self.confusion_m[prediction, label] += 1

    def get_cm(self):
        return self.confusion_m

    def get_accuracy_from_cm(self):
        trace = self.confusion_m.trace()
        sum = self.confusion_m.sum()
        return trace/sum

    def evalPlotter(self, nr_sequences, targets, results, cases, nr_classes=None):
        # TODO: terrible code, make nicer, more generic
        if nr_classes is None:
            if isinstance(targets, list):
                nr_classes = targets[0].shape[-1]
            else:
                nr_classes = targets.shape[-1]

        # display the results for each sequence from its beginning for every timestep
        for seq_nr in range(nr_sequences):

            if nr_classes > 3:
                colors = plt.cm.rainbow(np.linspace(0, 1, nr_classes))
            else:
                colors = config.settings.fig.colors_set_1

            label_names = []
            for i in range(nr_classes):
                txt = 'C' + str(i+1)
                label_names.append(txt)

            # get the class corresponding to the actual sequence
            if isinstance(targets, list):
                actual_class = np.where(targets[seq_nr][0] == 1.0)
            else:
                actual_class = np.where(targets[seq_nr, 0] == 1.0)

            for res, case in zip(results, cases):
                fig = plt.figure(figsize=config.settings.fig.fig_size_1)

                if case is not None:
                    fig.suptitle(case)

                ax = fig.add_subplot(111)
                fig.subplots_adjust(top=0.85)
                ax.set_xlabel('time step')
                ax.set_ylabel('prediction confidence')

                # print results[sequence]
                for label in range(nr_classes):
                    if isinstance(targets, list):
                        plt.plot(list(range(res[seq_nr][:, label].shape[0])), res[seq_nr][:, label], color=colors[label],
                                 label=label_names[label])
                    else:
                        plt.plot(list(range(res[seq_nr][:, label].shape[0])), res[seq_nr][:, label], color=colors[label],
                                 label=label_names[label])

                    if actual_class[0][0] == label:
                        ax.set_title('Actual class : ' + str(label + 1),
                                     color=colors[actual_class[0][0]], fontweight='bold', fontsize=12)

                plt.legend(loc=7, borderaxespad=0., fontsize=10) # bbox_to_anchor=(1.05, 1),
                plt.tight_layout()

            print('Printing result of test sequence ', seq_nr, '/', nr_sequences, '...')
            plt.show()


class GSResultsReader:
    def __init__(self, gs_nr):

        # TODO: make it a choice paired with GSnr if needed
        details_d = {
            'standard': [['DOValue', 'architecture', 'batch', 'lr', 'init'], ['epochs', 'loss', 'acc', 'Acc', 'cM']]
        }

        self.details = details_d['standard']
        self.dir = config.Config().root_dir + config.settings.results.gs_path + '/' + str(gs_nr) + '/'

        self.results = {
            'parameters': [],
            'performances': []
        }

        print('reading from ', self.dir, end=' ')
        print('results structure: ')
        print(self.details)

    def read_results(self):
        f_names = self.get_res_f_names_in_dir()

        for f_name in f_names:
            self.read_res_file(f_name)

    def print_best_results(self, nr_best, metric=3):

        metric_list = []
        for k in range(len(self.results['performances'])):
            metric_list.append(self.results['performances'][k][metric])

        metric_arr = np.asarray(metric_list)
        print(metric_arr)
        best_idx = np.argsort(-metric_arr)[:nr_best]

        performs = []
        params = []
        for idx in best_idx:
            params.append(self.results['parameters'][idx])
            performs.append(self.results['performances'][idx])

        for param, perform in zip(params, performs):
            print(param, perform)

    # private classes:
    def get_res_f_names_in_dir(self):

        # read all files in the corresponding directory
        f_names = [''.join([root, f_name])
                   for root, dirs, files in os.walk(self.dir)
                   for f_name in files]
        return f_names

    def read_res_file(self, fname):
        print(fname)
        with open(fname, 'rb') as file:
            try:
                res_list = pickle.load(file)
            except:
                res_list = None
                print(' reading file failed!')

        if res_list is not None:
            for k in range(1, len(res_list)):
                self.results['parameters'].append(res_list[0])
                self.results['performances'].append(res_list[k])


def read_gs_results(gs_nr, metric=3):

    nr_best = 30

    reader = GSResultsReader(gs_nr)
    reader.read_results()
    reader.print_best_results(nr_best, metric=metric)


def plot_prompt(data):

    if input('Show sequence length histogram? (y/n): ') == 'y':
        plot_seq_histogram(data.data_list, data.targets_raw, data.nr_labels), plt.show()

    if input('Show trajectories? (y/n): ') == 'y':
        labl_data_helper.plot_2_features(plt.figure(1, figsize=(8, 6)), data), plt.show()

    if input('Show features for sequence by sequence? (y/n): ') == 'y':
        for sequence, i in zip(data.data_list, list(range(data.nr_samples))):
            labl_data_helper.plot_sequence_features(sequence,
                                                    feature_names=data.ft.current_feature_list,
                                                    title=data.cs.label_list[data.targets_raw[i]])
            plt.show()
            if input('\t Next sequence? (y/n): ') != 'y':
                break


def build_architectures(nHiddenLayers, layerSizes):
    # TODO: refactor
    '''

    Args:
        nHiddenLayers: list of integer number of possible hidden layers the generated architectures should contain
        layerSizes: list of integer values

    Returns:
        list of architectures in which the next layer is not smaller than a specified ratio

    Example:
        build_architectures([0,1,2],[4,16,32])
        >> Generated 19 architectures.
        >> [[4], [16], [32], [4, 4], [16, 4], [16, 16], [32, 16], [32, 32], [4, 4, 4], [16, 4, 4], [16, 16, 4],
           [16, 16, 16], [32, 16, 4], [32, 16, 16], [32, 32, 16], [32, 32, 32]]
    '''

    # init
    ratio = 0.3
    arch = []
    nextLayer = [layerSizes[-1]]
    hiddenLayers = list(range(nHiddenLayers[-1] + 1))

    # for each number of hidden layers
    for nHidden in hiddenLayers:

        lastLayer = nextLayer
        nextLayer = []
        for lastArch in lastLayer:

            if nHidden == 0:
                parent = lastArch
            else:
                parent = lastArch[len(lastArch)-1]

            for siz in layerSizes:
                if (siz <= parent and siz >= int(parent*ratio)) or nHidden == 0:
                    if nHidden == 0:
                        nextLayer.append([siz])
                    else:
                        nextLayer.append(lastArch + [siz])

        arch = arch + nextLayer

    # remove smaller architectures
    i = 0
    while i < len(arch):
        if len(arch[i]) < nHiddenLayers[0] + 1:
            arch.remove(arch[i])
        else:
            i += 1

    print('Generated ' + str(len(arch)) + ' architectures.')
    print(arch)
    return arch


def data_loader(classifier, view=False, data_list=None, save_scalers=True, use_scalers=False, trim=True):
    # reads data for specific classifier plots and returns...
    d = labl_data_helper.Data(classifier)
    d.load(data_list)

    print('Dataset loaded')
    d.print_details()
    if view:
        plot_prompt(d)

    d.preprocess(trim=trim)
    print('Dataset preprocessed')
    d.print_details()
    if view:
        plot_prompt(d)

    d.slice_process()
    print('Dataset sliced')
    d.print_details()
    if view:
        plot_prompt(d)

    d.normalize(save_scalers=save_scalers, use_scalers=use_scalers)
    print('Dataset normalized')
    d.print_details()
    if view:
        plot_prompt(d)

    return d


# ------------------------------------------------------------------------------------------------- keras classification


def keras_recurrent_train(classifier, view=False, test_ratio=0.2, folds=10):

    # training settings
    verbose = 0

    # load the classifier parameters
    cs = classifier_helper.ClassifierSettings(classifier)

    arch = cs.architecture
    DOValue = cs.dropout_value
    lr = cs.learning_rate
    init = cs.init
    batch = cs.batch_size
    activ = cs.activ
    epochs = cs.epochs
    momentum = 0

    # prepare dataset
    d = data_loader(classifier, view=view, save_scalers=True)

    # minimum one sample per class for testing
    if test_ratio == 0:
        test_ratio = d.nr_labels

    # prepare test/train data
    sss = StratifiedShuffleSplit(d.targets_raw, 1, test_size=test_ratio, random_state=random.randint(1, 1000000))
    for train_idx, test_idx in sss:
        X_train = d.data_padded_np.take(train_idx, axis=0)
        y_train = d.targets_padded_np.take(train_idx, axis=0)
        X_test = d.data_padded_np.take(test_idx, axis=0)
        y_test = d.targets_padded_np.take(test_idx, axis=0)

    # prepare the indexes for validation folds
    skf = StratifiedKFold(common.list_take(d.targets_raw, train_idx), n_folds=folds, shuffle=True)

    np.random.seed(1337)

    # ----------------------------------------------------------        evaluation

    # model parameters set from data
    input_size = d.nr_features
    n_labels = d.nr_labels

    # init for model manipulation
    m = model_helper.ModelHelper()

    # reporting settings
    reportAtEpochs = list(range(epochs))
    trainForEpochs = np.diff([0] + reportAtEpochs)
    epoch_steps = len(reportAtEpochs)

    train_model_path = "%s%s" % (config.Config().root_dir, config.settings.models.training_path)
    common.clear_dir(train_model_path)

    # depending on whether cross-validation is selected perform corrsponding evaluation
    models = []

    X_trains = []
    y_trains = []
    X_vals = []
    y_vals = []

    # create model and dataset for each fold
    for train_idx, val_idx in skf:

        models.append(m.makeModelFromParam(input_size, n_labels, arch, DOValue, lr, init, activ, momentum))

        X_trains.append(X_train.take(train_idx, axis=0))
        y_trains.append(y_train.take(train_idx, axis=0))
        X_vals.append(X_train.take(val_idx, axis=0))
        y_vals.append(y_train.take(val_idx, axis=0))

    evaluator = Evaluator(n_labels, batch, verbose=verbose)

    metric = 2

    # init metrics
    #         [ loss, accuracy, custom Accuracy, confusion matrix ]
    # TODO: create some reasonable class for metrics
    total_epoch_val_metrics = [[0, 0, 0, np.zeros((n_labels, n_labels))]]*len(trainForEpochs)
    total_epoch_val_metrics_var = []
    best_of_epoch_val_metrics = [0]*len(trainForEpochs)
    best_of_epoch_test_metrics = []
    model_saved_for_fold = []

    # loop through iterations, at each train further each model with corresponding dataset and save the best model
    for epochs, epochBatch, e in zip(reportAtEpochs, trainForEpochs, list(range(epoch_steps))):

        print('Epoch ', e, '--------------------------')

        epoch_metrics = []
        best_at_epoch = 0

        evaluator.reset()

        # for each model at the
        for model, X_tr, y_tr, X_val, y_val, i in zip(models, X_trains, y_trains, X_vals, y_vals, list(range(folds))):

            print('training on fold ', i)

            # train the model
            model.fit(X_tr, y_tr,
                      nb_epoch=epochBatch,
                      verbose=verbose,
                      batch_size=batch)

            # collect the results from evaluate_custom and custom evaluator

            epoch_metrics.append(evaluator.evaluate_complete_next(model, X_val, y_val))

            # update the best model and metrics at the given epoch
            if epoch_metrics[-1][metric] > best_at_epoch:
                best_at_epoch = epoch_metrics[-1][metric]
                best_model = i
                model_saved_for_fold.append(i)
                best_of_epoch_val_metrics[e] = epoch_metrics[-1]

        # average metrics and add to collector
        epoch_sum = [0, 0, 0, np.zeros((n_labels, n_labels))]
        for item, i in zip(epoch_metrics, list(range(len(epoch_metrics)))):
            epoch_sum = list(map(add, epoch_sum, item))
        epoch_average = [item/folds for item in epoch_sum]
        total_epoch_val_metrics[e] = list(map(add, total_epoch_val_metrics[e], epoch_average))
        total_epoch_val_metrics_var.append(total_epoch_val_metrics[e] + [evaluator.get_batch_variance()])

        print(total_epoch_val_metrics_var[e])

        # test the best model and save it
        best_of_epoch_test_metrics.append(evaluator.evaluate_complete_once(models[best_model], X_test, y_test))
        m.saveModel(models[best_model], classifier + '_' + str(epochs), train_model_path)

    # find the best performing combination of model and epoch
    metrics_only = []
    for metr in total_epoch_val_metrics:
        metrics_only.append(metr[:-1])

    metrics_only = np.asarray(metrics_only)
    # overwrite the first column to track the order
    metrics_only[:, 0] = list(range(epoch_steps))

    # take the indexes in correct order
    sorted_idx = metrics_only[metrics_only[:, metric].argsort()[::-1][:epoch_steps]][:, 0]

    print(sorted_idx.shape)
    sorted_idx = sorted_idx.astype(int).tolist()
    print(sorted_idx)

    # print the results
    for epoch in sorted_idx[0:30]:
        print('----------------------- @ ', reportAtEpochs[epoch])
        print(total_epoch_val_metrics_var[epoch])
        print(best_of_epoch_val_metrics[epoch])
        print(best_of_epoch_test_metrics[epoch])

    ans = input('See predictinos with any of the models? : ')
    model_nr = None
    try:
        model_nr = int(ans)
    except:
        pass

    if ans == 'y' or model_nr:
        X_test = common.list_take(d.data_list, test_idx)
        y_test = common.list_take(d.targets_list, test_idx)
        print(X_test[0].shape)
        print(y_test[0].shape)
        while 1:
            if model_nr is None:
                model_nr = input('Model number? (or q) : ')
            if model_nr != 'q':

                test_model = m.loadModel(classifier + '_' + str(model_nr), path=train_model_path)
                test_model = m.compileModel(test_model)

                evaluator = Evaluator(d.nr_labels)
                print(evaluator.evaluate_complete_once(test_model, d.data_padded_np.take(test_idx, axis=0),
                                                       d.targets_padded_np.take(test_idx, axis=0)))

                plot_predictions(test_model, X_test, y_test)
            else:
                if input('Quit?') == 'y':
                    break

            model_nr = None


def keras_recurrent_debug(classifier, load=False, save=False, plot=False, best=False, analyse=False):
    # # TODO: make use of Data class!!!
    #
    # # load own classes
    # ldh = labl_data_helper.LabeledDataHelper()
    #
    # # ----------------------------------------------------------        load dataset
    # sequenceTrim = 250
    # padding = 'pre'
    #
    # # TODO: load the classifier parameters and codition the data treatment based on that
    #
    # # read all data, interpolate and get rid of the time axis
    # allData, targets, targetDict = ldh.read_dir_to_list(classifier=classifier)
    # timeData, dataBuf = ldh.separateTimeFromBuf(allData)
    #
    # nr_sequences = len(dataBuf)
    #
    # # prepare the targets
    # targets_categ = to_categorical(targets)
    # targets = []
    # for i in range(nr_sequences):
    #     targets.append(target_to_distributed(targets_categ[i], dataBuf[i].shape[0]))
    #
    # if common.get_classifier_param(classifier, 'normalize_body_parts'):
    #     common.get_classifier_param(classifier, 'parts_to_normalize')
    #     dataBuf = ldh.normalize_body_parts(dataBuf,
    #                                       classifier,
    #                                       ['SpineBasePx',
    #                                         'SpineBasePy'],
    #                                       common.get_classifier_param(classifier, 'parts_to_normalize')
    #                                       )
    #
    # if common.get_classifier_param(classifier, 'add_step_indexes'):
    #     dataBuf = ldh.add_timestep_nrs(dataBuf)
    #
    # if common.get_classifier_param(classifier, 'normalize'):
    #     dataBuf = ldh.normalize_data(dataBuf, classifier=classifier, save=save)
    #
    # if plot: plt.figure(0), plot_seq_histogram(dataBuf), plt.show()
    #
    # ldh.cropBufFrontToLen(dataBuf, sequenceTrim)
    #
    # if plot: plt.figure(0), plot_seq_histogram(dataBuf), plt.show()
    # if plot: ldh.plot_2_features_from_buf(plt.figure(1, figsize=(8, 6)), dataBuf, targets, classifier), plt.show()
    #
    # X_train_buf, X_test_buf, y_train, y_test = train_test_split(dataBuf, targets, test_size=0.2, random_state=0)
    #
    # X_train, padLenTr = ldh.kerasPadBuf(X_train_buf, padding=padding, maxLen=sequenceTrim)
    # y_train, padLenTr = ldh.kerasPadBuf(y_train, padding=padding, maxLen=sequenceTrim)
    # X_test, padLenTe = ldh.kerasPadBuf(X_test_buf, padding=padding, maxLen=sequenceTrim)
    # y_test, padLenTr = ldh.kerasPadBuf(y_test, padding=padding, maxLen=sequenceTrim)
    #
    # print 'Dataset ready:'
    # try:
    #     print 'Train/test data shape : ' + str(X_train.shape) + ' / ' + str(X_test.shape)
    # except:
    #     print 'Train/test data len,shape : ' + str(len(X_train)) + ' , ' + str(X_train[0].shape) + ' / ' # + str(X_test.shape)
    # try:
    #     print 'Train/test target shape : ' + str(y_train.shape) + ' / ' + str(y_test.shape)
    # except:
    #     print 'Train/test target len,shape : ' + str(len(y_train)) + ' , ' + str(y_test[0]) + ' / ' # + str(X_test.shape)
    #
    # np.random.seed(1337)
    #
    # # ----------------------------------------------------------        evaluation
    #
    # # load the classifier parameters
    # # TODO: refactor from classifier_helper!!!
    # cC = classifConfig.classifConfig()
    # classifierConfig = cC.read_classifier_conf(classifier)
    #
    # arch = classifierConfig['architecture']
    # DOValue = classifierConfig['dropout_value']
    # lr = classifierConfig['learning_rate']
    # init = classifierConfig['init']
    # batchSize = classifierConfig['batch_size']
    # epochs = classifierConfig['epochs']
    # activ = classifierConfig['activ']
    # momentum = 0
    #
    # # model parameters set from data
    # inpSize = X_train.shape[2]
    # nLabels = y_train.shape[-1]
    #
    # # init for model manipulation
    # M = model_helper.ModelHelper()
    # eval = Evaluator(y_test[0,0].shape[0])
    #
    # if load:
    #     print 'loading ', classifier, ' model...'
    #     model = M.loadModel(classifier)
    #     model = M.compileModel(model)
    # else:
    #     model = M.makeModelFromParam(inpSize, nLabels, arch, DOValue, lr, init, activ, momentum)
    #     if best:
    #         best_acc = 0
    #         for i in xrange(epochs):
    #             model.fit(X_train, y_train, nb_epoch=1, batch_size=batchSize)
    #             eval.evaluate_custom(model, X_test, y_test)
    #             if eval.get_accuracy_from_cm() > best_acc:
    #                 best_acc = eval.get_accuracy_from_cm()
    #                 print best_acc
    #                 if save:
    #                     M.saveModel(model, classifier)
    #             eval.reset()
    #
    #     else:
    #         model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batchSize)
    #         if save:
    #             M.saveModel(model, classifier)
    #
    # # create same, but stateful model and get the weights from the trained model
    # sfModel = M.makeModelFromParam(inpSize, nLabels, arch, DOValue, lr, init, activ, stateful=True, sliceSize=1)
    # sfModel = M._cloneWeights(model, sfModel)
    # sfModel = M.compileModel(sfModel)

    # evaluation of the model
    print('Evaluating the model...')
    eval.evaluate_custom(model, X_test, y_test)
    print('Resulting ustom accuracy: ', eval.get_accuracy_from_cm())
    print(eval.get_cm())

    if analyse:
        plotAnalysis(sfModel, model, X_test, X_test_buf, y_test)


def test_model(test_idx, model_name, classifier=None, data=None, plot=False):
    # TODO: test, useful at all?
    if data is None:
        d = labl_data_helper.Data(classifier)

    model = model_helper.ModelHelper().loadModel(model_name)

    data_list = common.list_take(d.data_list, test_idx)
    targets = common.list_take(d.targets_raw, test_idx)

    plot_predictions(model, data_list, targets)


def grid_search(number, classifier, view=False, folds=5, from_instance=0):
    # TODO: make it log the details about the GS settings as well as th classifier settings automatically
    '''

    Performs grid search on the hyperparameter space (see GS settings section). Models are evaluated using the
    Evaluator class and the intermediate results for each hyperparameter setting are collected on each epoch from report_at_epochs
    to the report variable, averaged over the number of validation folds and stored to a separate result data file.

    Args:
        number: the number of grid search for specification of the results saving
        classifier: classifier name based on whose configuration the dataset should be loaded
        view: view the data before actual gridsearch
        folds: number of validation folds for the grid search
        from_instance: instance of grid search to begin from in the case that gs is not performed in one go

    '''

    # ----------------------------------------------------------        Dataset preparation

    d = data_loader(classifier, view=view)

    # load own classes
    ldh = labl_data_helper.LabeledDataHelper()

    # prepare the cross-validation folds
    np.random.seed(1337)
    nr_folds = float(folds)
    if nr_folds > 1:
        nr_folds = int(nr_folds)
        skf = StratifiedKFold(d.targets_raw, n_folds=nr_folds, shuffle=True)
    else:
        skf = ldh.getRandomTrainTestIndexes(d.targets_raw, nr_folds)
        print(skf)

    # ----------------------------------------------------------        GS settings

    # search space settings
    nHiddenLayers = [2]
    layerSizes = [6, 10, 14, 18, 24, 28, 34, 40]

    architectures = build_architectures(nHiddenLayers, layerSizes)
    architectures = [
                     [16],
                     [24],
                     [8, 8],
                     [14, 10],
                     [18, 12],
                     [10, 10, 10],
                     [20, 16, 10],
                     [22, 12, 8],
                     ]

    drop_out_val = [0.2]
    lrs = [0.01, 0.015, 0.02, 0.03]

    all_inits = ['normal', 'lecun_uniform', 'glorot_normal', 'he_uniform', 'he_normal']  # *
    inits = [all_inits[2], all_inits[-1]]

    all_activs = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid']    # *
    activs = [all_activs[1]]

    momenta = [0]       # not used at the time
    batch_sizs = [20, 32, 48]

    # reporting settings
    report_at_epochs = [4, 6, 9, 12, 15, 19, 23, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150]
    report_at_epochs = list(range(60))[1:]

    # * see http://keras.io/
    # ----------------------------------------------------------        GS search

    # initialize all parameters needed
    train_for_epochs = np.diff([0] + report_at_epochs)
    instances = len(lrs)*len(inits)*len(drop_out_val)*len(architectures)*len(batch_sizs)*len(momenta)*len(activs)
    instance = 1

    mh = model_helper.ModelHelper()

    start_time = time.time()

    # main search space loop
    for lr in lrs:
        for init in inits:
            for DOValue in drop_out_val:
                for momentum in momenta:
                    for activ in activs:
                        for architecture in architectures:
                            for batch in batch_sizs:

                                print('Instance ' + str(instance) + '/' + str(instances))

                                # initialize report details
                                report = [[DOValue, architecture, batch, lr, init, activ, momentum]]
                                print('[DOValue, architecture, batch, lr, init, activ, momentum]')
                                print(report)

                                if instance < from_instance:
                                    print('Skipping instance...')
                                    instance += 1
                                    continue

                                evaluator = Evaluator(d.nr_labels, batch, verbose=0)

                                # init metrics
                                #         [ loss, accuracy, custom Accuracy, confusion matrix ]     TODO: make a class
                                metrics = [[0, 0, 0, np.zeros((d.nr_labels, d.nr_labels))]]*len(train_for_epochs)

                                # depending on whether cross-validation is selected perform corresponding evaluation
                                for train_idx, val_idx in skf:

                                    # rebuild the build the model
                                    model = mh.makeModelFromParam(d.nr_features, d.nr_labels, architecture, DOValue,
                                                                  lr, init, activ, momentum)

                                    X_train = d.data_padded_np.take(train_idx, axis=0)
                                    y_train = d.targets_padded_np.take(train_idx, axis=0)
                                    X_val_pad = d.data_padded_np.take(val_idx, axis=0)
                                    y_val_pad = d.targets_padded_np.take(val_idx, axis=0)
                                    X_val = common.list_take(d.data_list, val_idx)
                                    y_val = common.list_take(d.targets_list, val_idx)

                                    metric_collector = []

                                    for epochs, epochBatch in zip(report_at_epochs, train_for_epochs):

                                        # train the model
                                        model.fit(X_train, y_train,
                                                  nb_epoch=epochBatch,
                                                  verbose=0,
                                                  batch_size=batch)

                                        # update metrics
                                        metric_collector.append(
                                            evaluator.evaluate_complete_once(model,
                                                                             X_val_pad, y_val_pad,
                                                                             x_orig=X_val, y_orig=y_val))

                                    # add metrics
                                    for item, i in zip(metrics, list(range(len(metrics)))):
                                        metrics[i] = list(map(add, metrics[i], metric_collector[i]))

                                # average the results
                                if nr_folds > 1:
                                    metrics = [[item / nr_folds for item in itemaAtEpoch]
                                               for itemaAtEpoch in metrics]

                                # add epoch info for report
                                for item, i in zip(metrics, list(range(len(metrics)))):
                                    metrics[i] = [report_at_epochs[i]] + item
                                print(metrics)

                                # report the progress
                                report = report + metrics

                                # write results to file
                                name = config.settings.results.gs_path + '/' + str(number) + '/' +\
                                       str(instance) + '_of_' + str(instances) + '.txt'
                                f = open(name, "wb+")
                                pickle.dump(report, f)
                                f.close()

                                instance += 1

    end = time.time()
    print('')
    print(' :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) ')
    print('GS finished after only ' + str(end - start_time) + ' seconds.')
    print(' :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) :-) ')

# ----------------------------------------------------------------------------------------------- keras helper functions


def plot_predictions(model, x, y, title=None, nr_targets=None):
    results = []
    for sequence in tqdm(x):
        history = model.predict(sequence[np.newaxis, :])
        history = np.squeeze(history)

        results.append(history)

    if nr_targets:
        ev = Evaluator(nr_targets)
    else:
        ev = Evaluator(y[0].shape[1])

    ev.evalPlotter(len(x), y, [results], [title], nr_classes=nr_targets)


def plotAnalysis(sfModel, model, X_test, X_test_buf, y_test):

    # model testing
    print('Testing the stateful model...')
    sequenceResults = []
    sfResults = []
    for sequence in tqdm(list(range(len(X_test)))):

        # sfModel.reset_states()
        for tStep in range(X_test[sequence].shape[0]):

            history = sfModel.predict_on_batch(X_test[np.newaxis, np.newaxis, sequence, tStep])

            if tStep == 0:
                sequenceResults = np.squeeze(history)
            else:
                sequenceResults = np.vstack((sequenceResults, np.squeeze(history)))

        sfResults.append(sequenceResults)
        sfModel.reset_states()

    # evaluate_custom model
    print('Testing the model with padded data...')
    sequenceResults = []
    results = []
    for sequence in tqdm(list(range(len(X_test)))):

        history = model.predict(X_test[np.newaxis, sequence])
        history = np.squeeze(history)

        results.append(history)

    # evaluate_custom model
    print('Testing the model with buf data...')
    sequenceResults = []
    results_nopad = []
    for sequence in tqdm(list(range(len(X_test)))):

        history = model.predict(X_test_buf[sequence][np.newaxis])
        history = np.squeeze(history)

        results_nopad.append(history)

    E = Evaluator(y_test.shape[0])
    E.evalPlotter(X_test.shape[0], y_test, [sfResults, results, results_nopad], ['stateful', 'padded', 'no padding'])


########################################################################################################################
# main -----------------------------------------------------------------------------------------------------------------
########################################################################################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-p', '--program', help='Program/function to run ', required=True, default='test')
    parser.add_argument('-c', '--classifier', help='Classifier name', required=False, default=False)
    parser.add_argument('-s', '--save', help='Save the model or not', required=False, action='store_true')
    parser.add_argument('-l', '--load', help='Load the existing model for given classifier', required=False,
                        action='store_true')
    parser.add_argument('-a', '--analyse', help='Analyse all the models - stateful, stateless, ...', required=False,
                        action='store_true')
    parser.add_argument('-b', '--best', help='Epoch by epoch saves the best, ...', required=False, action='store_true')
    parser.add_argument('-plt', '--plot', help='Enable plotting of various steps in the programs', required=False,
                        action='store_true')
    parser.add_argument('-f', '--folds', help='Number of folds for cross-validation', required=False, default=5)
    parser.add_argument('--test_ratio', help='Ratio between test/train data', required=False, default=0.2)
    parser.add_argument('-m', '--metric', help='Metric to sort results - 2 for keras accuracy, 3 for custom accuracy',
                        required=False, default=3)
    parser.add_argument('-w', '--view', help='View the loaded dataset or not', required=False, action='store_true')
    parser.add_argument('-nr', '--number', help='search number', required=False)
    parser.add_argument('--save_scalers', help='Save scalers when viewing the data', required=False, action='store_true')
    parser.add_argument('-from', '--from_instance', help='Instance nr to begin the grid search with', required=False,
                        default=0)

    args = parser.parse_args()

    if args.program == 'skM':
        skl_multi_classif(args.classifier)

    elif args.program == 'train':
        if args.classifier:
            keras_recurrent_train(args.classifier,
                                  view=args.view,
                                  test_ratio=float(args.test_ratio),
                                  folds=int(args.folds))
        else:
            print('needs more args')
            parser.print_usage()

    elif args.program == 'debug':
        if args.classifier:
            keras_recurrent_debug(args.classifier,
                                  save=args.save,
                                  load=args.load,
                                  plot=args.plot,
                                  analyse=args.analyse,
                                  best=args.best)
        else:
            print('needs more args')
            parser.print_usage()

    elif args.program == 'view':

        data_loader(args.classifier, view=True, save_scalers=args.save_scalers)

    elif args.program == 'readgs':
        read_gs_results(int(args.number), metric=int(args.metric))

    elif args.program == 'gs':
        if args.number is False:
            raise ValueError('Number argument needed!')
        grid_search(int(args.number), args.classifier, view=args.view, folds=args.folds,
                    from_instance=int(args.from_instance))

    else:
        parser.print_help()
        print('')
        print('See the main function for more details about the arguments.')
