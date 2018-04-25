
# custom modules
from config import config
from classes.helpers import common


class ClassifierSettings(object):
    # TODO: cleanup, move the functions like read_classifier_settings inside the class
    def __init__(self, classifier):

        self.classifier_name = classifier

        # TODO: could not it be done more conventionally?
        setting_parameters = ['feature_list',
                              'current_feature_list',
                              'parts_to_normalize',
                              'label_list',
                              'limit_instances',
                              'buffer_length',
                              'dead_steps',
                              'running_window',
                              'prediction_threshold',
                              'persons',
                              'architecture',
                              'dropout_value',
                              'learning_rate',
                              'init',
                              'batch_size',
                              'epochs',
                              'activ',
                              'normalize',
                              'min_max_scale',
                              'normalize_body_parts',
                              'add_derivatives',
                              'add_step_indexes',
                              'apply_pca',
                              'shoulder_orientation',
                              'sequence_trim',
                              'padding_type',
                              'shoulder_orientation',
                              'slice_size',
                              'threshold',
                              'smallest_predict_buf',
                              ]

        config_settings = read_classifier_conf(self.classifier_name)

        for param in setting_parameters:
            try:
                param_value = config_settings[param]
            except KeyError:
                param_value = None
                # print 'does not contain the parameter ', param

            # HANDLE ANNOYING SPECIAL CASES
            if param == 'label_list' and param_value is not None:
                source_label_list = get_source_label_list(param_value)
                setattr(self, 'source_label_list', source_label_list)

                label_list = get_label_list(param_value)
                setattr(self, 'label_list', label_list)

                dict_to_merge = get_dict_to_merge(param_value)
                setattr(self, 'dict_to_merge', dict_to_merge)

                continue

            setattr(self, param, param_value)

    def view_settings(self):
        print ''
        print self.classifier_name, ' classifier parameters:'
        print '----------------------------------------------------'
        p = [a for a in dir(self) if not a.startswith('__')]
        for att in p:
            if att is not None:
                print
                print att, '::'
                print getattr(self, att)


def get_classifier_setting(name, parameter):
    file_name = "%s%s%s%s%s" % (config.Config().root_dir, config.settings.configs.classifiers_path, '/', name, '.py')
    classif_configuration = {}
    try:
        execfile(file_name, classif_configuration)
    except IOError, e:
        print e
        print "Error: Problem finding configuration file for classifier %s" % name
        return False
    return classif_configuration[parameter]


def read_classifier_conf(classifier):

    f_name = "%s%s%s%s%s" % (config.Config().root_dir, config.settings.configs.classifiers_path, '/', classifier, '.py')
    classif_configuration = {}
    execfile(f_name, classif_configuration)
    return classif_configuration


def get_source_label_list(label_struct):
    # returns the labels of the original dataset
    source_list = []
    for item in label_struct:
        if not isinstance(item, list):
            source_list.append(item)
        else:
            for subitem in item[1]:
                source_list.append(subitem)

    return source_list


def get_label_list(label_struct):
    # returns the merged labels
    label_list = []
    for item in label_struct:
        if not isinstance(item, list):
            label_list.append(item)
        else:
            label_list.append(item[0])

    return label_list


def get_dict_to_merge(label_struct):
    # returns a dictionary of labels to merge with the new label as key
    out_dict = {}
    for item in label_struct:
        if isinstance(item, list):
            out_dict[item[0]] = item[1]

    return out_dict


def find_feat_idx(classifier, features):
    feature_list = get_classifier_setting(classifier, 'feature_list')
    return common.match_lists_to_idx(feature_list, features)
