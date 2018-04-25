import pytz
from datetime import datetime
from sys import platform as _platform
import numpy as np

# external modules
#from external.transitions.extensions import MachineFactory
from transitions.extensions import MachineFactory
#from external.transitions import logger
from transitions import logger
import logging
logger.setLevel(logging.WARNING)

# custom modules
from .helpers import common, kinect_data_helper
from config import config
from classes.classifiers import rnn, proximity, touch, speech, tracking, engagement, presence, still
from config.transitions import interaction_fsm as fsm_definition
# from config.transitions.unit_tests import presence_test_fsm as fsm_definition
# from config.transitions.unit_tests import proximity_test_fsm as fsm_definition
# from config.transitions.unit_tests import still_test_fsm as fsm_definition
# from config.transitions.unit_tests import speaking_test_fsm as fsm_definition
# from config.transitions.unit_tests import touch_test_fsm as fsm_definition
# from config.transitions.unit_tests import leaving_test_fsm as fsm_definition
# from config.transitions.unit_tests import engagement_test_fsm as fsm_definition


class Person:
    def __init__(self, person_id, prt_lock, update_q, graph_logging_on=False):

        # initialize details
        self.id = person_id
        self.prt_lock = prt_lock
        self.classifiers, self.state_classifiers = self.initialize_classifiers()
        self.update_q = update_q
        self.lost_track = 0
        self.lost_at = 0

        self.frame_idx = kinect_data_helper.KinectDataHelper(version=3).get_features_indexes(['FrameNr'])
        self.frame_nr = 2

        if _platform == "linux" or _platform == "linux2":

            if graph_logging_on:
                self.log_path = "%s%s" % (config.Config().root_dir, config.settings.log.graph_path)

                # add the graphical logger after all transitions
                self._graph_logger_ct = 0
                self.root_path = config.Config().root_dir
                for transition in fsm_definition.state_transitions:
                    if 'after' in transition:
                        transition['after'] = common.listify(transition['after']) + ['log']
                    else:
                        transition['after'] = '_graph_log'

            # initialize the machine with its configuration
            machine = MachineFactory.get_predefined(graph=True, nested=True)
            self.machine = machine(model=self,
                                   states=fsm_definition.person_states,
                                   transitions=fsm_definition.state_transitions,
                                   auto_transitions=False,
                                   initial='not-tracked_',
                                   ignore_invalid_triggers=False,  # TODO: to avoid exceptinos misusage??
                                   title="Engagement State Machine",
                                   send_event=True,
                                   show_conditions=False,
                                   )

            if graph_logging_on:
                self._graph_log(None)

        else:
            # TODO: make it accept nested states
            machine = MachineFactory.get_predefined(nested=True)
            self.machine = machine(model=self,
                                   states=fsm_definition.person_states,
                                   transitions=fsm_definition.state_transitions,
                                   auto_transitions=False,
                                   initial='not-tracked_',
                                   ignore_invalid_triggers=False,  # TODO: to avoid exceptinos misusage??
                                   send_event=True,
                                   )

            if graph_logging_on:
                print 'Graph logging will not work on Windows platform due to graphviz bug!'

        # register the classifier buffer cleanup on state changes
        for state in self.machine.states:
            if '_' in state:
                getattr(self.machine, 'on_enter_' + state)('clean_old_buffers')
                getattr(self.machine, 'on_enter_' + state)('report_state_change')

        # make sure everything is clean when starting a new tracking
        self.transitions = self.initialize_transitions()

        # DEBUG printing
        if person_id == 0:
            self.id_str = '0          '
        elif person_id == 1:
            self.id_str = '  1        '
        elif person_id == 2:
            self.id_str = '    2      '
        elif person_id == 3:
            self.id_str = '      3    '
        elif person_id == 4:
            self.id_str = '        4  '
        elif person_id == 5:
            self.id_str = '          5'

    # transitions function
    def init_buffers(self, event):
        data = event.kwargs.get('data', False)
        if data:
            print 'initializing buffers for person ', self.id
        self.clean_all_buffers(None)

    # graphing
    def _graph_log(self, event):
        f_name = "%s%s%s_%d_%d.png" % (self.log_path, '/', chr(65+self.id), self.id, self._graph_logger_ct)
        self.machine.graph.draw(f_name, prog='dot')
        self._graph_logger_ct += 1

    def initialize_classifiers(self):

        # kinda container...
        classifiers = {
                       'preInt': rnn.Rnn('preInt'),
                       'proximity': proximity.Proximity('proximity'),
                       'touch': touch.Touch('touch'),
                       'speech': speech.Speech('speech'),
                       'tracking': tracking.Tracking('tracking'),
                       'postEngag': rnn.Rnn('postEngag'),
                       'engagement': engagement.Engagement('engagement'),
                       'presence': presence.Presence('presence'),
                       'still': still.Still('still'),
                       }

        # check if all classifiers were initialized
        state_classifiers = self._resolve_state_classifiers(classifiers)
        all_classifiers = []
        for state in state_classifiers:
            all_classifiers += state_classifiers[state]

        if all(key in classifiers for key in list(set(all_classifiers))):
            return classifiers, state_classifiers
        raise ValueError('Every classifier from transitions configuration has to be initialized!')

    def initialize_transitions(self):
        transitions = []
        methods = [method for method in dir(self) if callable(getattr(self, method))]
        for method in methods:
            if method.startswith('T_'):
                transitions.append(method)
        return transitions

    def update_active_classifiers(self, dat):
        if dat is not None:
            try:
                self.frame_nr = np.asarray(dat).take(self.frame_idx, axis=0)
            except:
                pass

            for classifier_name in self.state_classifiers[self.state]:
                # TODO: find why dat is sometimes nan!!
                self.classifiers[classifier_name].update_data(np.nan_to_num(dat))
                self.classifiers[classifier_name].update_decision()

                # DEBUG
                self.classifiers[classifier_name].log_decision_change(self.id, self.frame_nr, self.classifiers['tracking'].counter)


            # DEBUG:
            # print 'pers ', self.id, 'got data!'
        else:
            # TODO: convert from special case to a normal case
            # print self.id, ' lost at ', self.classifiers['tracking'].counter
            self.lost_at = self.classifiers['tracking'].counter
            self.classifiers['tracking'].clean_buffer()
            self.classifiers['tracking'].update_decision()

            # print 'No valid data for update'

    def decisions_agree(self, args):

        # DEBUG:
        # print 'ST: ', self.state, ' cond: ', sorted(args), 'decis: ', sorted(
        #     [getattr(self.classifiers[classifier], 'get_decision')() for classifier in
        #      self.state_classifiers[self.state]]), ' :: ', sorted(args) == \
        #       sorted([getattr(self.classifiers[classifier], 'get_decision')() for classifier in
        #       self.state_classifiers[self.state]])

        return set(args).issubset(\
               set([getattr(self.classifiers[classifier], 'get_decision')() for classifier in
                    self.state_classifiers[self.state]]))

    def clean_all_buffers(self, args):
        for classifier_name in self.classifiers:
            self.classifiers[classifier_name].clean_buffer()

    def clean_old_buffers(self, args):
        # TODO: make generic, with clear_buffers_for(...)
        # print self.state_classifiers
        # cleans the buffers of classifiers that are not needed in the current state
        class_to_clean = [classifier for classifier in self.classifiers
                          if classifier not in self.state_classifiers[self.state]]

        # print 'entering', self.state, 'to clean:', class_to_clean

        for classifier_name in class_to_clean:
            self.classifiers[classifier_name].clean_buffer()

        # # DEBUG:
        # print 'p ,', self.id, 'after clean: ', self.state
        # for classifier_name in class_to_clean:
        #     try:
        #         print self.classifiers[classifier_name].buffer.shape
        #     except AttributeError:
        #         print 'buf clean'

    def report_state_change(self, args):
        # DEBUG:
        # try:
        #     frame = str(int(self.frame_nr)).zfill(6)
        # except:
        #     frame = '0'

        if self.state == 'not-tracked_':
            ctr = self.lost_at
        else:
            ctr = self.classifiers['tracking'].counter

        self.prt_lock.acquire()
        print 'F|', str(int(self.frame_nr)).zfill(6), '|', str(ctr).zfill(4), '\t',\
            '::', self.id_str, '::', '--->', self.state
        self.prt_lock.release()
            # '@ FRAME::', frame,\
            # '@', datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],\


        person_update = {'id': self.id, 'state': self.state, 'frame': self.frame_nr,
                         'step': self.classifiers['tracking'].counter}
        self.update_q.put(person_update)

    def _resolve_state_classifiers(self, classifiers):
        decision_dict = {}
        # load decisions of all classifiers into dict {decision:classifier}
        for classifier in classifiers:
            for label in classifiers[classifier].label_list:
                decision_dict[label] = classifiers[classifier].name

        # print decision_dict

        # add a classifier name to each of the states, based on conditions using from decision_dict
        state_classifiers = dict.fromkeys(fsm_definition.states)

        # print state_classifiers
        for transition in fsm_definition.state_transitions:
            transition_states = common.listify(transition['source'])
            for transition_state in transition_states:
                if '_' not in transition_state:
                    states = []
                    for st in fsm_definition.states:
                        if st.startswith(transition_state):
                            states.append(st)
                else:
                    states = common.listify(transition_state)

                for state in states:
                    for condition in transition['conditions']['decisions_agree']:
                        classifier_name = decision_dict[condition]

                        # assign the classifier names to each state
                        if state_classifiers[state]:
                            if not classifier_name in state_classifiers[state]:
                                state_classifiers[state].append(classifier_name)
                        else:
                            state_classifiers[state] = [classifier_name]

        # print state_classifiers
        return state_classifiers
