# global imports
import sys
import logging
import numpy as np
import pytz
from datetime import datetime

# custom modules
from classes.helpers import common
from config import config
from config.transitions import interaction_fsm


class Interperson_analyzer:
    def __init__(self):
        logging.basicConfig(filename=config.Config().root_dir + config.settings.log.path + '/state_log.log',
                            level=logging.DEBUG)

        self.state_prio_dict = interaction_fsm.state_priorities
        self.state_dict = common.dictify(interaction_fsm.states, inv=True)
        self.state_dict_out = common.dictify(interaction_fsm.states)
        parameters = ['state_prio', 'id', 'state']
        self.state_param_dict = {k: v for (k, v) in zip(parameters, range(len(parameters)))}

        # initialize the global state
        self.state = np.asarray([[0] * 6]*len(parameters)).T
        self.state[:, 1] = np.arange(6)

        # LOG
        logging.info('@' + datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] +
                     '------------------- initialized')

    def update_state(self, update):

        # updates and reorders the entries in the self.state based on the highest state priority
        print 'received ', update

        # update the state
        for person_nr in range(6):
            if self.state[person_nr, 1] == update['id']:
                try:
                    self.state[person_nr, 0] = self.state_prio_dict[update['state']]
                except KeyError:
                    print 'State to update does not correspond to Indriya - compatible states.' \
                          ' Interperson processor ends execution. Ignore the error in case of testing.'
                    sys.exit()

                # DEBUG
                # print update['state']
                # print self.state

                self.state[person_nr, 2] = self.state_dict[update['state']]
                continue

        # reorder persons based on prios
        # TODO: don't be lazy to find more elegant way to sort without keeping top prio without switching!
        self.state = self.state[self.state[:, 0].argsort()[::-1][:6]]
        self.state = self.state[self.state[:, 0].argsort()[::-1][:6]]

        print 'after update of ', update['id']
        print self.state

        # print '-updating_stateholder_state-'

        # LOG # TODO: deal with no frame more elegantly
        try:
            frame = str(int(update['frame'])).zfill(6)
        except:
            frame = '0'
        logging.info(str(update['step']) + '\t'
                     '|FR|' + frame +
                     ':: ' + str(update['id']) + ' to state: ' + update['state'] + ' ::' +
                     '@' + datetime.now(tz=pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                     )

    def get_persons_parameter(self, ids, param):

        ids = common.listify(ids)

        # TODO: make a generic dictionary for all parameters to automatize this
        if param == 'person_nr':
            # DEBUG
            # print 'id ', ids, ' person number: ', self.get_person_number(ids)
            return self.get_person_number(ids)
        else:
            params = []
            for id in ids:
                idx = np.where(self.state[:, self.state_param_dict['id']] == id)[0][0]
                params.append(self.state[idx, self.state_param_dict[param]])

        if param == 'state':
            params = [self.state_dict_out[item] for item in params]

        return params

    def get_person_number(self, ids):
        nrs = []
        ids = common.listify(ids)
        for id in ids:
            # print id, ':', np.where(self.state[:, 1] == id)[0][0]
            nrs.append(np.where(self.state[:, 1] == id)[0][0])
        return [x + 1 for x in nrs]

    def get_tracked_params(self, parameters):
        tracked_idx = np.where(self.state[:, 0] > 0)[0]
        # print self.state
        tracked_ids = self.state[:, 1].take(tracked_idx).tolist()

        params = []
        for person in range(len(tracked_ids)):
            params.append({'id': tracked_ids[person]})

        for param in parameters:
            persons_parameter = self.get_persons_parameter(tracked_ids, param)
            for person in range(len(params)):
                params[person][param] = persons_parameter[person]

        return params

    def print_state(self):
        print self.state

    # def get_tracked_ids_numbers(self):
    #     idx = np.where(self.state[:, 0] > 0)[0]
    #     ids = self.state[:, 1].take(idx).tolist()
    #     return ids, self.get_person_number(ids)
