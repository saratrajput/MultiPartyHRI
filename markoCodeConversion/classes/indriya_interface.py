import os
import sys
import time
import zmq
from sys import platform as _platform

from config.transitions import interaction_fsm

if _platform == 'win32':
    dev = os.environ["INDRIYA_ROOT"]
    dir1 = os.path.join(dev, "scripts", "msgs")
    dir2 = os.path.join(dev, "scripts", "experimot_robot_interface")

    sys.path.append(dir1)
    sys.path.append(dir2)

    sys.path.append("C:\Work\Develop\src\github\indriya\src\indriya_msgs\python")

    import engagement_pb2, engagement_state_pb2


class PersonUpdater:
    def __init__(self, name, paramServer):

        # Utils
        import parameter_utils
        self.node = parameter_utils.getNodeParameters(name, paramServer, 1000)

        states = interaction_fsm.states
        print states
        print states[1]
        self.indriya_dict = {
                states[1]: [0, 'person_spotted'],
                states[2]: [1, 'person_shows_attention'],
                states[3]: [2, 'person_approaches'],
                states[4]: [3, 'person_approached'],
                states[5]: [4, 'person_leaves'],
                states[6]: [5, 'person_engaged'],
                states[7]: [6, 'person_not_engaged'],
                states[0]: [7, 'peron_lost'],
                'reengages': [8, 'person_reengages'],
        }

        if self.node is not None:

            pub = parameter_utils.getPublisherInfo(self.node, "EngagementMsgs")
            print 'Indriya publisher: ', pub

            PUB_IP = pub.host.encode('utf-8')
            PUB_PORT = pub.port
            PUB_TOPIC = pub.topic.encode('utf-8')

            context = zmq.Context()

            print 'binding socket on ', PUB_PORT, PUB_IP
            self.socket = context.socket(zmq.PUB)
            self.socket.bind("%s:%s" % (PUB_IP, PUB_PORT))
            self.topic = PUB_TOPIC
            time.sleep(5)
            print 'Pub socket to on topic ', PUB_TOPIC, ' is up!'

            self.sent = 0

        else:
            print 'Indriya not running? Establishing communication failed!'

    def send_current_state(self, current_state):

        if len(current_state) > 0:
            # build the EngagementMsgs structure
            full_msg = engagement_pb2.EngagementMsgs()

            # DEBUG
            # print 'to send: ', current_state

            for person_state in current_state:
                engagement_msg = self.build_single_msg(person_state)
                try:
                    full_msg.engagementMsg.extend([engagement_msg])
                except TypeError:
                    # print 'sending type error!'
                    pass

            # send several times so that it gets transmitted
            for i in xrange(5):
                self.send_full_msg(full_msg)
                time.sleep(0.02)
        else:
            pass
            # print 'empty current state!!'

    def build_single_msg(self, person_state):

        if self.node is not None:

            # print 'abobut to send: ', person_state

            engagement_msg = engagement_pb2.EngagementMsg()
            engagement_state = engagement_state_pb2.EngagementState()

            engagement_state.State = self.indriya_dict[person_state['state']][0]
            engagement_msg.humanId = person_state['id']
            engagement_msg.personNr = int(person_state['person_nr'])
            engagement_msg.state.State = engagement_state.State

            return engagement_msg

    def send_full_msg(self, msg):

        if self.node is not None:

            message = msg.SerializeToString()

            self.socket.send_string(self.topic, zmq.SNDMORE)
            self.socket.send_string(message)

            # DEBUG:
            # print 'full message sent'
            # print self.sent, ' sent, next: ', self.engagement.humanId, self.engagement.state.State











    # TODO: make sure not needed and delete


    # def send_current_state(self, current_state):
    #     # TODO: instead enumerate the protobuf message!
    #     for i in xrange(5):
    #         for person_state in current_state:
    #             self.send_person_state(person_state)
    #             time.sleep(0.02)
    #         time.sleep(0.05)

    # def send_person_state(self, person_state):
    #
    #     if self.node is not None:
    #
    #         # print 'abobut to send: ', person_state
    #
    #         self.engagement_state.State = self.indriya_dict[person_state['state']][0]
    #         self.engagement.humanId = person_state['id']
    #         self.engagement.personNr = int(person_state['person_nr'])
    #         self.engagement.state.State = self.engagement_state.State
    #
    #         message = self.engagement.SerializeToString()
    #
    #         # for a in range(5):
    #         #     self.socket.send_string(self.topic, zmq.SNDMORE)
    #         #     self.socket.send_string(message)
    #         #     time.sleep(0.1)
    #         #     print self.sent, ' sent, next: ', self.engagement.humanId, self.engagement.state.State
    #         #     self.sent += 1
    #
    #         self.socket.send_string(self.topic, zmq.SNDMORE)
    #         self.socket.send_string(message)
    #         # time.sleep(0.1)
    #
    #         print self.sent, ' sent, next: ', self.engagement.humanId, self.engagement.state.State
    #         # self.sent += 1
    #
    #     else:
    #         # print 'Could not send info to Indriya, no communication established!'
    #         pass


class CoupleUpdater:
    def __init__(self, name, paramServer):
        pass
