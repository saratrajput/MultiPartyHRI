
# general imports
import numpy as np
import sys
import logging
from collections import deque
import google.protobuf.internal.decoder as decoder
import google.protobuf.internal.encoder as encoder

# custom modules
from classes.helpers import common, math_helper
from config import dataConfig, config


class KinectDataHelper:
    # TODO: instead of versioning make the message reading based on something like this:
    # def dump_object(obj):
    #     for descriptor in obj.DESCRIPTOR.fields:
    #         value = getattr(obj, descriptor.name)
    #         if descriptor.type == descriptor.TYPE_MESSAGE:
    #             if descriptor.label == descriptor.LABEL_REPEATED:
    #                 map(dump_object, value)
    #             else:
    #                 dump_object(value)
    #         elif descriptor.type == descriptor.TYPE_ENUM:
    #             enum_name = descriptor.enum_type.values[value].name
    #             print "%s: %s" % (descriptor.full_name, enum_name)
    #         else:
    #             print "%s: %s" % (descriptor.full_name, value)

    def __init__(self, version=3):
        self.version = version
        self.initFPS()      # outdated, not used really
        self.data_dict = self.get_data_dict()
        self.cubePose = None    # [0, 0, 0, 0, 0, 0, 0]
        self.message_type = None

        # for lengths see protobuf messages
        if version == 3 or version == 4:
            self.msg_len = 67
        if version == 1 or version == 2:
            self.msg_len = 116
        elif version == 0:
            self.msg_len = 114

        # logging.basicConfig(filename=config.config().root_dir + config.settings.log.root + '/data_log.log',
        #     level=logging.DEBUG)

    def load_message_type(self):

        import imp
        cfg = config.Config()
        full_file = cfg.root_dir + '/config/msg/human_engagement_pb2.py'
        human_engagement_pb2 = imp.load_source('example', full_file)
        self.message_type = human_engagement_pb2.KinectEngagStreams()

        return self.message_type

    def get_msg_dict(self):
        return self.data_dict

    def time_arr_from_msg(self, msg):
        self.updateFPS(msg.KinectStream[0].CaptureTimeMs)
        # print [str(msg.KinectStream[0].CaptureTime), msg.KinectStream[0].CaptureTimeMs, self.getFPS()]
        return [str(msg.KinectStream[0].CaptureTime), msg.KinectStream[0].CaptureTimeMs, self.getFPS()]

    # takes raw message, normalizes Kinect vectors, returns the data and ID separately
    def msg_to_person_data_d(self, msg):

        out_dict = {}
        capture_time = msg.KinectStream[0].CaptureTimeMs

        if self.version == 4:
            # version for reading old recorded files (remember, there will be no head orientation and Engagement!)

            for KS in msg.KinectStream:
                tmp = np.empty(self.msg_len)
                i = 0

                # feed all into tmp
                tmp[i] = capture_time
                i += 1
                for JT in KS.Joints:
                    # get the positions
                    vector = [JT.Position.x, JT.Position.y, JT.Position.z]
                    transformed_vect = self.vect_to_frame(vector, self.cubePose)

                    # fill the positions
                    tmp[i] = transformed_vect[0]
                    i += 1
                    tmp[i] = transformed_vect[1]
                    i += 1
                    tmp[i] = transformed_vect[2]
                    i += 1
                    tmp[i] = JT.State
                    i += 1
                    if JT.Type == 3:
                        # in face frame
                        face_z = [0, 0, 1, 1]

                        # in kinect fram
                        face_q = [JT.Orientation.w,
                                  JT.Orientation.x,
                                  JT.Orientation.y,
                                  JT.Orientation.z]
                        rm = math_helper.quat_to_rot_matrix(face_q)
                        face_z = np.dot(rm, face_z)

                        # in cube frame
                        cube_q = self.cubePose[3:7]
                        rm = math_helper.quat_to_rot_matrix(cube_q)
                        rm = rm.transpose()  # because this is the way it arrives from alvar
                        face_z = np.dot(rm, face_z)

                        tmp[i] = face_z[0]
                        i += 1
                        tmp[i] = face_z[1]
                        i += 1
                        tmp[i] = face_z[2]
                        i += 1

                # for compatibility, insert Engagement = 0 in front of other recorded features
                # if there are no FP, insert 6x zeros (see nr of face properties in protobuf)
                fp = 0
                for FP in KS.FaceProperties:
                    fp += 1
                    if fp == 1:
                        tmp[i] = 0
                        i += 1
                    # print FP.Type
                    tmp[i] = FP.Result
                    i += 1
                if fp == 0:
                    for fill in range(6):
                        tmp[i] = 0
                        i += 1

                tmp[i] = KS.FaceOrientation.Y
                i += 1
                tmp[i] = KS.FaceOrientation.P
                i += 1
                tmp[i] = KS.FaceOrientation.R
                i += 1
                tmp[i] = KS.SpeakingConfidence
                i += 1
                tmp[i] = KS.FrameNr

                out_dict[KS.TrackingId] = tmp.tolist()

        elif self.version == 3:

            for KS in msg.KinectStream:
                tmp = np.empty(self.msg_len)
                i = 0

                # feed all into tmp
                tmp[i] = capture_time    # ----------------------!!!!!!!!!
                i += 1
                for JT in KS.Joints:
                    # get the positions
                    vector = [JT.Position.x, JT.Position.y, JT.Position.z]
                    transformed_vect = self.vect_to_frame(vector, self.cubePose)

                    # fill the positions
                    tmp[i] = transformed_vect[0]
                    i += 1
                    tmp[i] = transformed_vect[1]
                    i += 1
                    tmp[i] = transformed_vect[2]
                    i += 1
                    tmp[i] = JT.State
                    i += 1
                    if JT.Type == 3:

                        # in face frame
                        face_z = [0, 0, 1, 1]

                        # in kinect fram
                        face_q = [JT.Orientation.w,
                                  JT.Orientation.x,
                                  JT.Orientation.y,
                                  JT.Orientation.z]
                        rm = math_helper.quat_to_rot_matrix(face_q)
                        face_z = np.dot(rm, face_z)

                        # in cube frame
                        cube_q = self.cubePose[3:7]
                        rm = math_helper.quat_to_rot_matrix(cube_q)
                        rm = rm.transpose()     # because this is the way it arrives from alvar
                        face_z = np.dot(rm, face_z)

                        tmp[i] = face_z[0]
                        i += 1
                        tmp[i] = face_z[1]
                        i += 1
                        tmp[i] = face_z[2]
                        i += 1

                # if the received message was not fully init, the face results are most probably missing => set to sth.
                if msg.IsInitialized():
                    for FP in KS.FaceProperties:
                        # print FP.Type
                        tmp[i] = FP.Result
                        i += 1
                else:
                    # print 'face_missing'
                    for fill in range(6):
                        tmp[i] = 0
                        i += 1

                tmp[i] = KS.FaceOrientation.Y
                i += 1
                tmp[i] = KS.FaceOrientation.P
                i += 1
                tmp[i] = KS.FaceOrientation.R
                i += 1
                tmp[i] = KS.SpeakingConfidence
                i += 1
                tmp[i] = KS.FrameNr

                out_dict[KS.TrackingId] = tmp.tolist()

        elif self.version == 2:

            for KS in msg.KinectStream:
                tmp = np.empty(self.msg_len)
                i = 0

                tmp[i] = capture_time
                i += 1
                for JT in KS.Joints:
                    # get the positions
                    vector = [JT.Position.x, JT.Position.y, JT.Position.z]
                    transformed_vect = self.vect_to_frame(vector, self.cubePose)

                    # fill the positions
                    tmp[i] = transformed_vect[0]
                    i += 1
                    tmp[i] = transformed_vect[1]
                    i += 1
                    tmp[i] = transformed_vect[2]
                    i += 1

                    # fill the orientations
                    tmp[i] = JT.Orientation.x
                    i += 1
                    tmp[i] = JT.Orientation.y
                    i += 1
                    tmp[i] = JT.Orientation.z
                    i += 1
                    tmp[i] = JT.Orientation.w
                    i += 1
                    tmp[i] = JT.State
                    i += 1
                if msg.IsInitialized():
                    for FP in KS.FaceProperties:
                        # print FP.Type
                        tmp[i] = FP.Result
                        i += 1
                else:
                    # print 'face_missing'
                    for fill in range(6):
                        tmp[i] = 0
                        i += 1
                tmp[i] = KS.FaceOrientation.Y
                i += 1
                tmp[i] = KS.FaceOrientation.P
                i += 1
                tmp[i] = KS.FaceOrientation.R
                i += 1
                tmp[i] = KS.SpeakingConfidence
                i += 1
                # for compatibility with older versions, otherwise v3 should read all versions
                try:
                    tmp[i] = KS.FrameNr
                except:
                    tmp[i] = 0

                out_dict[KS.TrackingId] = tmp.tolist()

        if self.version == 1:
            for KS in msg.KinectStream:
                tmp = np.empty(self.msg_len)
                i = 0

                tmp[i] = capture_time
                i += 1
                for JT in KS.Joints:
                    # get the positions
                    vector = [JT.Position.x, JT.Position.y, JT.Position.z]
                    transformed_vect = self.vect_to_frame(vector, self.cubePose)

                    # fill the positions
                    tmp[i] = transformed_vect[0]
                    i += 1
                    tmp[i] = transformed_vect[1]
                    i += 1
                    tmp[i] = transformed_vect[2]
                    i += 1

                    # fill the orientations
                    tmp[i] = JT.Orientation.x
                    i += 1
                    tmp[i] = JT.Orientation.y
                    i += 1
                    tmp[i] = JT.Orientation.z
                    i += 1
                    tmp[i] = JT.Orientation.w
                    i += 1
                    tmp[i] = JT.State
                    i += 1
                for FP in KS.FaceProperties:
                    tmp[i] = FP.Result
                    i += 1
                tmp[i] = KS.FaceOrientation.Y
                i += 1
                tmp[i] = KS.FaceOrientation.P
                i += 1
                tmp[i] = KS.FaceOrientation.R
                i += 1
                tmp[i] = KS.SpeakingConfidence
                i += 1
                tmp[i] = KS.FrameNr

                out_dict[KS.TrackingId] = tmp.tolist()

        elif self.version == 0:

            for KS in msg.KinectStream:
                tmp = np.empty(self.msg_len)
                i = 0

                # feed all into tmp
                tmp[i] = capture_time
                i += 1
                for JT in KS.Joints:
                    # get the positions
                    vector = [JT.Position.x, JT.Position.y, JT.Position.z]
                    transformed_vect = self.vect_to_frame(vector, self.cubePose)

                    # fill the positions
                    tmp[i] = transformed_vect[0]
                    i += 1
                    tmp[i] = transformed_vect[1]
                    i += 1
                    tmp[i] = transformed_vect[2]
                    i += 1

                    # fill the orientations
                    tmp[i] = JT.Orientation.x
                    i += 1
                    tmp[i] = JT.Orientation.y
                    i += 1
                    tmp[i] = JT.Orientation.z
                    i += 1
                    tmp[i] = JT.Orientation.w
                    i += 1
                    tmp[i] = JT.State
                    i += 1
                for FP in KS.FaceProperties:
                    tmp[i] = FP.Result
                    i += 1
                tmp[i] = KS.FaceOrientation.Y
                i += 1
                tmp[i] = KS.FaceOrientation.P
                i += 1
                tmp[i] = KS.FaceOrientation.R
                i += 1
                tmp[i] = KS.SpeakingConfidence

                out_dict[KS.TrackingId] = tmp.tolist()

        return out_dict

    # returns array of values for each timestep, either orig val or filled with 'value'
    def get_complete_axis(self, time_data, data_buf_i, filler=False):
        # TODO: needs rewriting, it seems to get stuck when only one frame of a person was recorded! use np.where...

        conf = dataConfig.dataConfig()
        if filler is False or filler == 'default':
            filler = conf.defaultFill

        darr = np.asarray(data_buf_i)
        fill_arr = np.zeros(self.msg_len)
        fill_arr.fill(filler)
        for i in range(0, len(time_data)):
            size = darr.shape

            # stack fill if the data is missing in the end
            if size[0] <= i and i > 1:
                fill_arr[0] = time_data[i][1]
                darr = np.vstack((darr, fill_arr))

            # insert fill if the data is missing within the interval
            elif time_data[i][1] != darr[i, 0]:
                fill_arr[0] = time_data[i][1]
                darr = np.insert(darr, i, fill_arr, axis=0)
            else:
                pass

        return darr.tolist()

    def fill_data_buf_axes(self, time_list, data_list, IDs=False, filler=False):

        conf = dataConfig.dataConfig()
        if filler == False or filler == 'default':
            filler = conf.defaultFill
        if IDs == False or IDs == 'default':
            IDs = range(6)

        for p in IDs:
            if data_list[p] != 'e' or len(data_list[p]) > 5:
                print 'filling axis for ID' + str(p) + '...'
                data_list[p] = self.get_complete_axis(time_list, data_list[p], filler)
            else:
                print 'data fo ID', p, 'too short or not recorded'

            # dataBuf[ID] = self.get_complete_axis(timeBuf, dataBuf[ID], filler)

        return data_list

    # transforms vectors to a given frame frame
    def vect_to_frame(self, vector, cubePose):

        # get translations and rotation quaternions
        try:
            cubeTran = cubePose[0:3]
            cubeQuat = np.asarray(cubePose[3:7])
        except TypeError:
            print 'Could not read cube_pose when trying to change the frame of kinect data.' \
                  ' Has the marker cube position been initialized?'
            sys.exit()

        cubeTran = [val/1000 for val in cubeTran]
        # # debug!!!!
        # cubePose = [0, 0, 0, 1, 0, 0, 0]
        # cubeTran = [-1, -1, 0]
        # cubeQuat = [0.70711, 0, 0, 0.70711]

        # vector to homogenous vector
        homVector = np.empty(4)
        homVector[0:3] = vector
        homVector[3] = 1

        # build the rotation matrix
        M = math_helper.quat_to_rot_matrix(cubeQuat) # 90 deg about z axis
        M = M.transpose()

        # translate
        homVector[0:3] = homVector[0:3] - np.asarray(cubeTran)
        # rotate
        homVector = M.dot(homVector)

        return homVector[0:3].tolist()

    # loads the calibration file so that vectors can be normalised
    def load_marker_cube_pose(self, fstring='last_cube_calibration'):

        fileName = config.Config().root_dir + config.settings.calib.path + '/' + fstring + '.txt'
        fo = open(fileName, 'r')
        calibFileContent = fo.read()
        calibBuf = calibFileContent.split(';')
        fo.close()

        calibBuf = [float(i) for i in calibBuf[0:7]]
        print 'read the following calib data: ', calibBuf[0:7]
        # return calibBuf[0:7]
        self.cubePose = calibBuf[0:7]

    def get_data_dict(self):
        return common.dictify(self.get_feature_list(), inv=True)

    def get_feature_list(self):

        feat_list = []

        if self.version == 3 or self.version == 4:
            # feed all into tmp
            # tmp[i] = KS.CaptureTimeMs
            # i += 1
            feat_list.append('CaptureTimeMs')
            # for JT in KS.Joints:
            for joint_type in ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                               'ShoulderRight', 'ElbowRight', 'WristRight', 'HipLeft', 'HipRight', 'SpineShoulder']:
                feat_list.append(joint_type + 'Px')
            #     tmp[i] = JT.Position.x
            #     i += 1
                feat_list.append(joint_type + 'Py')
            #     tmp[i] = JT.Position.y
            #     i += 1
                feat_list.append(joint_type + 'Pz')
            #     tmp[i] = JT.Position.z
            #     i += 1
                feat_list.append(joint_type + 'St')
            #     tmp[i] = JT.State
            #     i += 1
            # for FP in KS.FaceProperties:
                if joint_type == 'Head':
                    feat_list.append(joint_type + 'OrientVecX')
                    feat_list.append(joint_type + 'OrientVecY')
                    feat_list.append(joint_type + 'OrientVecZ')

            for face_property in ['Engaged', 'LeftEyeClosed', 'RightEyeClosed', 'MouthOpen', 'MouthMoved',
                                  'LookingAway']:
                feat_list.append(face_property)
            #     tmp[i] = FP.Result
            #     i += 1
            feat_list.append('faceY')
            # tmp[i] = KS.FaceOrientation.Y
            # i += 1
            feat_list.append('faceP')
            # tmp[i] = KS.FaceOrientation.P
            # i += 1
            feat_list.append('faceR')
            # tmp[i] = KS.FaceOrientation.R
            # i += 1
            feat_list.append('SpeakingConf')
            # tmp[i] = KS.SpeakingConfidence
            # i += 1
            feat_list.append('FrameNr')

        elif self.version == 1 or self.version == 2:
            # feed all into tmp
            # tmp[i] = KS.CaptureTimeMs
            # i += 1
            feat_list.append('CaptureTimeMs')
            # for JT in KS.Joints:
            for joint_type in ['SpineBase', 'HipLeft', 'HipRight', 'SpineMid', 'SpineShoulder', 'Neck', 'Head',
                               'ShoulderLeft', 'ShoulderRight', 'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight']:
                feat_list.append(joint_type + 'Px')
            #     tmp[i] = JT.Position.x
            #     i += 1
                feat_list.append(joint_type + 'Py')
            #     tmp[i] = JT.Position.y
            #     i += 1
                feat_list.append(joint_type + 'Pz')
            #     tmp[i] = JT.Position.z
            #     i += 1
                feat_list.append(joint_type + 'Ox')
            #     tmp[i] = JT.Orientation.x
            #     i += 1
                feat_list.append(joint_type + 'Oy')
            #     tmp[i] = JT.Orientation.y
            #     i += 1
                feat_list.append(joint_type + 'Oz')
            #     tmp[i] = JT.Orientation.z
            #     i += 1
                feat_list.append(joint_type + 'Ow')
            #     tmp[i] = JT.Orientation.w
            #     i += 1
                feat_list.append(joint_type + 'St')
            #     tmp[i] = JT.State
            #     i += 1
            # for FP in KS.FaceProperties:
            for face_property in ['Engaged', 'LeftEyeClosed', 'RightEyeClosed', 'MouthOpen', 'MouthMoved',
                                  'LookingAway']:
                feat_list.append(face_property)
            #     tmp[i] = FP.Result
            #     i += 1
            feat_list.append('faceY')
            # tmp[i] = KS.FaceOrientation.Y
            # i += 1
            feat_list.append('faceP')
            # tmp[i] = KS.FaceOrientation.P
            # i += 1
            feat_list.append('faceR')
            # tmp[i] = KS.FaceOrientation.R
            # i += 1
            feat_list.append('SpeakingConf')
            # tmp[i] = KS.SpeakingConfidence
            # i += 1
            feat_list.append('FrameNr')

        elif self.version == 0:

            feat_list.append('CaptureTimeMs')
            # for JT in KS.Joints:
            for joint_type in ['SpineBase', 'HipLeft', 'HipRight', 'SpineMid', 'SpineShoulder', 'Neck', 'Head',
                               'ShoulderLeft', 'ShoulderRight', 'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight']:
                feat_list.append(joint_type + 'Px')
            #     tmp[i] = JT.Position.x
            #     i += 1
                feat_list.append(joint_type + 'Py')
            #     tmp[i] = JT.Position.y
            #     i += 1
                feat_list.append(joint_type + 'Pz')
            #     tmp[i] = JT.Position.z
            #     i += 1
                feat_list.append(joint_type + 'Ox')
            #     tmp[i] = JT.Orientation.x
            #     i += 1
                feat_list.append(joint_type + 'Oy')
            #     tmp[i] = JT.Orientation.y
            #     i += 1
                feat_list.append(joint_type + 'Oz')
            #     tmp[i] = JT.Orientation.z
            #     i += 1
                feat_list.append(joint_type + 'Ow')
            #     tmp[i] = JT.Orientation.w
            #     i += 1
                feat_list.append(joint_type + 'St')
            #     tmp[i] = JT.State
            #     i += 1
            # for FP in KS.FaceProperties:
            for face_property in ['EyeLeftClosed', 'EyeRightClosed', 'MouthOpen', 'MouthMoved', 'LookingAway']:
                feat_list.append(face_property)
            #     tmp[i] = FP.Result
            #     i += 1
            feat_list.append('faceY')
            # tmp[i] = KS.FaceOrientation.Y
            # i += 1
            feat_list.append('faceP')
            # tmp[i] = KS.FaceOrientation.P
            # i += 1
            feat_list.append('faceR')
            # tmp[i] = KS.FaceOrientation.R
            # i += 1
            feat_list.append('SpeakingConf')
            # tmp[i] = KS.SpeakingConfidence
            # i += 1

        else:
            raise ValueError('Data dictionary version not defined?')


        # print len(feat_list)
        # print feat_list
        # raw_input()

        return feat_list

    def get_features_indexes(self, features):
        return common.match_lists_to_idx(self.get_feature_list(), features)

# old to update, clean/remove
    def initFPS(self):
        self.FPSq = deque()

    def updateFPS(self, mSec):
        if mSec != 0:
            if not self.FPSq:
                self.FPSq.append(mSec)
            elif mSec != self.FPSq[len(self.FPSq) - 1]:
                self.FPSq.append(mSec)
                for k in range(0, len(self.FPSq)):
                    if mSec - self.FPSq[0] > 1000:
                        self.FPSq.popleft()
                    else:
                        break

    def getFPS(self):
        return len(self.FPSq)


# --------------------------------------------------------------------------------------------------- data file handling


def read_proto_file(f_string, version=3):

    time_list = []
    data_list = {k: v for k, v in zip(range(6), 'e' * 6)}       # 6 kinect bodies

    kdh = KinectDataHelper(version=version)
    msg_type = kdh.load_message_type()
    kdh.load_marker_cube_pose(f_string)

    # if custom name was passed, load that, otherwise data with timeString
    full_file_name = config.settings.recd.path + '/' + f_string + '.txt'
    with open(full_file_name, 'rb') as content_file:
        file_content = content_file.read()

    print 'reading the file...'

    end = 0
    i = 0
    while end != len(file_content):

        (size, position) = decoder._DecodeVarint(file_content[end:len(file_content)], 0)

        # offset
        position = end + position
        end = position + size

        msg_type.ParseFromString(file_content[position:end])

        # if the message is not empty
        if len(msg_type.KinectStream) > 0:
            time_list.append(kdh.time_arr_from_msg(msg_type))

            person_data = kdh.msg_to_person_data_d(msg_type)
            for key in person_data:
                if data_list[key] == 'e':
                    data_list[key] = [person_data[key]]
                else:
                    data_list[key].append(person_data[key])

        if i % 30 == 0:
            sys.stdout.write('.')
        i += 1

    print 'File ' + full_file_name + '.txt was successfully read'
    return time_list, data_list
