import nep
import json
    
import sys
import datetime
date_string = datetime.datetime.now().strftime("%d-%m-%H-%M")

node = nep.node("kinect_human") # Create a new node
#conf = node.conf_sub() # Select the configuration of the subscriber
# For direct connection: ip= ip of system connected to Kinect 
conf = node.conf_sub(network="direct", ip="192.168.0.108", port="9090") 
sub = node.new_sub("/kinect_human", conf) # Set the topic and the configuration of the subscriber

data={}
data_defined = False

# Read the information published in the topic registered

i = 0
while i < 100:
    s, msg = sub.listen_info()
    if s:
        if data_defined == False:
            for key, value in msg.iteritems():
                data[key] = []
            data_defined = True

        for key, value in msg.iteritems():
            data[key].append(value)

        print msg["face_yaw"]
        i=i+1


with open('data' + date_string + '.txt', 'w') as outfile:
    json.dump(data, outfile)

#============================== Sample list of features ============================== 
####"SpineMid"
####"SpineShoulder"
####"WristLeft"
####"WristRight"
####"ElbowLeft"
####"ElbowRight"
####"HipLeft"
####"HipRight"
####"Neck"
####"Head"
####"face_yaw"
####"face_pitch"
####"face_roll"
#===============================================================================
