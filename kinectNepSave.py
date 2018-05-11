import nep
import json
    
node = nep.node("kinect_human") # Create a new node
#conf = node.conf_sub() # Select the configuration of the subscriber
conf = node.conf_sub(network="direct", ip="192.168.11.43", port="9090")
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



##
##
with open('data.txt', 'w') as outfile:
    json.dump(data, outfile)
##        
##
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
##

