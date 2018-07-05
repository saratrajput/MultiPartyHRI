# Python 2.7 Script
# Signal Handler to break while loop on Keyboard Interrupt: Ctrl-Z
import signal 
import sys
# NEP
import nep
import json
# Adding date and time to filename
import sys
import datetime
<<<<<<< HEAD
date_string = datetime.datetime.now().strftime("%d-%m-%H-%M")
=======

date_string = datetime.datetime.now().strftime("%d-%m-%H-%M")

# Signal Handler to break while loop on Keyboard Interrupt: Ctrl-Z
def signal_handler(signum, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGTSTP, signal_handler)

interrupted = False
>>>>>>> e238e61bc2ed6214ee1f63b273179912dcbe696f

node = nep.node("kinect_human") # Create a new node
# Select the configuration of the subscriber
#conf = node.conf_sub() # Select the configuration of the subscriber
conf = node.conf_sub(network = "direct", ip = "192.168.11.21", port = 9090)
sub = node.new_sub("/kinect_human", conf) # Set the topic and the configuration of the subscriber

data={}
data_defined = False

# Read the information published in the topic registered

while True:
    s, msg = sub.listen_info()
    if s:
        if data_defined == False:
            for key, value in msg.iteritems():
                data[key] = []
            data_defined = True

        for key, value in msg.iteritems():
            data[key].append(value)

        print msg["face_yaw"]

    if interrupted:
        print("Gotta go")
        break

with open("data" + date_string +  ".txt", 'w') as outfile:
    json.dump(data, outfile)


print "Broke successfully"

