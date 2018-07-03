import threading
import nep
import json

node = nep.node("subs")
conf = node.conf_sub(mode="many2many")
sub = node.new_sub("/leap_motion",conf)

gesture = {}
gesture_defined = False
gesture_counter = 0
adquisition =  False

exit_ = False


while not exit_:
    s, data = sub.listen_info()
    if s:
        if gesture_defined == False:
            for key, value in data.iteritems():
                gesture[key] = []
            gesture_defined = True

        start = data["Hand_GrabStrength"]
        print start

        for key, value in data.iteritems():
            gesture[key].append(value)

        if start == 1.0:
                print "saving gesture:" + str(gesture_counter)
                gesture_counter = gesture_counter + 1 
            
                with open('gestures/gesture_' + str(gesture_counter) + '.txt', 'w') as gestureData:
                            json.dump(gesture, gestureData)
                exit_ = True
                

        
