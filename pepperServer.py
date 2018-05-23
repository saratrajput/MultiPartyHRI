import nep  
import time 

server = nep.server('127.0.0.1', 8010) #Create a new server instance

############## Pepper ##################
from naoqi import ALProxy
tts = ALProxy("ALTextToSpeech", "192.168.11.21", 9559)
########################################
names = ["Unknown"] * 2 
while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
#    print request
    server.send_info(msg) # Send server response

    if names[1] != request:
        names[1] = request
        #print names[1]
        # Send speech command to Pepper whenever a new face is detected
        # After '^' indicates the animation to be done
        tts.say("Hello!, ^start(animations/Stand/Gestures/Hey_1)  %s" % names[1])

#    tts.say("Hello, %s" % request)
    

#tts.say("Hello, world!")


