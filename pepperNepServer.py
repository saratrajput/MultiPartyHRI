import nep  
import time 

server = nep.server('127.0.0.1', 8010) #Create a new server instance

#============================== Pepper ==============================
from naoqi import ALProxy

robotIp = "192.168.1.105"
port = 9559

behaviourProxy = ALProxy("ALBehaviorManager", robotIp, port)
animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
autonomousLifeProxy = ALProxy("ALAutonomousLife", robotIp, port)
motionProxy = ALProxy("ALMotion", robotIp, port)
behaviourProxy.stopAllBehaviors()
#===============================================================================
names = ["Unknown"] * 2 
while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
    server.send_info(msg) # Send server response

    if names[1] != request:
        names[1] = request
        #print names[1]
        # Send speech command to Pepper whenever a new face is detected
        # After '^' indicates the animation to be done
        #tts.say("Hello!, ^start(animations/Stand/Gestures/Hey_1)  %s" % names[1])
        animatedSpeechProxy.say("Hey! ^run(animations/Think_1) %s" % names[1])
#        time.sleep(2)
#    tts.say("Hello, %s" % request)
    


