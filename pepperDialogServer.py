#===================================NEP===================================
import nep  
import time 

server = nep.server('127.0.0.1', 8010) #Create a new server instance
#-------------------------------------------------------------------------------

#============================== Pepper ==============================
from naoqi import ALProxy

robotIp = "192.168.0.108"
port = 9559

#-----------------------------------Proxies-----------------------------------
behaviourProxy = ALProxy("ALBehaviorManager", robotIp, port)
animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
autonomousLifeProxy = ALProxy("ALAutonomousLife", robotIp, port)
motionProxy = ALProxy("ALMotion", robotIp, port)
behaviourProxy.stopAllBehaviors()
#===============================================================================

while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
    server.send_info(msg) # Send server response
    print("%s" % request)
    normalSpeechProxy.say("%s" % request)

