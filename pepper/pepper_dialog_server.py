#===================================NEP===================================
import nep  
import time 

server = nep.server('127.0.0.1', 8010) #Create a new server instance
#============================== Pepper ==============================
from naoqi import ALProxy
import os

# Declare robot ip and port
robotIpPort = list()

with open("/home/sp/multiPartyHRI/robotIpPort.txt", "r") as myRobotInfo:
    for line in myRobotInfo.readlines():
        robotIpPort.append(line.strip())

robotIp = robotIpPort[0]
port = int(robotIpPort[1])
#-----------------------------------Proxies-----------------------------------
#animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
motionProxy = ALProxy("ALMotion", robotIp, port)
#===============================================================================

normalSpeechProxy.setParameter("speed", 80)
normalSpeechProxy.setParameter("pitchShift", 0.8)

while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
    
    try:
        normalSpeechProxy.say(request)
        motionProxy.setBreathEnabled('Body', True)
    except:
        pass
    finally:
        server.send_info(msg) # Send server response
