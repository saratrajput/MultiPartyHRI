#===================================NEP===================================
import nep  
import time 

server = nep.server('127.0.0.1', 8010) #Create a new server instance
#-------------------------------------------------------------------------------

#============================== Pepper ==============================
from naoqi import ALProxy
import qi # For Tablet Display

robotIp = "192.168.11.37"
port = 9559

#session = qi.Session()
#session.connect("tcp://{}:{}".format(robotIp, port))
#tabletService = session.service("ALTabletService")

#-----------------------------------Proxies-----------------------------------
#behaviourProxy = ALProxy("ALBehaviorManager", robotIp, port)
#animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
#autonomousLifeProxy = ALProxy("ALAutonomousLife", robotIp, port)
motionProxy = ALProxy("ALMotion", robotIp, port)
#behaviourProxy.stopAllBehaviors()
#===============================================================================

while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
    
    #print("This message is from Client: " +  request)
    #print(type(request))
    try:
        normalSpeechProxy.say(request)
       # motionProxy.setBreathEnabled('Body', True)
#        tabletService.showInputTextDialog(request, "-", "-")
#        tabletService.hideDialog()
    except:
        pass
    finally:
        server.send_info(msg) # Send server response


