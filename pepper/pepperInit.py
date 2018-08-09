from naoqi import ALProxy
import os
# Declare robot ip and port
robotIpPort = list()

with open("/home/sp/multiPartyHRI/robotIpPort.txt", "r") as myRobotInfo:
    for line in myRobotInfo.readlines():
        robotIpPort.append(line.strip())

robotIp = robotIpPort[0]
port = int(robotIpPort[1])

behaviourProxy = ALProxy("ALBehaviorManager", robotIp, port)
animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
autonomousLifeProxy = ALProxy("ALAutonomousLife", robotIp, port)
motionProxy = ALProxy("ALMotion", robotIp, port)

# First stop all behaviours
#behaviourProxy.stopAllBehaviors()

# To say something animatedly
#animatedSpeechProxy.say("Hey! ^run(animations/Think_1) you")

# To say something normally
#normalSpeechProxy.say("How are you?")

# To start breathing
#motionProxy.setBreathEnabled('Body', True)

# To stop breathing
#motionProxy.setBreathEnabled('Body', False)

# To rest
#motionProxy.rest()

# To wakeup
#motionProxy.wakeUp()

# To make it (Pepper) stand completly still in order to scan a QR code, I had to do: 
#ALAutonomousMoves.setExpressiveListeningEnabled(False) 
#ALBasicAwareness.stopAwareness() 
