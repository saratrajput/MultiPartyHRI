import os

from naoqi import ALProxy

# Declare robot ip and port
robot_ip_port = list()

with open("/home/sp/multiPartyHRI/robotIpPort.txt", "r") as myRobotInfo:
    for line in myRobotInfo.readlines():
        robot_ip_port.append(line.strip())

robot_ip = robot_ip_port[0]
port = int(robot_ip_port[1])

behaviour_proxy = ALProxy("ALBehaviorManager", robot_ip, port)
animated_speech_proxy = ALProxy("ALAnimatedSpeech", robot_ip, port)
normal_speech_proxy = ALProxy("ALTextToSpeech", robot_ip, port)
autonomous_life_proxy = ALProxy("ALAutonomousLife", robot_ip, port)
motion_proxy = ALProxy("ALMotion", robot_ip, port)

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
