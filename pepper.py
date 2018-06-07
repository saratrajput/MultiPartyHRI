from naoqi import ALProxy

# Declare robot ip and port
robotIp = "192.168.0.108"
port = 9559

behaviourProxy = ALProxy("ALBehaviorManager", robotIp, port)
animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
autonomousLifeProxy = ALProxy("ALAutonomousLife", robotIp, port)
motionProxy = ALProxy("ALMotion", robotIp, port)

# First stop all behaviours
#def stopBehaviour():
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
