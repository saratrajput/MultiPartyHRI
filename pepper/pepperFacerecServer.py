"""
This script takes the output of the face recognizer and gives it to the 
robot's text to speech. At the time of writing this script, the decision 
making block is not ready. Because of this we have had to write some if-else
blocks which will help in a smoother interaction. Without these condition 
blocks, the robot keeps on repeating the recognized face names which can cause
problems with the speech recognition and chatbot. The decision blocks help
to avoid repeating names by the number specified in the counters once the face
has been recognized once.
"""
#===================================NEP===================================
import nep  
import time 

server = nep.server('127.0.0.1', 8011) #Create a new server instance
#============================== Pepper ==============================
from naoqi import ALProxy
import os

# Get robot ip and port data from file
robotIpPort = list()

with open("/home/sp/multiPartyHRI/robotIpPort.txt", "r") as myRobotInfo:
    for line in myRobotInfo.readlines():
        robotIpPort.append(line.strip())

robotIp = robotIpPort[0]
port = int(robotIpPort[1])
#===================================Proxies===================================
#animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)

normalSpeechProxy.setParameter("speed", 80)
normalSpeechProxy.setParameter("pitchShift", 0.8)
#===============================================================================
names = ["Stranger"] * 2 # List to store names from face recognition output

strangerCounter = 0 # Instantiate counter for when no face is recognized
recNameCounter = 0 # Instantiate counter for first face recognized
counterValue = 200

recName = str() # Instantiate string to store first face recognized

while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
#    request = raw_input("Give input:\n") # For debugging

    if names[1] != request:
        names[1] = request
        if names[1] == "Stranger":
            if strangerCounter == 0:
#                print "Hello", names[1], "What is your name?" # For debugging
                normalSpeechProxy.say("Hey! %s. What is your name?" % names[1]) # Pepper's tts
                strangerCounter = counterValue
        elif recNameCounter == 0:
            recName = names[1]
            if names[1] == recName:
#                print "Hello", names[1] # For debugging
                normalSpeechProxy.say("Hey! %s." % names[1]) # Pepper's tts
                recNameCounter = counterValue
        elif names[1] != recName:
#            print "Hello New", names[1] # For debugging
            normalSpeechProxy.say("Hey! %s." % names[1]) # Pepper's tts
    
    if strangerCounter > 0:
        strangerCounter = strangerCounter - 1
    if recNameCounter > 0:
        recNameCounter = recNameCounter -1

#    print "StrangerCounter:", strangerCounter # For debugging
#    print "recName Counter:", recNameCounter # For debugging

    server.send_info(msg) # Send server response
    


