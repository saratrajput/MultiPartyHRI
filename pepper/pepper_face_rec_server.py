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
import os
import time

import logging
import nep
from naoqi import ALProxy


server = nep.server("127.0.0.1", 8011)  # Create a new server instance
# Get robot ip and port data from file
robot_ip_port = list()

with open("/home/sp/multiPartyHRI/robotIpPort.txt", "r") as myRobotInfo:
    for line in myRobotInfo.readlines():
        robot_ip_port.append(line.strip())

robot_ip = robot_ip_port[0]
port = int(robot_ip_port[1])
# ===================================Proxies===================================
# animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normal_speech_proxy = ALProxy("ALTextToSpeech", robot_ip, port)

normal_speech_proxy.setParameter("speed", 80)
normal_speech_proxy.setParameter("pitchShift", 0.8)
# ===============================================================================
names = ["Stranger"] * 2  # List to store names from face recognition output

stranger_counter = 0  # Instantiate counter for when no face is recognized
rec_name_counter = 0  # Instantiate counter for first face recognized
counter_value = 200

recName = str()  # Instantiate string to store first face recognized

while True:
    msg = {"message": "hello client"}  # Message to send as response
    request = server.listen_info()  # Wait for client request
    #    request = raw_input("Give input:\n") # For debugging

    if names[1] != request:
        names[1] = request
        if names[1] == "Stranger":
            if stranger_counter == 0:
                #                print "Hello", names[1], "What is your name?" # For debugging
                normal_speech_proxy.say(
                    "Hey! %s. What is your name?" % names[1]
                )  # Pepper's tts
                stranger_counter = counter_value
        elif rec_name_counter == 0:
            recName = names[1]
            if names[1] == recName:
                #                print "Hello", names[1] # For debugging
                normal_speech_proxy.say("Hey! %s." % names[1])  # Pepper's tts
                rec_name_counter = counter_value
        elif names[1] != recName:
            #            print "Hello New", names[1] # For debugging
            normal_speech_proxy.say("Hey! %s." % names[1])  # Pepper's tts

    if stranger_counter > 0:
        stranger_counter = stranger_counter - 1
    if rec_name_counter > 0:
        rec_name_counter = rec_name_counter - 1

    #    print "StrangerCounter:", strangerCounter # For debugging
    #    print "recName Counter:", recNameCounter # For debugging

    server.send_info(msg)  # Send server response
