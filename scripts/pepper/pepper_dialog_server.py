import os
import time

import nep
from naoqi import ALProxy


# Declare robot ip and port
server = nep.server("127.0.0.1", 8010)  # Create a new server instance
robot_ip_port = list()

with open("/home/sp/multiPartyHRI/robotIpPort.txt", "r") as my_robot_info:
    for line in my_robot_info.readlines():
        robot_ip_port.append(line.strip())

robot_ip = robot_ip_port[0]
port = int(robot_ip_port[1])
# -----------------------------------Proxies-----------------------------------
# animatedSpeechProxy = ALProxy("ALAnimatedSpeech", robotIp, port)
normal_speech_proxy = ALProxy("ALTextToSpeech", robot_ip, port)
motion_proxy = ALProxy("ALMotion", robot_ip, port)
# ===============================================================================

normal_speech_proxy.setParameter("speed", 80)
normal_speech_proxy.setParameter("pitchShift", 0.8)

while True:
    msg = {"message": "hello client"}  # Message to send as response
    request = server.listen_info()  # Wait for client request

    try:
        normal_speech_proxy.say(request)
        motion_proxy.setBreathEnabled("Body", True)
    except:
        pass
    finally:
        server.send_info(msg)  # Send server response
