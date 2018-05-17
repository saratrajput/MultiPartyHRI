import nep  
import time 

server = nep.server('127.0.0.1', 8010) #Create a new server instance

########################################
#from naoqi import ALProxy
#tts = ALProxy("ALTextToSpeech", "192.168.11.46", 9559)
########################################
names = ["Unknown"] * 2 
while True:
    msg = {"message":"hello client"} # Message to send as response
    request = server.listen_info() # Wait for client request
#    print request
    server.send_info(msg) # Send server response

    if names[1] != request:
        names[1] = request
        print names[1]

#    tts.say("Hello, %s" % request)
    

#tts.say("Hello, world!")


