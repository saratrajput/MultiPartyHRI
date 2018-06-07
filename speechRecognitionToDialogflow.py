#=================================== Pepper ===================================

#from naoqi import ALProxy
#
## Declare robot ip and port
#robotIp = "192.168.1.101"
#port = 9559
#try:
#    normalSpeechProxy = ALProxy("ALTextToSpeech", robotIp, port)
#    normalSpeechProxy.say("I am ready.")
#except:
#    print("Pepper refuses connection")
#behaviourProxy = ALProxy("ALBehaviorManager", robotIp, port)
#behaviourProxy.stopAllBehaviors()
#-------------------------------------------------------------------------------

#=================================== NEP ===================================

import nep
import time
client = nep.client('127.0.0.1', 8010) #Create a new server instance
#-------------------------------------------------------------------------------

#============================== Speech Recognition ==============================
import speech_recognition as sr
import unicodedata
#-------------------------------------------------------------------------------


#=================================== DialogFlow ===================================
import os.path
import sys
import json

try:
    import apiai
except ImportError:
    sys.path.append(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
    )
    import apiai
#-------------------------------------------------------------------------------

CLIENT_ACCESS_TOKEN = '69b8fa6da6494f948c68bdf3242ec65d' # From DialogFlow Agent (V1)

r = sr.Recognizer()
m = sr.Microphone()

class DialogFlowAgent(object):
    def __init__(self):

        self.AI = apiai.ApiAI(CLIENT_ACCESS_TOKEN)

    def handle(self, text):
        self.request = self.AI.text_request()
        self.request.lang = 'en'

        self.request.query = text
        response = self.request.getresponse().read()
        speech = str(json.loads(response)['result']['fulfillment']['speech'])
        return speech
		#~ action = str(json.loads(response)['result']['action'])
		#~ if action is not '':
			#~ return speech, action
		#~ else:
			#~ return speech, None

def main():
    pepperAgent = DialogFlowAgent()

    try:
	print("A moment of silence, please...")
	with m as source: r.adjust_for_ambient_noise(source)
	print("Set minimum energy threshold to {}".format(r.energy_threshold))

        while True:
	   
            print("Say something!")
            with m as source: audio = r.listen(source)
            print("Got it! Now to recognize it...")
                    
            try:
                # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio)

                # Convert Speech Recognition to string for feeding Dialogflow
                valueString = unicodedata.normalize('NFKD', value).encode('ascii','ignore')

                # we need some special handling here to correctly print unicode characters to standard output
                if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                    print(u"You said {}".format(value).encode("utf-8"))

                    # Send request to Dialogflow and Pepper replies
                    pepperSpeech = pepperAgent.handle(valueString)
                    print(pepperSpeech)
#----------------------------------- NEP -----------------------------------

                    msg = pepperSpeech # Message to send as request
                    client.send_info(msg)   # Send request
                    client.listen_info()
                    time.sleep(1) # Wait one second
                    #print (client.listen_info()) # Wait for server response
#-------------------------------------------------------------------------------
                
                else:  # this version of Python uses unicode for strings (Python 3+)
                    print("You said {}".format(value))
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    except KeyboardInterrupt:
        pass
        

if __name__ == '__main__':
    main()


