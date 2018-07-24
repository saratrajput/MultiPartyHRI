#!/home/sp/myEnvPy2/bin/python
#==============================Break on Keystroke==============================
import eventlet
eventlet.monkey_patch()
#==============================Facebook Messenger==============================
from fbchat import Client
from fbchat.models import *
#=================================== NEP ===================================
import nep
import time
#============================== Speech Recognition ==============================
import speech_recognition as sr
import unicodedata
#===============================================================================
# Microsoft Bing Key
bingKey = "5112e9b177784f0796586c31fd82d8c6"
#==============================Facebook Messenger==============================
myUsername = "pattarsuraj@gmail.com"
myPassword = "TuIlvibgotVefM2"
userClient = Client(myUsername, myPassword) # Login to User's account

pepperUsername = "gentianeventurelab@gmail.com"
pepperPassword = "tokyo2009Noko"
pepperClient = Client(pepperUsername, pepperPassword) # Login to Robot's account
#===============================================================================

client = nep.client('127.0.0.1', 8010) #Create a new NEP client instance

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

CLIENT_ACCESS_TOKEN = '346ddf0ddbcc4063859ba598a0e70d42' # From DialogFlow Agent (V1)

class DialogFlowAgent(object):
    def __init__(self):

        self.AI = apiai.ApiAI(CLIENT_ACCESS_TOKEN)

    def handle(self, text):
        self.request = self.AI.text_request()
        self.request.lang = 'en'

        self.request.query = text
        response = self.request.getresponse().read()
        speech = str(json.loads(response)['result']['fulfillment']['speech'])
#        speech = u' '.join((json.loads(response)['result']['fulfillment']['speech'])).encode('utf-8').strip()
        return speech
#===============================================================================

recognizer = sr.Recognizer() # Speech Recognizer
microphone = sr.Microphone() # Microphone

def main():
    pepperAgent = DialogFlowAgent()

    try:
        print("Speech Recognizer: A moment of silence, please...")
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)  # listen for 1 second to calibrate the energy threshold for ambient noise levels
            print("Say something!")
            audio = recognizer.listen(source)

#       recognizer.energy_threshold = 100; # Manual setting for Energy Threshold

        print("Speech Recognizer: Set minimum energy threshold to {}".format(recognizer.energy_threshold))

        while True:
            print("Speech Recognizer: Say something!")
#            with eventlet.Timeout(10):
            with microphone as source: audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
            print("Speech Recognizer: Got it! Now to recognize it...")
                    
            try:
                # recognize speech using Google Speech Recognition
                print("Trying to Recognize ...")
#                with eventlet.Timeout(8):
                value = recognizer.recognize_bing(audio, key=bingKey)
                print("RECOGNIZED")

                # Convert Speech Recognition to string for feeding Dialogflow
                valueString = unicodedata.normalize('NFKD', value).encode('ascii','ignore')
                print("VALUESTRING CONVERTED")

                # we need some special handling here to correctly print unicode characters to standard output
                if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                    print(u"Speech Recognizer: You said {}".format(value).encode("utf-8"))

                    # Send User's speech to Facebook
                    print("SENDING TO FACEBOOK")
                    userClient.send(Message(text=valueString), thread_id=pepperClient.uid, thread_type=ThreadType.USER) 

                    # Send request to Dialogflow and get reply
                    print("SENDING TO DIALOGFLOW")
                    pepperSpeech = pepperAgent.handle(valueString) 
                    print("Dialogflow: " + pepperSpeech)

#=================================== NEP ===================================
# Send Dialogflow reply to Pepper's Text to Speech
                    msg = str(pepperSpeech) # Message to send as request
                    print (msg)
                    client.send_info(msg)   # Send request
                    print("Message for the server: " +  msg)
                    pepperClient.send(Message(text=msg), thread_id=userClient.uid, thread_type=ThreadType.USER) # Pepper replies on Facebook
                    client.listen_info()
#                    time.sleep(.01) # Wait one second
#===============================================================================
                
                else:  # this version of Python uses unicode for strings (Python 3+)
                    print("You said {}".format(value))
            except sr.UnknownValueError:
                print("Speech Recognizer: Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Speech Recognizer: Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    except KeyboardInterrupt:
        pass
        

if __name__ == '__main__':
    main()


