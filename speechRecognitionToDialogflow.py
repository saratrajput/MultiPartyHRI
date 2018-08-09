#!/home/sp/myEnvPy2/bin/python
# Import credentials for facebook login, bingKey, Dialogflow agent key
import os 
#==============================Facebook Messenger==============================
from fbchat import Client
from fbchat.models import *
#=================================== NEP ===================================
import nep
import time
#============================== Speech Recognition ==============================
import speech_recognition as sr
import unicodedata
#==============================Facebook Messenger==============================
# Get username, passwords and agent keys stored in a separate file
credentials = list()

with open("/home/sp/credentials.txt", "r") as myCredentialFile:
    for line in myCredentialFile.readlines():
        credentials.append(line.strip())

myUsername = credentials[0]
myPassword = credentials[1]
userClient = Client(myUsername, myPassword) # Login to User's account

pepperUsername = credentials[2]
pepperPassword = credentials[3]
pepperClient = Client(pepperUsername, pepperPassword) # Login to Robot's account
#===============================================================================
# Microsoft Bing Key
bingKey = credentials[4]
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

CLIENT_ACCESS_TOKEN = credentials[5] # From DialogFlow Agent (V1)

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
            # Pepper replies on Facebook
            pepperClient.send(Message(text="Say something!"), thread_id=userClient.uid, thread_type=ThreadType.USER)
            with microphone as source: audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
            print("Speech Recognizer: Got it! Now to recognize it...")
            # Pepper replies on Facebook
            pepperClient.send(Message(text="Got it! Now to recognize it..."), thread_id=userClient.uid, thread_type=ThreadType.USER)
                    
            try:
                # recognize speech using Google Speech Recognition
                print("Trying to Recognize ...")
                value = recognizer.recognize_google(audio)
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
                # Pepper replies on Facebook
                pepperClient.send(Message(text="Oops! Didn't catch that"), thread_id=userClient.uid, thread_type=ThreadType.USER)
            except sr.RequestError as e:
                print("Speech Recognizer: Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
                # Pepper replies on Facebook
                pepperClient.send(Message(text="Uh oh! Couldn't request results from Google Speech Recognition service"), thread_id=userClient.uid, thread_type=ThreadType.USER)
    except KeyboardInterrupt:
        pass
        

if __name__ == '__main__':
    main()


