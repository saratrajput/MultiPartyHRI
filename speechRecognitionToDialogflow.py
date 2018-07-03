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

#==============================Facebook Messenger==============================
myUsername = "pattarsuraj@gmail.com"
myPassword = "TuIlvibgotVefM2"
userClient = Client(myUsername, myPassword)

pepperUsername = "gentianeventurelab@gmail.com"
pepperPassword = "tokyo2009Noko"
pepperClient = Client(pepperUsername, pepperPassword)
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

CLIENT_ACCESS_TOKEN = '69b8fa6da6494f948c68bdf3242ec65d' # From DialogFlow Agent (V1)

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
#	with microphone as source: r.adjust_for_ambient_noise(source)
#       recognizer.energy_threshold = 100;
        print("Speech Recognizer: Set minimum energy threshold to {}".format(recognizer.energy_threshold))

        while True:
            print("Speech Recognizer: Say something!")
            with microphone as source: audio = recognizer.listen(source)
            print("Speech Recognizer: Got it! Now to recognize it...")
                    
            try:
                # recognize speech using Google Speech Recognition
                value = recognizer.recognize_google(audio)

                # Convert Speech Recognition to string for feeding Dialogflow
                valueString = unicodedata.normalize('NFKD', value).encode('ascii','ignore')

                # we need some special handling here to correctly print unicode characters to standard output
                if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                    print(u"Speech Recognizer: You said {}".format(value).encode("utf-8"))
                    print("Speech Recognizer: You said2: " + valueString)

                    # Send User's speech to Facebook
                    userClient.send(Message(text=valueString), thread_id=pepperClient.uid, thread_type=ThreadType.USER)

                    # Send request to Dialogflow and get reply
                    pepperSpeech = pepperAgent.handle(valueString)
                    print("Dialogflow: " + pepperSpeech)
#=================================== NEP ===================================
#
#                    msg = pepperSpeech # Message to send as request
#                    client.send_info(msg)   # Send request
#                    print("Message for the server: " +  msg)
#                    pepperClient.send(Message(text=msg), thread_id=userClient.uid, thread_type=ThreadType.USER)
#                    client.listen_info()
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


