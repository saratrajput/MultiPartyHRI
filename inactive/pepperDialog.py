import apiai
import uuid # To generate unique session id

CLIENT_ACCESS_TOKEN = '69b8fa6da6494f948c68bdf3242ec65d'
sessionId = uuid.uuid1()

def main():

    # Getting the service ALDialog
    ALDialog = session.service("ALDialog")
    ALDialog.setLanguage("English")

    # writing topics' qichat code as text strings (end-of-line characters are important!)
    topic_content_1 = ('topic: ~example_topic_content()\n'
                       'language: enu\n'
                       'u: ([e:FrontTactilTouched e:MiddleTactilTouched e:RearTactilTouched]) Hello human!\n'
                       'u:(_*) $text=$1'
                       )
    memory = ALProxy("ALMemory") 
    inputString = memory.getData("text")

    ai = apiai.ApiAI(CLIENT_ACCESS_TOKEN)

    request = ai.text_request()

    request.session_id = "sessionId"

    request.query = "Hello"

    response = request.getresponse()

    print (response.read())


if __name__ == '__main__':
    main()
