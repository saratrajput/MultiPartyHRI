#! /usr/bin/env python
# -*- encoding: UTF-8 -*-

"""Example: Using ALDialog Methods"""

import qi
import argparse
import sys
import time

def main(session):
    """
    This example uses ALDialog methods.
    It's a short dialog session with two topics.
    """
    # Getting the service ALDialog
    ALDialog = session.service("ALDialog")
    ALDialog.setLanguage("English")

#============================== Tablet ==============================
#    # Getting the service ALTabletService
#    tabletService = session.service("ALTabletService")
#
#    try: 
#        tabletService = session.service("ALTabletService")
#
#        # Display a local image located in img folder in the root of the web servier
#        # The ip of the robot from the tablet is 198.18.0.1
#        tabletService.showImage("http://198.18.0.1/img/help_charger.png")
#
#        time.sleep(3)
#
#        # Hide the web view
#        tabletService.hideImage()
#    
#    except Exception, e:
#        print "Error was: ", e
#===============================================================================

    # writing topics' qichat code as text strings (end-of-line characters are important!)
    topic_content_1 = ('topic: ~example_topic_content()\n'
                       'language: enu\n'
                       'u: (hello) Hello human!\n'
#                       'u: (e:onStart) Hello?\n'
#                       'u: (_*) You said $1\n'
#                       'u:(hello _*) nice to meet you $1\n'
#                       'u:(e:Dialog/LastInput) I heard \pau=700\ $Dialog/LastInput\n'
                       'u: ([e:FrontTactilTouched e:MiddleTactilTouched e:RearTactilTouched]) Stop touching my head!\n')

    # Loading the topics directly as text strings
    topic_name_1 = ALDialog.loadTopicContent(topic_content_1)

    # Activating the loaded topics
    ALDialog.activateTopic(topic_name_1)

    # Starting the dialog engine - we need to type an arbitrary string as the identifier
    # We subscribe only ONCE, regardless of the number of topics we have activated
    ALDialog.subscribe('my_dialog_example')

    try:
        raw_input("\nSpeak to the robot using rules from both the activated topics. Press Enter when finished:")
    finally:
        # stopping the dialog engine
        ALDialog.unsubscribe('my_dialog_example')

        # Deactivating all topics
        ALDialog.deactivateTopic(topic_name_1)

        # now that the dialog engine is stopped and there are no more activated topics,
        # we can unload all topics and free the associated memory
        ALDialog.unloadTopic(topic_name_1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot's IP address. If on a robot or a local Naoqi - use '127.0.0.1' (this is the default value).")
    parser.add_argument("--port", type=int, default=9559,
                        help="port number, the default value is OK in most cases")

    args = parser.parse_args()
    session = qi.Session()
    try:
        session.connect("tcp://{}:{}".format(args.ip, args.port))
    except RuntimeError:
        print ("\nCan't connect to Naoqi at IP {} (port {}).\nPlease check your script's arguments."
               " Run with -h option for help.\n".format(args.ip, args.port))
        sys.exit(1)
    main(session)
