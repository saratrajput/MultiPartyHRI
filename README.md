# MultiPartyHRI
MasterThesis code

=================================== Pepper Init ===================================
This script contains all the basic proxies required to run Pepper.
One can use it to put Pepper in Rest mode or Wake Up Mode.

Start Python2.7 in one terminal and import the script using:
from peppperInit import *

To Stop All Behaviours:
behaviourProxy.stopAllBehaviors()

To Put Pepper in Rest Mode:
motionProxy.rest()

To Wake Up Pepper:
motionProxy.wakeUp()

====================Using IP Webcamera with Face Recognition:====================

- Connect the IP Webcamera to ethernet cable of your network.
- Install an app like "MIPC" on your android device.
- Make sure your android device is on the same network as the IP camera.
- Connect the camera and the android device.(The steps should be explained in 
  the app.)
- Go in the network section of the app and find the IP address of the webcam.
- Type in this IP address in your browser and type in the password.
  (Password should be written on the bulletin board of lab.)
Following steps can be done on android device but it's easier on the PC:
- Once login is successfull in the browser, it asks to connect with the wifi. 
- Connect the device to the desired wifi, preferably a private wifi with not 
  many users, such as your own hotspot device. (Caution: It does not connect 
  to lab wifi due to many users using it.)
- Once this connection is done, make sure your PC and wifi device are on the same
  wifi network. (Connect PC to wifi with an external wifi device.)
- Go to the home section in the browser -> right click on the feed -> Copy image 
  address -> Paste in the url section of the script: webcam.py

NOTE: If you have connectivity issues, make sure your router is using 2.4 GHz 
	  as the IP-Webcam works with only 2.4 GHz.
================================================================================

============================== Speech Recognizer ==============================
------------------------- To Change Microphone ------------------------------
import speech_recognition as sr
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print("Microphone with name \"{1}\" found for `Microphone(device_index={0})`".format(index, name))

- You get some output along with device_indices.
- To use a specific microphone, change Microphone to Microphone(device_index=<number>)
--------------------------------------------------------------------------------
