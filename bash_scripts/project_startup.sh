# Edit Pepper's IP and Port
guake -n "cd ~/multiPartyHRI" -r robotIpPort # Folder where project files are kept
guake -e "vim ~/multiPartyHRI/robotIpPort.txt"

# Pepper Init
guake -n "cd ~/multiPartyHRI/pepper" -r py2pepperInit;
guake -e "source ~/myEnvPy2/bin/activate" # instead of myEnvPy2 enter your virtual environment name for python 2.7
guake -e "cd ~/multiPartyHRI/pepper";
guake -e "python"

# Face recognition processing
guake -n "cd ~/multiPartyHRI" -r ipWebcam;
guake -e "source activate myEnvPy3"
guake -e "cd ~/multiPartyHRI"
#guake -e "python knnIpWebcamAlternate.py"

# Pepper Face Recognition NEP Server
guake -n "cd ~/multiPartyHRI/pepper" -r faceRecServer;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "vim ~/multiPartyHRI/pepper/pepperFacerecServer.py"

# Speech Recognition and Dialogflow processer
guake -n "cd ~/multiPartyHRI" -r speechRec;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI"
#guake -e "python speechRecognitionToDialogflow.py"

# Pepper Speech Recognition NEP Server
guake -n "cd ~/multiPartyHRI/pepper/" -r dialogServer;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI/pepper/"
guake -e "vim pepperDialogServer.py"
#guake -e python pepperDialogServer.py

# Save data from Kinect
guake -n "cd ~/multiPartyHRI/kinect" -r humanTopicSave
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI/kinect"
#guake -e "python humanTopicSave.py"
