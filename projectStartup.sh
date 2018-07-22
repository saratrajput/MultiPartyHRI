guake -n "cd ~/multiPartyHRI/pepper" -r pepperInit;
guake -e "vim pepperInit.py"

guake -n "cd ~/multiPartyHRI/pepper" -r py2pepperInit;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI/pepper";
guake -e "python"

guake -n "cd ~/multiPartyHRI/pepper" -r faceRecServer;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI/pepper"
guake -e "vim pepperFacerecServer.py"

guake -n "cd ~/multiPartyHRI/pepper" -r dialogServer;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI/pepper"
guake -e "vim pepperDialogServer.py"

guake -n "cd ~/multiPartyHRI/kinect" -r humanTopicSave;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI/kinect"
#guake -e "python humanTopicSave.py"

guake -n "cd ~/multiPartyHRI" -r speechRec;
guake -e "source ~/myEnvPy2/bin/activate"
guake -e "cd ~/multiPartyHRI"
#guake -e "python speechRecognitionToDialogflow.py"

guake -n "cd ~/multiPartyHRI" -r ipWebcam;
guake -e "source activate myEnvPy3"
guake -e "cd ~/multiPartyHRI"
#guake -e "python knnIpWebcamAlternate.py"
