# ============================= NEP ==============================  
import nep
import time
client = nep.client('127.0.0.1', 8010) #Create a new server instance
#===============================================================================
import face_recognition
import cv2

#========================= For using IP web-camera =========================
import numpy as np
import urllib.request
import time
#===============================================================================

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get the url of the image address: Read the steps explained in README
url = "http://192.168.1.104/ccm/ccm_pic_get.jpg?hfrom_handle=887330&dsess=1&dsess_nid=MOSU9zPfqXlrjakwY.ckomtCD6JhAw&dsess_sn=1jfiegbqeabqq&dtoken=p0_xxxxxxxxxx"

#===============================================================================
# To Do: Create a for loop to create the variables (Bad programming practice)
#imagePath = ".data/images/"
#labMembersList = [ 'suraj', 'enrique']
#
## Create arrays of known face encodings and their names
#known_face_encodings = [];
#
#for name in labMembersList:
#    name + "_image"  = face_recognition.load_image_file(imagePath + name + ".jpg")
#    name + "_image_encoding" = face_recognition.face_encodings(name + "_image")[0]
#===============================================================================

# Load Suraj's image and learn how to recognize it.
my_image = face_recognition.load_image_file("./data/images/myPhoto.jpg")
my_image_encoding = face_recognition.face_encodings(my_image)[0]

# Load Enrique's image and learn how to recognize it.
enrique_image = face_recognition.load_image_file("./data/images/enrique.jpg")
enrique_image_encoding = face_recognition.face_encodings(enrique_image)[0]

# Load Tomoya's image and learn how to recognize it.
tomoya_image = face_recognition.load_image_file("./data/images/tomoya.jpg")
tomoya_image_encoding = face_recognition.face_encodings(tomoya_image)[0]

# Load Hamied's image and learn how to recognize it.
hamied_image = face_recognition.load_image_file("./data/images/hamied.jpg")
hamied_image_encoding = face_recognition.face_encodings(hamied_image)[0]

# Load Fawzi's image and learn how to recognize it.
fawzi_image = face_recognition.load_image_file("./data/images/fawzi.jpg")
fawzi_image_encoding = face_recognition.face_encodings(fawzi_image)[0]

# Load Misato's image and learn how to recognize it.
misato_image = face_recognition.load_image_file("./data/images/misato.jpg")
misato_image_encoding = face_recognition.face_encodings(misato_image)[0]

# Load Taka's image and learn how to recognize it.
taka_image = face_recognition.load_image_file("./data/images/taka.jpg")
taka_image_encoding = face_recognition.face_encodings(taka_image)[0]

# Load Raida's image and learn how to recognize it.
raida_image = face_recognition.load_image_file("./data/images/raida.jpg")
raida_image_encoding = face_recognition.face_encodings(raida_image)[0]

# Load Shohei's image and learn how to recognize it.
shohei_image = face_recognition.load_image_file("./data/images/shohei.jpg")
shohei_image_encoding = face_recognition.face_encodings(shohei_image)[0]

# Load Venture Sensei's image and learn how to recognize it.
ventureSensei_image = face_recognition.load_image_file("./data/images/ventureSensei.jpg")
ventureSensei_image_encoding = face_recognition.face_encodings(ventureSensei_image)[0]

# Load Liz Sensei's image and learn how to recognize it.
lizSensei_image = face_recognition.load_image_file("./data/images/lizSensei.jpg")
lizSensei_image_encoding = face_recognition.face_encodings(lizSensei_image)[0]

# Load Takamune's image and learn how to recognize it.
takamune_image = face_recognition.load_image_file("./data/images/takamune.jpg")
takamune_image_encoding = face_recognition.face_encodings(takamune_image)[0]

# Load akira's image and learn how to recognize it.
akira_image = face_recognition.load_image_file("./data/images/akira.jpg")
akira_image_encoding = face_recognition.face_encodings(akira_image)[0]

# Load kazushi's image and learn how to recognize it.
kazushi_image = face_recognition.load_image_file("./data/images/kazushi.jpg")
kazushi_image_encoding = face_recognition.face_encodings(kazushi_image)[0]

# Load kei's image and learn how to recognize it.
kei_image = face_recognition.load_image_file("./data/images/kei.jpg")
kei_image_encoding = face_recognition.face_encodings(kei_image)[0]

# Load kodai's image and learn how to recognize it.
kodai_image = face_recognition.load_image_file("./data/images/kodai.jpg")
kodai_image_encoding = face_recognition.face_encodings(kodai_image)[0]

# Load ren's image and learn how to recognize it.
ren_image = face_recognition.load_image_file("./data/images/ren.jpg")
ren_image_encoding = face_recognition.face_encodings(ren_image)[0]

# Load seiji's image and learn how to recognize it.
seiji_image = face_recognition.load_image_file("./data/images/seiji.jpg")
seiji_image_encoding = face_recognition.face_encodings(seiji_image)[0]

# Load takuya's image and learn how to recognize it.
takuya_image = face_recognition.load_image_file("./data/images/takuya.jpg")
takuya_image_encoding = face_recognition.face_encodings(takuya_image)[0]

# Load tatsuki's image and learn how to recognize it.
tatsuki_image = face_recognition.load_image_file("./data/images/tatsuki.jpg")
tatsuki_image_encoding = face_recognition.face_encodings(tatsuki_image)[0]
#===============================================================================

# Create arrays of known face encodings and their names
known_face_encodings = [
    my_image_encoding,
    enrique_image_encoding,
    tomoya_image_encoding,
    hamied_image_encoding,
    fawzi_image_encoding,
    misato_image_encoding,
    taka_image_encoding,
    raida_image_encoding,
    shohei_image_encoding,
    ventureSensei_image_encoding,
    lizSensei_image_encoding,
    takamune_image_encoding,
    akira_image_encoding,
    kazushi_image_encoding,
    kei_image_encoding,
    kodai_image_encoding,
    ren_image_encoding,
    seiji_image_encoding,
    takuya_image_encoding,
    tatsuki_image_encoding

]
known_face_names = [
    "Suraj",
    "Enrique",
    "Tomoya",
    "Hamied",
    "Fawzi",
    "Misato",
    "Taka",
    "Raida",
    "Shohei",
    "Venture Sensei",
    "Liz Sensei",
    "Takamune",
    "Akira",
    "Kazushi",
    "Kei",
    "Kodai",
    "Ren",
    "Seiji",
    "Takuya",
    "Tatsuki"


    
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    try:
        imgResp=urllib.request.urlopen(url)
    except:
        print("No input image")
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgNp, -1)

    # Resize frame of video to 1/4 size for faster face recognition processing
    if frame is not None: # To skip processing if there is an error or delay getting new image.
    #    small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        name = "Unknown" # Declare name here to access out of for loop

    # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

        # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

#============================== NEP ============================== 
        msg = name # Message to send as request
        client.send_info(msg)   # Send request
        client.listen_info()
        time.sleep(1) # Wait one second
#===============================================================================
    if frame is not None:
        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
