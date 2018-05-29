import face_recognition
import cv2

################################################################################
# For using IP web-camera 
################################################################################
import numpy as np
import urllib.request
import time

################################################################################
# For knn trained model
################################################################################
import math
from sklearn import neighbors
import pickle
################################################################################

# Get the url of the image address: Read the steps explained in README
url = "http://192.168.1.104/ccm/ccm_pic_get.jpg?hfrom_handle=887330&dsess=1&dsess_nid=MPuYbzOdQTBJrrAPJdM.8YxCDdRhAw&dsess_sn=1jfiegbqeabqq&dtoken=p0_xxxxxxxxxx"

################################################################################
# Predict: Taken from face_recognition_knn.py
################################################################################
def predict(inputImage, knn_clf=None, model_path=None, distance_threshold=0.6):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(inputImage)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(inputImage, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
################################################################################

# Initialize some variables
processThisFrame = 0 # To skip processing alternating frames

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
        small_frame = cv2.resize(frame, None, fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]
    
    # Only process every other frame of video to save time
    if processThisFrame == 0:
        matches = predict(rgb_small_frame, model_path="trained_knn_model.clf")
        processThisFrame = 4
    else:   processThisFrame = processThisFrame - 1 
    # Loop through each face in this frame of video
    for name, (top, right, bottom, left) in matches:
        # Draw a box around the face
        cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(small_frame, (left, bottom - 5), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(small_frame, name, (left + 6, bottom - 6), font, 0.4, (255, 255, 255), 1)

#    else:   processThisFrame = processThisFrame - 1 

    # Display the resulting image
    cv2.imshow('Video', small_frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
