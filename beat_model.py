
import face_recognition
from google.colab.patches import cv2_imshow
import pickle
import cv2
import os
from imutils import paths

imagePath = list(paths.list_images('/content/drive/MyDrive/ML LAB/lfw_funneled'))
print(imagePath)
print("\n")
kEncodings = []
kNames = []
'''
# loop over the image paths
for (i, ip) in enumerate(imagePath):
    # extract the person name from the image path
    name = ip.split(os.path.sep)[-2]
    # load the input image and convert it from BGR
    image = cv2.imread(ip)
    #image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model='cnn')
    # compute the facial embedding for the any face
    encodings = face_recognition.face_encodings(rgb, boxes)
# loop over the encodings
    for encoding in encodings:
        kEncodings.append(encoding)
        kNames.append(name)

data = {"encodings": kEncodings, "names": kNames}
# use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))  # to open file in write mode
f.close()  # to close file

# to find path of xml file containing haarCascade file

'''

cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"


fc = cv2.CascadeClassifier(cfp)

# load the known faces and embeddings saved in last file
#data = pickle.loads(open('/content/face_enc', "rb").read())
data = pickle.loads(open('beat_management/face_enc', "rb").read())

# Find path to the image you want to detect face and pass it here
image = cv2.imread('beat_management/lfw/Prince_William/Prince_William_0001.jpg')
#image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.1)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert image to Greyscale for HaarCascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = fc.detectMultiScale(gray,
                            scaleFactor=1.1,
                            minNeighbors=6,
                            minSize=(60, 60),
                            flags=cv2.CASCADE_SCALE_IMAGE)
# the facial embeddings for face in input
encodings = face_recognition.face_encodings(rgb)
names = []


for encoding in encodings:
    # Compare encodings with encodings in data["encodings"]
    # Matches contain array with boolean values True and False
    matches = face_recognition.compare_faces(data["encodings"],
                                             encoding)
    # set name =unknown if no encoding matches
    name = "Unknown"
    # check to see if we have found a match
    if True in matches:
        # Find positions at which we get True and store them
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        count = {}
        # loop over the matched indexes and maintain a count for
        # each recognized face face
        for i in matchedIdxs:
            # Check the names at respective indexes we stored in matchedIdxs
            name = data["names"][i]
            # increase count for the name we got
            count[name] = count.get(name, 0) + 1
        # set name which has highest count
        name = max(count, key=count.get)
    # will update the list of names
    names.append(name)
# do loop over the recognized faces
for ((x, y, w, h), name) in zip(faces, names):
    # rescale the face coordinates
    # draw the predicted face name on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#cv2_imshow(image)
print(name)
cv2.waitKey(0)
