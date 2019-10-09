from imutils import paths
from sklearn.cluster import DBSCAN
from imutils import build_montages

import face_recognition
import numpy as np
import argparse
import pickle
import cv2
import os

img_path='all'
enc_path='./pkl/dsface.pkl'

def face_cfg():
    if os.path.exists(enc_path):
        data = pickle.loads(open(enc_path, "rb").read())
        data = np.array(data)
        return data
  
    imagePaths = list(paths.list_images(img_path))
    data = []

    for (i, imagePath) in enumerate(imagePaths):
        print("img  {}".format(i + 1))
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,
		model='cnn')
        encodings = face_recognition.face_encodings(rgb, boxes)
 
        d = [{"imagePath": imagePath, "box": box, "enc": enc}
           for (box, enc) in zip(boxes, encodings)]
        data.extend(d)

    f = open(enc_path, "wb")
    f.write(pickle.dumps(data))
    f.close()
    return data


def db_clu(data):
    encs = [d["enc"] for d in data]

    clt = DBSCAN(metric="euclidean",eps=0.5,min_samples=2)
    clt.fit(encs)

    print(clt.labels_)
    labelIDs = np.unique(clt.labels_)
    print(labelIDs)
    numUniqueFaces = len(np.where(labelIDs > -1)[0])
    print("[INFO] # unique faces: {}".format(numUniqueFaces))

    for labelID in labelIDs:
        print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(25, len(idxs)),
            replace=False)
        faces = []
        for i in idxs:
            image = cv2.imread(data[i]["imagePath"])
            (top, right, bottom, left) = data[i]["box"]
            face = image[top:bottom, left:right]

            face = cv2.resize(face, (96, 96))
            faces.append(face)
        montage = build_montages(faces, (96, 96), (5, 5))[0]
        cv2.imwrite('./dbout/lab_'+str(labelID+1)+'_img.jpg', montage)

data=face_cfg()
db_clu(data)

