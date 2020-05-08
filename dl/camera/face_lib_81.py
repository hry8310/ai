import cv2
import dlib
 
path = "dataset/h3.jpg"
img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
 
predictor = dlib.shape_predictor(
    "E:/ai/dlib/shape_predictor_81_face_landmarks.dat"
)
 
dets = detector(gray, 1)
for face in dets:
    shape = predictor(img, face)   
    
    for pt in shape.parts():
        print(pt)
        pt_pos = (pt.x, pt.y)
        cv2.circle(img, pt_pos, 3, (0, 255, 0), 3)
    cv2.imwrite("./out/test4.jpg", img)
