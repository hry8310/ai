import dlib

import os
import sys

import numpy as np
import cv2



landmark_file = "/root/dlib/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmark_file)


def affineTrans(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def calDelaunayTriang(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList();

    delaunayTri = []

    pt = []

    for t in triangleList:

        ind = []
        # Get face-points (from 68 face detector) by coordinates
        for j in range(0, 3):
            for k in range(0, len(points)):
                if (abs(t[j*2] - points[k][0]) < 1.0 and abs(t[j*2+1] - points[k][1]) < 1.0):
                    ind.append(k)
        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri






def cpToImg(srcImg, tgimg, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    rect = srcImg[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    rect = affineTrans(rect, t1Rect, t2Rect, size)

    rect = rect * mask

    tgimg[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = tgimg[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    tgimg[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = tgimg[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + rect



def shape_to_np(shape, scale):
    coords = []
    for i in range(0, 68):
        coords.append((int(shape.part(i).x / scale), int(shape.part(i).y / scale)))
    return coords


def facePoints(img):
    scale = 200 / min(img.shape[0], img.shape[1])
    thumb = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    face_rects = face_detector(gray, 1)
    for i, rect in enumerate(face_rects):
        shape = predictor(gray, face_rects[i])
        return shape_to_np(shape, scale)

def i_f(file):
    return './test/'+file

def o_f(file):
    return './otest/'+file

def getHull(srcImg,toImg):
    points1 = facePoints(srcImg)
    points2 = facePoints(toImg)

    # Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])
    return hull1,hull2 

def exImg(srcImg,toImg):
    tgimg = np.copy(toImg);
    hull1,hull2=getHull(srcImg,toImg)

    size = toImg.shape
    rect = (0, 0, size[1], size[0])
    dt = calDelaunayTriang(rect, hull2)
    if len(dt) == 0:
        return None,None,None
    for i in range(0, len(dt)):
        t1 = []
        t2 = []

        for j in range(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        cpToImg(srcImg, tgimg, t1, t2)
        #cv2.imwrite(o_f('hull1_'+str(i)+'.jpg'), tgimg)
    return hull1,hull2,tgimg


def run():

    srcImg = cv2.imread(i_f('wzl.jpg'));
    toImg = cv2.imread(i_f('hxm.jpg'));

    hull1,hull2,tgimg=exImg(srcImg,toImg)


    tghull = []
    for i in range(0, len(hull2)):
        tghull.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(toImg.shape, dtype=toImg.dtype)
    cv2.fillConvexPoly(mask, np.int32(tghull), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull2]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    output = cv2.seamlessClone(np.uint8(tgimg), toImg, mask, center, cv2.NORMAL_CLONE)

    cv2.imwrite(o_f('out.jpg'), output)

    cv2.waitKey(0)


run()
