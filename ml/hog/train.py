import numpy as np
import cv2 as cv

from read import get_files
from svm import SVM
from model import Hog

def train():
    hog=Hog()
    hog.train()
    hog.test()


train()

