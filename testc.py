import multiprocessing
from re import subn
from cv2 import imwrite
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import difflib
import time
import cv2
import numpy as np
import glob
from natsort import natsorted
import multiprocessing
from PIL import Image , ImageTk , ImageOps
import pyttsx3 
from dictionary_word import speling
import difflib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import image_processing
import audio_output
from sklearn.neighbors import NearestNeighbors 
from io import BytesIO
import cython

import mymodule
img = cv2.imread("./camera1/camera2.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
h,w = img.shape
a = 4
b = 10
c = mymodule.add(a,b)
print(c)
img2 = img_processing2.Projection_V(img)
print(a)