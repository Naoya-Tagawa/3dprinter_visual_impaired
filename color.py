import cv2
import img_processing2
import matplotlib.pyplot as plt
img2 = cv2.imread("./base10.jpg")

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img2],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
    #histr += histr




plt.savefig("hist9.png")
plt.show()