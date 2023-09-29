import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from PIL import Image

files = glob.glob(r'C:\Users\Naoya Tagawa\Desktop\ex4\*')
files = natsorted(files)
p1 =np.zeros(2)
p2 =np.zeros(2)
p3 =np.zeros(2)
p4 =np.zeros(2)
count = 1
#167 130,190 129,168 170,192 170
#グレイスケール化
w = math.floor(((190-167)+(192-170))/2)
h =math.floor(((170-130)+(170-129))/2)

for f in files:
    img = cv2.imread(f)
    cv2.imwrite(r'C:\Users\Naoya Tagawa\Desktop\ex3\ex{0}.jpg'.format(count),img)
    count += 1
print(w,h)

#二値化とリサイズ
count = 1
for f in files:
    img = cv2.imread(f)
    #gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret , img_th = cv2.threshold(gray_img,130,255,cv2.THRESH_BINARY)
    #img_mask = cv2.medianBlur(img_th,3) #ノイズ除去
    
    #カーネル
    kernel = np.ones((3,3),np.uint8)

    #対象画像をグレイスケール化
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    #ノイズ除去
    img_mask = cv2.medianBlur(img_mask,3)
    #膨張化
    img_mask = cv2.dilate(img_mask,kernel)
    dst = cv2.resize(img_mask,dsize = (w,h))
    cv2.imwrite(r'C:\Users\Naoya Tagawa\Desktop\ex3\ex{0}.jpg'.format(count),dst)
    count += 1

#確認
datamake.read_image()
photos = np.load(r'C:\Users\Naoya Tagawa\OneDrive\dataset.npz')
x = photos['x']
y = photos['y']
idx =0
plt.figure(figsize=(10,10))
for i in range(70):
    plt.subplot(5,16,i+1)
    plt.axis('off')
    plt.title(y[i+idx])
    plt.imshow(x[i+idx],cmap='gray')
    #print(len(photos['x']))
plt.show()
plt.imshow(x[93],cmap='gray')
plt.show()
print(y[93])
