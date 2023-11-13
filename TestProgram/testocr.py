import os
from PIL import Image
import pyocr
import cv2
import ImageProcessing.img_processing2 as img_processing2
import numpy as np
#環境変数「PATH」にTesseract-OCRのパスを設定。
#Windowsの環境変数に設定している場合は不要。
path='C:\\Program Files\\Tesseract-OCR\\'
os.environ['PATH'] = os.environ['PATH'] + path
img = cv2.imread('./realtimeimg.jpg')


#kernel = np.ones((3,3),np.uint8)
#img = cv2.erode(img,kernel,iterations = 1)
#cv2.imwrite("realtimeimg.jpg",img)

#pyocrにTesseractを指定する。
pyocr.tesseract.TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tools = pyocr.get_available_tools()
tool = tools[0]
langs = tool.get_available_languages()
print(langs)
# 言語に日本語と今回の学習済みデータを指定
lang_setting = langs[0]+"+"+langs[3]
#文字を抽出したい画像のパスを選ぶ
img = Image.open('./mask.png')
#img = cv2.imread('./cameras.jpg')
#img = img_processing2.cut_blue_img2(img)
#p,img = img_processing2.mask_make(img)
#cv2.imwrite("heikou.png",img)
#画像の文字を抽出

builder = pyocr.builders.TextBuilder(tesseract_layout=6)
text = tool.image_to_string(img, lang=lang_setting, builder=builder)

print(text)

img = cv2.imread("./realtimeimg.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
frame = img_processing2.arrow_exist_judge(img)
cv2.imshow("kk",img)
cv2.waitKey(0)