import os
from PIL import Image
import pyocr
import cv2
import img_processing2
#環境変数「PATH」にTesseract-OCRのパスを設定。
#Windowsの環境変数に設定している場合は不要。
path='C:\\Program Files\\Tesseract-OCR\\'
os.environ['PATH'] = os.environ['PATH'] + path

#pyocrにTesseractを指定する。
pyocr.tesseract.TESSERACT_CMD = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tools = pyocr.get_available_tools()
tool = tools[0]

#文字を抽出したい画像のパスを選ぶ
img = Image.open('./bef.png')

#画像の文字を抽出
builder = pyocr.builders.TextBuilder(tesseract_layout=6)
text = tool.image_to_string(img, lang="Eng", builder=builder)

print(text)

img = cv2.imread("./realtimeimg.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
frame = img_processing2.arrow_exist_judge(img)
cv2.imshow("kk",frame)
cv2.waitKey(0)