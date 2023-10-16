import pickle
import cv2
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np


#横方向のProjection profileを得る
def Projection_V(img,h,w):
    array_V = np.full(w,h)
    count = [np.count_nonzero(img[:,i:i+1]) for i in range(w)]
    count = np.asarray(count,dtype=int)
    array_V = array_V - count
    return array_V

#Projection profileから横方向の座標を得る
def Detect_WidthPosition(W_THRESH, width, array_V):
    char_List = np.array([])
 
    flg = False
    posi1 = 0
    posi2 = 0
    for i in range(width):
        val = array_V[i]
        if (flg==False and val < W_THRESH):
            flg = True
            posi1 = i

        if (flg == True and val >= W_THRESH):
            flg = False
            posi2 = i
            char_List = np.append(char_List, posi1)
            char_List = np.append(char_List, posi2)
 
    return char_List


def preprocess_new_image(img):
  pixel_values_list = []
  #グレースケール化
  #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  #サイズ統一
  #image = cv2.resize(img, (26, 36))
  #画像のピクセル値を特徴量として1次元配列に変換
  pixels = img.flatten()
  pixel_values_list.append(pixels)
  return pixel_values_list


def load_model():
    with open("CharacterModel/3dprinter_model.pickle",mode="rb") as fp:
        model = pickle.load(fp)
    with open("CharacterModel/3dprinter_scaler.pickle",mode="rb") as fp:
        scaler = pickle.load(fp)
    with open("CharacterModel/3dprinter_pca.pickle",mode="rb") as fp:
        pca = pickle.load(fp)
    return model,scaler,pca


def predict(model,scaler,pca,img):
    image = preprocess_new_image(img)
    image_scaled = scaler.transform(image)
    image_pc = pca.transform(image_scaled)
    prediction = model.predict(image_pc[-1].reshape(1, -1))
    prediction = str(prediction[0])
    return prediction

def TextRecog(model,scaler,pca,frame):
    #対象画像をリサイズ
    #対象画像をグレイスケール化
    #gray_img = cv2.cvtColor(syaei_resize_img,cv2.COLOR_BGR2GRAY)
    #二値画像へ
    #ret, img_mask = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
    img_mask = frame
    #img_mask = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,7,-3)
    #ノイズ除去
    #img_mask = cv2.medianBlur(img_mask,3)
    #膨張化
    #img_mask = cv2.dilate(img_mask,kernel)
    #高さ、幅を保持
    kernel = np.ones((3,3),np.uint8)
    height,width = img_mask.shape
    img_mask = cv2.erode(img_mask,kernel,iterations = 1)
    #img_mask = cv2.erode(img_mask,np.ones((5,5),np.uint8),iterations=1)
    #cv2.imshow("ll",img_erode)
    #cv2.waitKey(0)
    out_modify = "" #修正したテキスト
    #横方向のProjection Profileを得る
    array_V = Projection_V(img_mask,height,width)
    W_THRESH = max(array_V)
    char_List2 = Detect_WidthPosition(W_THRESH,width,array_V)
    out_modify = "" #修正したテキスト
    out = "" #読み取ったテキスト
    print(char_List2)
    img_mask = cv2.dilate(img_mask,kernel,iterations = 1)
    
    for j in range(0,len(char_List2)-1,2):
        #end_time = time.perf_counter()
        #print(end_time-start_time)
        #一文字ずつ切り取る
        match_img = img_mask[:,int(char_List2[j])-1:int(char_List2[j+1])+1]
        #cv2.imwrite("match.jpg",match_img)
        try:
            match_img = cv2.resize(match_img,dsize=(26,36))
            #cv2.imshow("kk",match_img)
            #cv2.waitKey(0)
        except cv2.error:
            return ""
        height_m,width_m = match_img.shape
        cv2.imshow("kk",match_img)
        cv2.waitKey(0)
        cv2.imwrite("match{0}.jpg".format(j),match_img)

        prediction = predict(model,scaler,pca,match_img)
        print(prediction)
        if (j != 0) & (char_List2[j] > (width_m + char_List2[j-1])):
            if (j+1) == len(char_List2)-1:
                out_modify = out_modify+ ' ' + prediction
                out = out + out_modify + ' '
                #output_text.append('\n')
                out_modify = ""
                continue
                #out_modify = speling.correct(out_modify)
                #out_modify += label_temp[new_d[0][1]]
            out_modify += ' '
                #out = out + out_modify
                #output_text.append(' ')
                #output_text.append(out_modify)
                #print(out_modify)
                #out_modify = ""
        #行の最後の時
        if (j+1) == len(char_List2)-1:
            out_modify = out_modify + prediction    
            #out_modify = speling.correct(out_modify)
            out = out + out_modify + ' '
            #output_text.append('\n')
            out_modify = ""
            continue
        #print(label_temp[new_d[0][1]])
        out_modify = out_modify + prediction
        #print(out_modify)
        continue
    
    return out

if __name__ == "__main__":
     print("Scikit-learnバージョン:", sklearn.__version__)

     img = cv2.imread("before_frame_row1.jpg")
     img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
     model,scaler,pca = load_model()
     text = TextRecog(model,scaler,pca,img)
     print(text)
     cv2.imshow("kk",img)
     cv2.waitKey(0)