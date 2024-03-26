import cv2
def coordinates(event,x,y, flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
cap = cv2.VideoCapture(1)
count =1
while True:
    
    ret, frame = cap.read()
    if not ret:
        cv2.destroyAllWindows()
        
    cv2.imshow("frame",frame)
    count +=1
    key = cv2.waitKey(1) & 0XFF
    if key == ord('c'):
        cv2.imwrite("./dataset_make/calibration/testImage/cameraca.jpg".format(count),frame)
        count +=1
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
