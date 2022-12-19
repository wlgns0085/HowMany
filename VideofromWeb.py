import cv2

cap = cv2.VideoCapture('http://192.168.234.16:8090/?action=stream')

#^^ Opened in new tab URL ^^
#cap = cv2.VideoCapture('http://345.63.46.1256/html/')

cv2.namedWindow('live cam', cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()
    #img_resize = cv2.resize(frame, (960, 540))
    cv2.imshow('live cam', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
