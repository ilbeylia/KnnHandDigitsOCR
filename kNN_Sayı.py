import cv2 as cv
import numpy as np

def createKnn():
    knn = cv.ml.KNearest_create()
    img = cv.imread('digits.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
    train = np.array(cells).reshape(-1,400).astype(np.float32) 
    trainLabel = np.repeat(np.arange(10),500)
    return knn, train, trainLabel


def updateKnn(knn, train, trainLabel, newData=None, newDataLabel=None):
    if (newData is not None) and (newDataLabel is not None):
        print(train.shape, newData.shape)
        newData = newData.reshape(-1,400).astype(np.float32)
        train = np.vstack((train,newData))   
        trainLabel = np.hstack((trainLabel,newDataLabel))   
    knn.train(train,cv.ml.ROW_SAMPLE,trainLabel)
    return knn, train, trainLabel



def imgSet(img):
    rois=[]
    imgGray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgBlur= cv.GaussianBlur(imgGray,(11,11),0)
    imgThresh = cv.adaptiveThreshold(imgBlur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,5,1)
    kontur,_=cv.findContours(imgThresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) #birincisi kaynak görüntü, ikincisi kontur alma modu, üçüncüsü kontur yaklaşımı metodu
    for k in kontur:
        (x,y,w,h)=cv.boundingRect(k)
        rois.append((x,y,w,h))
        if cv.contourArea(k) > 150:
            cv.rectangle(imgThresh, (x - 20, y - 10), (w + x + 20, h + y + 10), (255, 255, 255), 1)
    return rois, imgThresh
