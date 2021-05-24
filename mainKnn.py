import cv2 as cv
import numpy as np
import kNN_Sayı as kNN

img= cv.imread('testData1.png')
knn, train, trainLabel = kNN.createKnn()
knn, train, trainLabel = kNN.updateKnn(knn, train, trainLabel)

rois, imgThresh = kNN.imgSet(img)
digits=[]
for r in rois:
    x, y, w, h = r
    if (w > 10) & (h > 10):
        Ri = imgThresh[y - 10:y + h + 10, x - 20:x + w + 20]
        Ri = cv.resize(Ri, (20, 20), interpolation=cv.INTER_AREA)  # bu işlemi img set içinde yap burda olmuyor enden anlamadım
        # digits.append(Ri)
        # if int(w*h)
        outData = Ri.reshape(-1, 400).astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(outData, k=5)
        result = int(result[0][0])
        cv.putText(img,str(result),(x,y-20),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv.imshow("result",img)
        print("sonuc",result)
cv.waitKey(0)
cv.destroyAllWindows()