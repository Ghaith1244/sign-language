import cv2
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
import numpy as np

offset = 20
imgSize = 300

counter = 0
folder = "Data/A"

while True:

    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, W, h = hand['bbox']


        imgCrop = img[ y-offset:y+ h + offset, x-offset:x + W +offset]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255


        imgWhite[0:imgCrop.shape[0], 0:imgCrop.shape[1]] = imgCrop

        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgWhite)
    
    
    
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
            print(counter)
        