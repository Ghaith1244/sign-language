import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
import numpy as np

offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, W, h = hand['bbox']


        imgwwidth = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[ y-offset:y+ h + offset, x-offset:x + W +offset] 
        cv2.imshow("Image Crop", imgCrop)
        cv2.imshow("Image White", imgwwidth)
    
    
    
        cv2.imshow("Image", img)
        cv2.waitKey(1)