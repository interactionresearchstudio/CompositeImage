import cv2
import json
import time
import datetime
import numpy as np

# configuration file
config = json.load(open("config.json"))
# end of configuration file

# window
cv2.namedWindow("Output")
# end of window

# camera
capture = cv2.VideoCapture(0)
capture.set(3, 320)
capture.set(4, 240)
if capture.isOpened():
    rval, frame = capture.read()
else:
    rval = False
time.sleep(config["camera_warmup"])
# end of camera

avg = None

firstImage = None

# main loop
while rval:
    # new frame
    rval, image = capture.read()
    # end of new frame

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    if avg is None:
        avg = gray.copy().astype("float")
        continue

    cv2.accumulateWeighted(gray, avg, 0.8)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 1, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    thresh_floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros(image.shape[:2], np.uint8)
    #cv2.floodFill(thresh_floodfill, mask, (0,0), 255)
    #thresh_floodfill_inverted = cv2.bitwise_not(thresh_floodfill)
    #thresh_output = thresh | thresh_floodfill_inverted

    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts:
        # do nothing
        largestContour = None
    else:
        areas = [cv2.contourArea(c) for c in cnts]
        max_index = np.argmax(areas)
        largestContour = cnts[max_index]

    #(x, y, w, h) = cv2.boundingRect(largestContour)
    #cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

    if largestContour is not None:
        #hull = cv2.convexHull(largestContour)
        #cv2.drawContours(image, cnts, max_index, (0,255,0), -1)
        cv2.drawContours(mask, cnts, max_index, 255, -1)
        #cv2.drawContours(image, [hull], -1, (0,0,255), 2)

    

    cv2.imshow("Output", image)
    cv2.imshow("thresh", thresh)
    cv2.imshow("mask", mask)
    cv2.imshow("change", frameDelta)
    if firstImage is not None:
        cv2.imshow("composite", firstImage)

    # wait for keys
    key = cv2.waitKey(10)
    if key == 27:
        break
    if key == ord("a"):
        avg = gray.copy().astype("float")
    if key == ord(" "):
        if firstImage is None:
            print("First image captured")
            firstImage = image.copy()
        else:
            locs = np.where(mask!=0)
            firstImage[locs[0], locs[1]] = image[locs[0], locs[1]]
            
    # end of loop

# cleanup
cv2.destroyWindow("Output")
