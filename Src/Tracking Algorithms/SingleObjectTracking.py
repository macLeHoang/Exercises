import cv2 as cv
import imutils
import time
import numpy as np

BACKGROUND_SUBTRACTION = {
    "mog2" : cv.createBackgroundSubtractorMOG2(),
    "knn" : cv.createBackgroundSubtractorKNN(),
    "mog" :cv.bgsegm.createBackgroundSubtractorMOG()
                        }

OBJECT_TRACKER = {
    "kcf" : cv.legacy.TrackerKCF_create(),
    "csrt" : cv.legacy.TrackerCSRT_create()
                  }

trackers = cv.legacy.MultiTracker_create()

color = {
    "kcf" : (0, 255, 0),
    "csrt" : (255, 0, 255)
}

def find_contour(img, origin_img, potential_box = []):
    contours, hierachy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if 350 < area:
            # cv.drawContours(origin_img, contour, -1, (255, 255, 0), 3)
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.2*perimeter, True)
            x,y,w,h = cv.boundingRect(approx)
            potential_box.append([x, y, w, h])
            # cv.rectangle(origin_img, (x,y), (x+w, y+h), (255, 255, 0), 1)
    
    return potential_box

def background_subtraction(img, type = 'knn'):
    mask = BACKGROUND_SUBTRACTION[type].apply(img)
    return mask

kernel = np.ones(shape = (3,3), dtype = np.uint8)

vid = cv.VideoCapture("./panoramic_fisheye_1.mp4")

count_frame = 0
while True:
    _, frame = vid.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width = 650)

    background_mask = background_subtraction(frame, 'knn')

    erode_image = cv.erode(background_mask, kernel, iterations = 2)
    dilate_image = cv.dilate(erode_image, kernel, iterations = 4)
    inrange_image = cv.inRange(dilate_image, 128, 255)

    pbox = find_contour(inrange_image, frame)

    cv.putText(frame, f'{count_frame}', (0,20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    success, boxes = trackers.update(frame)
    i = 0
    for box in boxes:
        x,y,w,h = [int(v) for v in box]
        if i%2 == 0:
            cv.rectangle(frame, (x,y), (x+w, y+h), color['kcf'], 1)
            cv.putText(frame, "KCF", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, color['kcf'])
        else:
            cv.rectangle(frame, (x,y), (x+w, y+h), color['csrt'], 1)  
            cv.putText(frame, "CSRT", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, color['csrt'])
        i += 1

    if count_frame == 341:
        tracker_1 = OBJECT_TRACKER['kcf']
        tracker_2 = OBJECT_TRACKER['csrt']
        for box in pbox:
            trackers.add(tracker_1, frame, box)
            trackers.add(tracker_2, frame, box)
        
    if count_frame == 343:
        cv.waitKey(3000)

    count_frame += 1
    pbox.clear()

    cv.imshow ("Origin", frame)
    cv.imshow ("process frame", inrange_image)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()   
