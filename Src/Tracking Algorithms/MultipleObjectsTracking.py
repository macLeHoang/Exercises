import cv2 as cv
import imutils
import numpy as np

BACKGROUND_SUBTRACTION = {
    "mog2" : cv.createBackgroundSubtractorMOG2(),
    "knn" : cv.createBackgroundSubtractorKNN(),
    "mog" :cv.bgsegm.createBackgroundSubtractorMOG()
                        }

OBJECT_TRACKER = {
    "kcf" : cv.legacy.TrackerKCF_create,
    "csrt" : cv.legacy.TrackerCSRT_create
                  }

multi_tracker = cv.legacy.MultiTracker_create()

color = {
    "kcf" : (0, 255, 0),
    "csrt" : (255, 0, 255)
}

data = {}

def find_contour(img, origin_img):
    potential_box = []

    contours, hierachy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            # cv.drawContours(origin_img, contour, -1, (255, 255, 0), 3)
            perimeter = cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, 0.2*perimeter, True)
            x,y,w,h = cv.boundingRect(approx)
            if w*h > 100 and w*h < 4500:
                potential_box.append([x, y, w, h])
                cv.rectangle(origin_img, (x,y), (x+w, y+h), (255, 255, 0), 1)
    
    return potential_box

def background_subtraction(img, type = 'knn'):
    mask = BACKGROUND_SUBTRACTION[type].apply(img)
    return mask

def find_new_object(box_1, box_2, threshold = 0.9): 
    #box 1 contains bounding box used in update tracking
    #box 2 contains boundning box just detected
    if len(box_2) > 0:
        box_1 = np.asarray(box_1)               
        box_2 = np.asarray(box_2) 

        x_1 = np.asarray([box_1[:, 0] + box_1[:, 2] / 2])
        y_1 = np.asarray([box_1[:, 1] + box_1[:, 3] / 2])

        x_2 = np.asarray([box_2[:, 0] + box_2[:, 2] / 2])
        y_2 = np.asarray([box_2[:, 1] + box_2[:, 3] / 2])

        distance = x_1**2 + y_1**2 + (x_2**2).T + (y_2**2).T - 2*(x_2).T * x_1 - 2*(y_2).T *y_1

        object_to_tracking_box_min_distance = np.argmin(distance, axis = 1)

        idx_new_object = [i for i in range(len(object_to_tracking_box_min_distance)) 
                            if distance[i, object_to_tracking_box_min_distance[i]] > threshold]

        return box_2[idx_new_object]

    else: 
        return box_2
 
def opening_by_reconstruction(img, kernel, n = 2):
    erode_img = cv.erode(img, kernel, iterations = n)
    reconstruct_img = erode_img

    while True:
        dilate_img = cv.dilate(reconstruct_img, kernel)
        tmp_img = cv.min(dilate_img, img)

        mask = (tmp_img == reconstruct_img)
        if False not in mask:
            break

        reconstruct_img = tmp_img

    return reconstruct_img

kernel = np.ones(shape = (3,3))

count_frame = 1
object_flag = 0 #check if already has object being tracked

vid = cv.VideoCapture('./panoramic_fisheye_2.mp4')
while True:
    _, frame = vid.read()
    if frame is None:
        break

    frame = imutils.resize(frame, width = 650)

    background_mask = background_subtraction(frame, 'knn')
    reconstruct_img = opening_by_reconstruction(background_mask, kernel, 2)
    # inrange_img = cv.inRange(reconstruct_img, 128, 255)

    success, boxes = multi_tracker.update(frame)
    i = 0
    for box in boxes:
        x,y,w,h = [int(v) for v in box]
        cv.rectangle(frame, (x,y), (x+w, y+h), color['csrt'], 1)  
        cv.putText(frame, f"id_{i}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, color['kcf'])
        i+=1

    # p_boxes = find_contour(reconstruct_img, frame)
    if count_frame % 25 == 0:
        p_boxes = find_contour(reconstruct_img, frame)
        if object_flag == 0:
            for box in p_boxes:
                object_flag = 1
                try:
                    tracker = OBJECT_TRACKER['kcf']()
                    multi_tracker.add(tracker, frame, box)
                except Exception as e:
                    print(e)
        else:
            new_object = find_new_object(boxes, p_boxes, threshold = 150.0)
            for box in new_object:
                try:
                    tracker = OBJECT_TRACKER['csrt']()
                    multi_tracker.add(tracker, frame, box)
                except Exception as e:
                    print(e)

        p_boxes.clear()

    count_frame += 1
    cv.imshow ("Origin", frame)
    cv.imshow ("process frame", reconstruct_img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()   
