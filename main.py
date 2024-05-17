from ultralytics import YOLO
import cv2

import util
from sort import *
from util import get_car, read_license_plate, write_csv
import torch

results = {}

mot_tracker = Sort()

# load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
coco_model = YOLO('yolov8l.pt')
coco_model.to(device)
license_plate_detector = YOLO('./models/license_plate_detector.pt')

# load video
# cap = cv2.VideoCapture('./170609_A_Delhi_026.mp4')
# cap = cv2.VideoCapture('./sample_old1.mp4')
cap = cv2.VideoCapture('./car1.jpeg')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        try:
            track_ids = mot_tracker.update(np.asarray(detections_))
        except:
            track_ids = []
        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                cv2.imshow('license_plate_crop', license_plate_crop_gray)
                cv2.imwrite('license_plate_crop.jpg', license_plate_crop_gray)
                # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                license_plate_crop_thresh = cv2.Canny(license_plate_crop_gray, 30, 200)

                cv2.imshow('license_plate_crop_thresh', license_plate_crop_thresh)
                # cv2.imwrite('license_plate_crop_thresh.jpg', license_plate_crop_thresh)
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_gray)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
                    
                    # rectrangle around car number plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    # rectrangle around car
                    cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0), 2)
                    # put text on car number plate
                    cv2.putText(frame, license_plate_text, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # put text on car
                    cv2.putText(frame, str(car_id), (int(xcar1), int(ycar1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# print(results)
# write results
write_csv(results, './test.csv')