import cv2
from segmentation import segmentation
from localize1 import localize1
import csv

with open('Output.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['License plate', 'Frame no.', 'Timestamp(seconds)'])
cap = cv2.VideoCapture('trainingsvideo.avi')
cont = True
# frameNum, Current position of the video file in seconds, printed best Value
data = [1, 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1]
plates = []
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    data[1] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
    cropped = localize1(frame)
    if cropped is None:
        data[0] += 1
        continue
    plates = segmentation(cropped, data, plates)
    data[0] += 1
cap.release()
cv2.destroyAllWindows()
