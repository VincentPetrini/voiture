import cv2
import numpy as np

videoName = "road_car_view2.mp4"
video = cv2.VideoCapture(videoName)

while True:
   ret, orig_frame = video.read()
   if not ret:
       video = cv2.VideoCapture(videoName)
       continue
   frame = cv2.GaussianBlur(orig_frame, (7, 7), 0)
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   low_white = np.array([190, 190, 190])
   up_white = np.array([216, 216, 216])
   low_yellow = np.array([18, 94, 140])
   up_yellow = np.array([48, 255, 255])
   mask = cv2.inRange(hsv, low_yellow, up_yellow)
   #mask = cv2.inRange(hsv, low_white, up_white)
   edges = cv2.Canny(mask, 75, 150)
   lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=10)
   if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

   cv2.imshow("frame", frame)
   cv2.imshow("edges", edges)

   key = cv2.waitKey(1)
   if key == 27:
       break
video.release()
cv2.destroyAllWindows()
