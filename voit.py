import cv2 as cv
import numpy as np

videoName = "road_car_view4.mp4"
video = cv.VideoCapture(videoName)

widthVideo = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
heightVideo = int(video.get(cv.CAP_PROP_FRAME_HEIGHT ))
print(widthVideo)
print(heightVideo)
minLineLength = 0
maxLineLength = 300
lineLength = 9

minGapHorizon = 0
maxGapHorizon = 400
gapHorizon = 84

minGapVertical = 0
maxGapVertical = 400
gapVertical = 90



minHorizonLine = 0
maxHorizonLine = heightVideo
horizon = 119

minGapLine = 2
maxGapLine = 300
lineGap = 109

minHthreshold = 2
maxHthreshold = 100
hThreshold = 60

minVitesseVideo = 1
maxVitesseVideo = 200
vitesseVideo = maxVitesseVideo

max_value = 255
max_value_H = 360
low_H = 0
low_S = 0
low_V = 104
high_H = max_value
high_S = max_value
high_V = 230
window_detection_name = 'Object Detection'
window_line_name = 'Lines Detected'

arrayLeftLines = []
arrayRightLines = []

def averageLine(arrayLines):
  count = 0
  X1 = 0
  X2 = 0
  Y1 = 0
  Y2 = 0
  resu = np.zeros(4)
  for lines in arrayLines:
      x1, y1, x2, y2 = lines[0]
      X1 = X1 + x1
      X2 = X2 + x2
      Y1 = Y1 + y1
      Y2 = Y2 + y2
      count = count + 1
      if count != 0:
          resu = np.array([int(X1/count), int(Y1/count), int(X2/count), int(Y2/count)])
  return (resu)

video = cv.VideoCapture(videoName)
while True:
   ret, orig_frame = video.read()
   if not ret:
       video = cv.VideoCapture(videoName)
       continue
   frame = cv.GaussianBlur(orig_frame, (7, 7), 0)
   hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)   
   frame_threshold = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
   roadMap = cv.inRange(hsv, (0, 0, 0), (0, 0, 0))
   
   low_yellow = np.array([low_H, low_S, low_V])
   up_yellow = np.array([high_H, high_S, high_V])
   mask = cv.inRange(hsv, low_yellow, up_yellow)
   ## Make pixels row and column 300-400 black
   mask[0:horizon,0:widthVideo] = (0)
   #mask = cv.inRange(hsv, low_white, up_white)
   edges = cv.Canny(mask, 75, 150)
   lines = cv.HoughLinesP(edges, 1, np.pi/180, hThreshold, minLineLength=lineLength, maxLineGap=lineGap)
   if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          #print(line[0])
          if abs(y1 - y2) > gapHorizon and abs(x1 - x2) > gapVertical :
              if y1  > y2 :
                  cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                  arrayLeftLines.append(line)
              elif y1 < y2 :
                  cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                  arrayRightLines.append(line)
          #print(line)

   """
   print("stop")
   print("left")
   print(arrayLeftLines)
   print("Moy Left")
   print(averageLine(arrayLeftLines))
   print("right")
   print(arrayRightLines)
   print("Moy Right")
   print(averageLine(arrayRightLines))
   #Suppression tableau de lignes
   arrayLeftLines.clear()
   arrayRightLines.clear()
   """
   if  np.count_nonzero(averageLine(arrayLeftLines)) != 0 :
       x1, y1, x2, y2 = averageLine(arrayLeftLines)
       cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

   if  np.count_nonzero(averageLine(arrayRightLines)) != 0 :
       x1, y1, x2, y2 = averageLine(arrayRightLines)
       cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
   
   arrayLeftLines.clear()
   arrayRightLines.clear()
   #ligne d'horizon
   #cv.line(frame, (0, horizon), (widthVideo, horizon), (0, 0, 255), 5)

   cv.imshow(window_line_name, frame)
   #cv.imshow(window_detection_name, frame_threshold)
   #cv.imshow("lol", frame_threshold)
   #cv.imshow("edges", edges)

   cv.line(roadMap, (int((widthVideo/3)), heightVideo), (int(widthVideo/3), 0), (255, 255, 255), 5)
   cv.line(roadMap, (int((widthVideo/3))*2, heightVideo), (int(widthVideo/3)*2, 0), (255, 255, 255), 5)
   cv.imshow("Map", roadMap)

   key = cv.waitKey(vitesseVideo)
   if key == 27:
       break
video.release()
cv.destroyAllWindows()
