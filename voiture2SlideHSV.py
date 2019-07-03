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
lineLength = minLineLength
minGapHorizon = 0
maxGapHorizon = 400
minGapVertical = 0
maxGapVertical = 400
gapVertical = minGapVertical
gapHorizon = maxGapHorizon
minHorizonLine = 0
maxHorizonLine = heightVideo
horizon = minHorizonLine
minGapLine = 2
maxGapLine = 300
minHthreshold = 2
maxHthreshold =100
hThreshold = minHthreshold
lineGap = minGapLine
minVitesseVideo = 1
maxVitesseVideo = 200
vitesseVideo = maxVitesseVideo
max_value = 255
max_value_H = 360
low_H = 0
low_S = 0
low_V = 0
high_H = max_value
high_S = max_value
high_V = max_value
window_detection_name = 'Object Detection'
window_line_name = 'Lines Detected'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'
gapLine_name = 'Gap Line'
hThreshold_name = 'H threshold'
vitesseVideo_name = 'V Speed'
horizon_name = 'Horizon'
gapHorizon_name = 'Gap H'
gapVertical_name = 'Gap V'
lineLength_name = 'Line length'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)

def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)

def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)

def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)

def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)

def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_line_gap_thresh_trackbar(val):
    global lineGap
    lineGap = val
    cv.setTrackbarPos(gapLine_name, window_detection_name, lineGap)

def on_h_threshold_thresh_trackbar(val):
    global hThreshold
    hThreshold = val
    cv.setTrackbarPos(hThreshold_name, window_detection_name, hThreshold)

def on_vitesse_thresh_trackbar(val):
    global vitesseVideo
    vitesseVideo = val
    cv.setTrackbarPos(vitesseVideo_name, window_detection_name, vitesseVideo)

def on_horizon_trackbar(val):
    global horizon
    horizon = val
    cv.setTrackbarPos(horizon_name, window_detection_name, horizon)

def on_gap_horizon_trackbar(val):
    global gapHorizon
    gapHorizon = val
    cv.setTrackbarPos(gapHorizon_name, window_detection_name, gapHorizon)

def on_gap_vertical_trackbar(val):
    global gapVertical
    gapVertical = val
    cv.setTrackbarPos(gapVertical_name, window_detection_name, gapVertical)

def on_gap_line_length_trackbar(val):
    global lineLength
    lineLength = val
    cv.setTrackbarPos(lineLength_name, window_detection_name, lineLength)
    
#parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
#parser.add_argument('--camera', help='Camera devide number.', default=0, type=int)
#args = parser.parse_args()
video = cv.VideoCapture(videoName)
#cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)
cv.createTrackbar(gapLine_name, window_detection_name , minGapLine, maxGapLine, on_line_gap_thresh_trackbar)
cv.createTrackbar(hThreshold_name, window_detection_name , minHthreshold, maxHthreshold, on_h_threshold_thresh_trackbar)
cv.createTrackbar(vitesseVideo_name, window_detection_name , minVitesseVideo, maxVitesseVideo, on_vitesse_thresh_trackbar)
cv.createTrackbar(horizon_name, window_detection_name , minHorizonLine, maxHorizonLine, on_horizon_trackbar)
cv.createTrackbar(gapHorizon_name, window_detection_name , minGapHorizon, maxGapHorizon, on_gap_horizon_trackbar)
cv.createTrackbar(gapVertical_name, window_detection_name , minGapVertical, maxGapVertical, on_gap_vertical_trackbar)
cv.createTrackbar(lineLength_name, window_detection_name , minLineLength, maxLineLength, on_gap_line_length_trackbar)

while True:
   ret, orig_frame = video.read()
   if not ret:
       video = cv.VideoCapture(videoName)
       continue
   frame = cv.GaussianBlur(orig_frame, (7, 7), 0)
   hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
   frame_threshold = cv.inRange(hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))
   
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
          if abs(y1 - y2) > gapHorizon and abs(x1 - x2) > gapVertical :
          #if abs(x1 - x2) > gapVertical :
              cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
          #print(line)
          
   cv.line(frame, (0, horizon), (widthVideo, horizon), (0, 0, 255), 5)

   cv.imshow(window_line_name, frame)
   cv.imshow(window_detection_name, frame_threshold)
   cv.imshow("lol", frame_threshold)
   cv.imshow("edges", edges)

   key = cv.waitKey(vitesseVideo)
   if key == 27:
       break
video.release()
cv.destroyAllWindows()
