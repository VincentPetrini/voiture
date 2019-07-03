import cv2 as cv
import numpy as np
from pprint import pprint

videoName = "road_car_view.mp4"
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
low_R = 0
low_G = 0
low_B = 0
high_R = max_value
high_G = max_value
high_B = max_value
window_detection_name = 'Object Detection'
window_line_name = 'Lines Detected'
low_R_name = 'Low R'
low_G_name = 'Low G'
low_B_name = 'Low B'
high_R_name = 'High R'
high_G_name = 'High G'
high_B_name = 'High B'
gapLine_name = "Gap Line"
hThreshold_name = "H threshold"
vitesseVideo_name = "V Speed"
horizon_name = "Horizon"
gapHorizon_name = "Gap H"
gapVertical_name = "Gap V"
lineLength_name = "Line length"

def on_low_R_thresh_trackbar(val):
    global low_R
    global high_R
    low_R = val
    low_R = min(high_R-1, low_R)
    cv.setTrackbarPos(low_R_name, window_detection_name, low_R)
def on_high_R_thresh_trackbar(val):
    global low_R
    global high_R
    high_R = val
    high_R = max(high_R, low_R+1)
    cv.setTrackbarPos(high_R_name, window_detection_name, high_R)
def on_low_G_thresh_trackbar(val):
    global low_G
    global high_G
    low_G = val
    low_G = min(high_G-1, low_G)
    cv.setTrackbarPos(low_G_name, window_detection_name, low_G)
def on_high_G_thresh_trackbar(val):
    global low_G
    global high_G
    high_G = val
    high_G = max(high_G, low_G+1)
    cv.setTrackbarPos(high_G_name, window_detection_name, high_G)
def on_low_B_thresh_trackbar(val):
    global low_B
    global high_B
    low_B = val
    low_B = min(high_B-1, low_B)
    cv.setTrackbarPos(low_B_name, window_detection_name, low_B)
def on_high_B_thresh_trackbar(val):
    global low_B
    global high_B
    high_B = val
    high_B = max(high_B, low_B+1)
    cv.setTrackbarPos(high_B_name, window_detection_name, high_B)

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
cv.createTrackbar(low_R_name, window_detection_name , low_R, max_value, on_low_R_thresh_trackbar)
cv.createTrackbar(high_R_name, window_detection_name , high_R, max_value, on_high_R_thresh_trackbar)
cv.createTrackbar(low_G_name, window_detection_name , low_G, max_value, on_low_G_thresh_trackbar)
cv.createTrackbar(high_G_name, window_detection_name , high_G, max_value, on_high_G_thresh_trackbar)
cv.createTrackbar(low_B_name, window_detection_name , low_B, max_value, on_low_B_thresh_trackbar)
cv.createTrackbar(high_B_name, window_detection_name , high_B, max_value, on_high_B_thresh_trackbar)
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
   frame_threshold = cv.inRange(hsv, (low_R, low_G, low_B), (high_R, high_G, high_B))
   
   low_yellow = np.array([low_R, low_G, low_B])
   up_yellow = np.array([high_R, high_G, high_B])
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
