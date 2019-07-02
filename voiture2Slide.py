import cv2 as cv
import numpy as np

videoName = "road_car_view4.mp4"
video = cv.VideoCapture(videoName)

minGapLine = 2
maxGapLine = 300
lineGap = minGapLine
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
    lineGap = val
    cv.setTrackbarPos(gapLine_name, window_detection_name, lineGap)
    
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
   #mask = cv.inRange(hsv, low_white, up_white)
   edges = cv.Canny(mask, 75, 150)
   lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=lineGap)
   if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

   cv.imshow(window_line_name, frame)
   cv.imshow(window_detection_name, frame_threshold)
   cv.imshow("edges", edges)

   key = cv.waitKey(145)
   if key == 27:
       break
video.release()
cv.destroyAllWindows()
