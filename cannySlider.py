# script for tuning parameters
import cv2
import numpy as np
import argparse

# parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

# reads the image
img = cv2.imread(args['image'])
    
# empty callback function for creating trackar
def callback(foo):
    pass

# create windows and trackbar
cv2.namedWindow('parameters')
cv2.createTrackbar('threshold1', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
cv2.createTrackbar('threshold2', 'parameters', 0, 255, callback)  # change the maximum to whatever you like
cv2.createTrackbar('apertureSize', 'parameters', 0,2, callback)
cv2.createTrackbar('L1/L2', 'parameters', 0, 1, callback)
cv2.createTrackbar('LoR', 'parameters', 0, 255, callback)
cv2.createTrackbar('LoG', 'parameters', 0, 255, callback)
cv2.createTrackbar('LoB', 'parameters', 0, 255, callback)
cv2.createTrackbar('UpR', 'parameters', 0, 255, callback)
cv2.createTrackbar('UpG', 'parameters', 0, 255, callback)
cv2.createTrackbar('UpB', 'parameters', 0, 255, callback)

while(True):
    # get threshold value from trackbar
    th1 = cv2.getTrackbarPos('threshold1', 'parameters')
    th2 = cv2.getTrackbarPos('threshold2', 'parameters')
    
    # aperture size can only be 3,5, or 7
    apSize = cv2.getTrackbarPos('apertureSize', 'parameters')*2+3
    
    # true or false for the norm flag
    norm_flag = cv2.getTrackbarPos('L1/L2', 'parameters') == 1

    # Get color values
    LoR = cv2.getTrackbarPos('LoR', 'parameters')
    LoG = cv2.getTrackbarPos('LoG', 'parameters')
    LoB = cv2.getTrackbarPos('LoB', 'parameters')
    UpR = cv2.getTrackbarPos('UpR', 'parameters')
    UpG = cv2.getTrackbarPos('UpG', 'parameters')
    UpB = cv2.getTrackbarPos('UpB', 'parameters')
    
    # print out the values
    print('')
    print('threshold1: {}'.format(th1))
    print('threshold2: {}'.format(th2))
    print('apertureSize: {}'.format(apSize))
    print('L2gradient: {}'.format(norm_flag))

    frame = cv2.GaussianBlur(img, (7, 7), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_white = np.array([LoR, LoG, LoB])
    up_white = np.array([UpR, UpG, UpB])
    mask = cv2.inRange(hsv, low_white, up_white)
    #edges = cv2.Canny(mask, 75, 150)
    edge = cv2.Canny(mask, th1, th2, apertureSize=apSize, L2gradient=norm_flag)
    cv2.imshow('canny', edge)
    
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
        
cv2.destroyAllWindows()
