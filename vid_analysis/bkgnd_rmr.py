#!/usr/bin/env python3

import cv2
import numpy as np
from skimage import data, filters
 
# Open Video
cap = cv2.VideoCapture('20240531_113642.158.mp4')
 
# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate the number of frames in the first 30 seconds
num_frames_in_30_seconds = int(fps * 30)

# Randomly select 25 frames from the first 30 seconds
frameIds = np.random.choice(range(num_frames_in_30_seconds), size=25, replace=False)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Reset frame number to 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
 
# Convert background to grayscale
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
 
# Create variable to write video output
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (int(cap.get(3)), int(cap.get(4))), False)

# Loop over all frames
ret = True
while(ret):
 
  # Read frame
  ret, frame = cap.read()
  if not ret:
     break
  
  # Convert current frame to grayscale
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Calculate absolute difference of current frame and the median frame
  dframe = cv2.absdiff(gray_frame, grayMedianFrame)
  # Blur the frame
  #blurred_dframe = cv2.GaussianBlur(dframe,(3,3),0)
  # Treshold to binarize
  #th, binarized_dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
  # Display image
  # cv2.imshow('frame', dframe)
  # cv2.waitKey(20)
  # Write Image
  out.write(dframe)

# Release the video capture object
cap.release()
out.release()
cv2.destroyAllWindows()