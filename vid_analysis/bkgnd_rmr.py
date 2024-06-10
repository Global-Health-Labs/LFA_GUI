#!/usr/bin/env python3

import cv2
import numpy as np
import os
 
# Open Video
video_file = '20240531_113642.158.mp4'
cap = cv2.VideoCapture(video_file)
 
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
 

# Create output directory if it doesn't exist
new_dir = os.path.splitext(video_file)[0]
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Create variable to write video output
output_video_path = os.path.join(new_dir, 'output.avi')
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (int(cap.get(3)), int(cap.get(4))), False)

# Loop over all frames
while True:

  # Read frame
  ret, frame = cap.read()
  if not ret:
     break
  
  # Convert current frame to grayscale
  gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # Calculate absolute difference of current frame and the median frame
  dframe = cv2.absdiff(gray_frame, grayMedianFrame)
  
  # Display image
  # cv2.imshow('frame', dframe)
  # cv2.waitKey(20)
  # Write Image
  out.write(dframe)

# Release the video capture object
cap.release()
out.release()

#### GETTING THE LAST FRAME
cap2 = cv2.VideoCapture(output_video_path)

# Extract last frame
last_frame = None

while cap2.isOpened():
   ret, frame = cap2.read()
   if not ret:
      break
   last_frame = frame

# Save the last frame
if last_frame is not None:
    last_frame_pic = os.path.join(new_dir, 'last_frame.jpg')
    cv2.imwrite(last_frame_pic, last_frame)

# Release the output
cap2.release()

#### DRAWING CONTOURS 
# Read the image
cap3 = cv2.imread(last_frame_pic)

# convert the image to grayscale format
img_gray = cv2.cvtColor(cap3, cv2.COLOR_BGR2GRAY)

# Blur the image
img_blurred = cv2.GaussianBlur(img_gray,(3,3),0)

# Treshold to binarize
_, img_binarized = cv2.threshold(img_blurred, 30, 255, cv2.THRESH_BINARY)

# Detect the contours on the binary image
contours, _ = cv2.findContours(image=img_binarized, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image for `CHAIN_APPROX_SIMPLE`
cv2.drawContours(cap3, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

## Insert bounding rectangles
# Starting position for the first rectangle
x_start = 128
y_start = 309
x_end = 178
y_end = 410
# Number of rectangles
num_rectangles = 8
# Spacing between rectangles
spacing = 234
for i in range(num_rectangles):
    # Calculate the position of the current rectangle
    x_offset = i * spacing
    cv2.rectangle(cap3, (x_start + x_offset, y_end), (x_end + x_offset, y_start), (0, 0, 255), 3)

cv2.imwrite(f'./{new_dir}/contours.jpg', cap3)

cv2.destroyAllWindows()