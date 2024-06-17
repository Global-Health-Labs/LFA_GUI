#!/usr/bin/env python3

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter,find_peaks
 
# Open Video
video_file = '20240531_114934.240.mp4'
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
# Read image
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

# Save image
contours_pic = os.path.join(new_dir, 'contours.jpg')
cv2.imwrite(contours_pic, cap3)

#### INSERT BOUNDING RECTANGLES
# Read the image
cap4 = cv2.imread(contours_pic)

# Starting position for the first rectangle
x_start = 128
y_start = 309
x_end = 178
y_end = 410
# Number of rectangles
num_rectangles = 8
# Spacing between rectangles
spacing = 234

# Create a figure for histograms
plt.figure(figsize=(15, 10))

# Function to find LFA peaks
def find_lfa_peaks(line_profile, top=0):
    N = 3  # Assuming 3 lines for this example; adjust based on your needs
    filtered = savgol_filter(line_profile, 3, 2)
    lowest_length = np.clip(len(filtered) // 2, 1, 50) - 1
    lowest = np.sort(filtered)[0:lowest_length]
    background = np.mean(lowest)
    peaks_X, _ = find_peaks(filtered)
    peaks_Y = filtered[peaks_X]
    
    # Generate line_vals and interval_vals
    line_vals = [top + 25, top + 50, top + 80]
    interval_vals = [20, 20, 20]
    
    X_intervals = [[int(line - interval - top), int(line + interval - top)] for line, interval in zip(line_vals, interval_vals)]
    peaks_XY_max = [max([[X, Y] for (X, Y) in zip(peaks_X, peaks_Y) if X >= a and X <= b], key=lambda x: x[1], default=[None, None]) for a, b in X_intervals]
    peaks_XY_max.append([None, background])
    
    peaks_X_by_location, peaks_Y_by_location = zip(*peaks_XY_max)
    return filtered, list(peaks_X_by_location), list(peaks_Y_by_location)

for i in range(num_rectangles):
    # Calculate the position of the current rectangle
    x_offset = i * spacing
    # Draw the rectangle
    cv2.rectangle(cap4, (x_start + x_offset, y_start), (x_end + x_offset, y_end), (0, 0, 255), 1)
    # Extract the region of interest (ROI)
    roi = last_frame[y_start:y_end, x_start + x_offset:x_end + x_offset]
    # Convert ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Calculate the mean pixel intensity along the vertical axis
    line_profile = np.mean(roi_gray, axis=1)
    # Find peaks in the line profile
    filtered, peaks_X, peaks_Y = find_lfa_peaks(line_profile, top=y_start)
    
    # Plot the line profile and detected peaks
    #plt.subplot(num_rectangles, 1, i + 1)
    plt.plot(filtered, label='Filtered Line Profile')
    plt.scatter(peaks_X, peaks_Y, color='red', label='Peaks')
    plt.title(f'Rectangle {i + 1}')
    plt.xlabel('Pixel')
    plt.ylabel('Intensity')
    #plt.legend()

    # Save the plot
    plot_filename = os.path.join(new_dir, f'rectangle_{i + 1}.png')
    plt.savefig(plot_filename)
    plt.close()


# # Display the plots
# plt.tight_layout()
# plt.show()

# Save image with rectangles
rect_pic = os.path.join(new_dir, 'rect.jpg')
cv2.imwrite(rect_pic, cap4)

cv2.destroyAllWindows()