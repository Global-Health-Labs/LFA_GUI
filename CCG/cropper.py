#!/usr/bin/env python3

import cv2
import os
import argparse

def crop_image_vertically(image_path, output_dir):
    # Extract the base name of the input file (without extension)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape

    # Calculate the width of each vertical portion
    portion_width = image_width // 8

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through and save each portion
    for i in range(8):
        # Calculate the bounding box of the current portion
        left = i * portion_width
        right = (i + 1) * portion_width if i < 7 else image_width
        
        # Crop the image to get a vertical strip
        cropped_image = image[0:image_height, left:right]
        
        # Save the cropped image with a dynamic name based on the input file name
        output_path = os.path.join(output_dir, f"{base_name}_portion_{i + 1}.tif")
        cv2.imwrite(output_path, cropped_image)
        #print(f"Saved: {output_path}")

def process_images_in_directory(input_dir, output_base_dir):
    # Ensure the input directory exists
    if not os.path.isdir(input_dir):
        print(f"The directory {input_dir} does not exist.")
        return

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # Construct the full path to the image file
        image_path = os.path.join(input_dir, filename)
        
        # Ensure the file is an image (you can add more extensions if needed)
        if filename.lower().endswith(('.tif', '.jpg', '.png')):
            # Set output directory for cropped images
            output_dir = os.path.join(output_base_dir, os.path.splitext(filename)[0])
            
            # Crop and save portions of the image
            crop_image_vertically(image_path, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop images vertically and save portions.')
    parser.add_argument('input_dir', type=str, help='Directory containing input images')
    parser.add_argument('output_dir', type=str, help='Base directory for saving cropped images')

    args = parser.parse_args()

    process_images_in_directory(args.input_dir, args.output_dir)